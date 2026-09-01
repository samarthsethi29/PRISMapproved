"""
Step 4: train the PRISM multi-task GNN on ABL1 / c-KIT / PDGFRB.

Reads : data/train.csv, data/val.csv, data/test.csv   (Step 3 outputs)
Uses  : model.py  (your MultiTargetGNN + MaskedMultiTaskLoss, unchanged)
Writes: models/best_model.pt, models/thresholds.json,
        models/test_metrics.json, models/train_log.json

Hyperparameters are the published PRISM v1 settings; only max-epochs /
patience raised (we now have ~10x more data than v1). CPU is fine;
Colab T4 is ~5x faster.      python step4_train.py
"""
import json, os, random, time
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem, RDLogger
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef)
from model import MultiTargetGNN, MaskedMultiTaskLoss

RDLogger.DisableLog("rdApp.*")

# ── config (published v1 settings) ─────────────────────────────────────────
TARGETS   = ["ABL1", "c-KIT", "PDGFRB"]
SEED      = 42
HIDDEN, LAYERS, DROPOUT = 256, 4, 0.2
LR, WD, BATCH, CLIP = 1e-3, 1e-5, 64, 1.0
MAX_EPOCHS, PATIENCE, T_MAX = 150, 20, 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOND_TYPES  = [Chem.BondType.SINGLE,  Chem.BondType.DOUBLE,
               Chem.BondType.TRIPLE,  Chem.BondType.AROMATIC]
BOND_STEREO = [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOZ,
               Chem.BondStereo.STEREOE,    Chem.BondStereo.STEREOANY]

# ── featurisation: 7-dim atoms, 8-dim edges (same spec as PRISM v1) ────────
def atom_features(a):
    return [float(a.GetAtomicNum()),     # 1
            float(a.GetChiralTag()),     # 2
            float(a.GetFormalCharge()),  # 3
            float(a.IsInRing()),         # 4
            float(a.GetIsAromatic()),    # 5
            float(a.GetDegree()),        # 6
            float(a.GetTotalNumHs())]    # 7

def bond_features(b):
    return ([1.0 if b.GetBondType() == t else 0.0 for t in BOND_TYPES] +
            [1.0 if b.GetStereo()   == s else 0.0 for s in BOND_STEREO])

def mol_to_graph(smiles, labels):
    """labels: K values in {1, 0, -1}. Stored as [1, K] so PyG batching
    yields batch.y of shape [B, K]."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumBonds() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()],
                     dtype=torch.float)
    ei, ef = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        ei += [[i, j], [j, i]]
        ef += [f, f]
    return Data(x=x,
                edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(ef, dtype=torch.float),
                y=torch.tensor([labels], dtype=torch.float))

def load_split(path):
    df = pd.read_csv(path)
    graphs = []
    for _, r in df.iterrows():
        g = mol_to_graph(r["smiles"], [float(r[t]) for t in TARGETS])
        if g is not None:
            g.mol_id = str(r["molecule_chembl_id"])
            graphs.append(g)
    print(f"  {os.path.basename(path):9s}: {len(graphs):5d} graphs "
          f"({len(df) - len(graphs)} unparseable dropped)")
    return df, graphs

def pos_weights_from(df):
    """BCE pos_weight = n_negative / n_positive per task, computed on TRAIN."""
    ws = []
    for t in TARGETS:
        v = df.loc[df[t] != -1, t]
        ws.append(float((v == 0).sum()) / max(float((v == 1).sum()), 1.0))
    return torch.tensor(ws, dtype=torch.float)

@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    ys = {t: [] for t in TARGETS}
    ps = {t: [] for t in TARGETS}
    for batch in loader:
        batch = batch.to(device)
        prob = torch.sigmoid(model(batch)).cpu().numpy()   # [B, K]
        y = batch.y.cpu().numpy()                           # [B, K]
        for k, t in enumerate(TARGETS):
            ys[t].append(y[:, k]); ps[t].append(prob[:, k])
    return ({t: np.concatenate(ys[t]) for t in TARGETS},
            {t: np.concatenate(ps[t]) for t in TARGETS})

def best_threshold(y, p):
    best_thr, best_mcc = 0.5, -1.0
    for thr in np.arange(0.05, 0.951, 0.01):
        mcc = matthews_corrcoef(y, (p >= thr).astype(int))
        if mcc > best_mcc:
            best_thr, best_mcc = float(thr), float(mcc)
    return best_thr, best_mcc

def metrics_for(y, p, thr):
    m = {"n_labeled": int(len(y)),
         "n_active":  int((y == 1).sum()),
         "n_inactive": int((y == 0).sum())}
    if m["n_active"] and m["n_inactive"]:
        m["AUC"]   = float(roc_auc_score(y, p))
        m["AUPRC"] = float(average_precision_score(y, p))
        m["MCC"]   = float(matthews_corrcoef(y, (p >= thr).astype(int)))
    else:
        m["AUC"] = m["AUPRC"] = m["MCC"] = None
    return m

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs("models", exist_ok=True)
    print(f"device: {DEVICE}\nloading splits:")
    df_tr, g_tr = load_split("data/train.csv")
    df_va, g_va = load_split("data/val.csv")
    df_te, g_te = load_split("data/test.csv")

    pw = pos_weights_from(df_tr)
    print("pos_weights (neg/pos, from TRAIN):",
          {t: round(w, 3) for t, w in zip(TARGETS, pw.tolist())})

    loader_tr = DataLoader(g_tr, batch_size=BATCH, shuffle=True)
    loader_va = DataLoader(g_va, batch_size=BATCH)
    loader_te = DataLoader(g_te, batch_size=BATCH)

    model = MultiTargetGNN(hidden_dim=HIDDEN, num_layers=LAYERS,
                           dropout=DROPOUT, target_names=TARGETS).to(DEVICE)
    print(f"parameters: {model.count_parameters():,}")

    loss_fn = MaskedMultiTaskLoss(pw.to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)

    best, best_ep, wait, log = -1.0, -1, 0, []
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); tot, nb = 0.0, 0
        for batch in loader_tr:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            tot += float(loss); nb += 1
        sched.step()

        y_v, p_v = collect(model, loader_va, DEVICE)
        auprcs = []
        for t in TARGETS:
            m = y_v[t] != -1
            if (y_v[t][m] == 1).any() and (y_v[t][m] == 0).any():
                auprcs.append(average_precision_score(y_v[t][m], p_v[t][m]))
        va_avg = float(np.mean(auprcs)) if auprcs else 0.0
        log.append({"epoch": ep, "train_loss": tot / max(nb, 1),
                    "val_avg_auprc": va_avg})
        if ep == 1 or ep % 5 == 0:
            print(f"  epoch {ep:3d} | train_loss {tot/max(nb,1):8.4f} "
                  f"| val avg AUPRC {va_avg:.4f}")
        if va_avg > best:
            best, best_ep, wait = va_avg, ep, 0
            torch.save({"state_dict": model.state_dict(), "targets": TARGETS,
                        "hidden_dim": HIDDEN, "num_layers": LAYERS,
                        "dropout": DROPOUT, "pos_weights": pw.tolist()},
                       "models/best_model.pt")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  early stop (no val-AUPRC gain for {PATIENCE} epochs)")
                break
    print(f"done in {time.time()-t0:.0f}s | best val AUPRC {best:.4f} "
          f"@ epoch {best_ep}")

    ckpt = torch.load("models/best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    # MCC-maximising thresholds, locked on val before touching test
    y_v, p_v = collect(model, loader_va, DEVICE)
    thresholds = {}
    for t in TARGETS:
        m = y_v[t] != -1
        thresholds[t], mcc = best_threshold(y_v[t][m], p_v[t][m])
        print(f"  {t}: threshold {thresholds[t]:.2f} (val MCC {mcc:.3f})")

    y_t, p_t = collect(model, loader_te, DEVICE)
    test_metrics = {t: metrics_for(y_t[t][y_t[t] != -1],
                                   p_t[t][y_t[t] != -1],
                                   thresholds[t]) for t in TARGETS}
    print("\n===== SCAFFOLD-SPLIT TEST METRICS =====")
    print(f"{'target':8s} {'n_lab':>6s} {'AUC':>7s} {'AUPRC':>7s} "
          f"{'MCC':>7s} {'thr':>5s}")
    for t in TARGETS:
        m = test_metrics[t]
        print(f"{t:8s} {m['n_labeled']:6d} {m['AUC']:7.3f} {m['AUPRC']:7.3f} "
              f"{m['MCC']:7.3f} {thresholds[t]:5.2f}")
    aucs = [m["AUC"] for m in test_metrics.values() if m["AUC"] is not None]
    if aucs:
        print(f"average AUC: {np.mean(aucs):.3f}")

    json.dump(thresholds, open("models/thresholds.json", "w"), indent=2)
    json.dump({"config": {"targets": TARGETS, "hidden_dim": HIDDEN,
                          "num_layers": LAYERS, "dropout": DROPOUT, "lr": LR,
                          "weight_decay": WD, "batch_size": BATCH,
                          "pos_weights": pw.tolist()},
               "best_epoch": best_ep, "best_val_avg_auprc": best,
               "runtime_s": round(time.time() - t0, 1),
               "thresholds": thresholds, "test_metrics": test_metrics},
              open("models/test_metrics.json", "w"), indent=2)
    json.dump(log, open("models/train_log.json", "w"), indent=2)
    print("\nwrote models/best_model.pt, thresholds.json, "
          "test_metrics.json, train_log.json")

if __name__ == "__main__":
    main()