"""
paper_assets.py — PRISM v3 (binary classifier): complete paper figure, table
and report generator. Run inside the v3 directory (PRISM_p2_).

Recomputes every reported number deterministically from frozen primary
artefacts (models/best_model.pt + data/*) so figures, tables and paper text
cannot drift. Inference = sigmoid(logits), as in the v3 pipeline.
Writes ONLY to outputs/paper/ — existing outputs are never overwritten.
Frozen (hardcoded) numbers are only ones that are supposed to be fixed:
the pre-specified gold-standard reference affinities and the v3 external
numbers already recorded from your step-6 runs.
"""
import glob, json, os, re, traceback
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, roc_curve,
                             precision_recall_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MultiTargetGNN
from step4_train import mol_to_graph, TARGETS

RDLogger.DisableLog("rdApp.*")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIVE_PACT, SIM_THRESHOLD = 6.0, 0.30
OUT, FIGS = "outputs/paper", "outputs/paper/figs"
os.makedirs(FIGS, exist_ok=True)
REPORT, S = {}, {}

GT = {"imatinib":  ("CHEMBL941",     (6.658, 6.685, 6.517)),
      "dasatinib": ("CHEMBL5416410", (8.509, 7.983, 7.553)),
      "nilotinib": ("CHEMBL255863",  (7.553, 6.801, 7.185)),
      "ponatinib": ("CHEMBL1171837", (9.432, 8.770, 8.921))}
KINASE   = ["sunitinib", "sorafenib", "pazopanib", "regorafenib", "lenvatinib",
            "cabozantinib", "midostaurin", "axitinib", "bosutinib", "vandetanib"]
CONTROLS = ["metformin", "atorvastatin", "warfarin", "ibuprofen", "amoxicillin",
            "oseltamivir", "fluoxetine", "loratadine"]
BDB = {"ABL1":  ("data/bindingdb/ABL1.tsv",  "P00519"),
       "c-KIT": ("data/bindingdb/KIT.tsv",   "P10721"),
       "PDGFRB": ("data/bindingdb/PDGFRB.tsv", "P09619")}
V3_FROZEN = {"ABL1":  {"AUC": 0.627, "rho": 0.443, "n": 682,  "CI": "0.55-0.70"},
             "c-KIT": {"AUC": 0.873, "rho": 0.106, "n": 1175, "CI": "0.79-0.94"},
             "PDGFRB": {"AUC": 0.719, "rho": 0.679, "n": 1165, "CI": "0.58-0.83"}}
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
MUT = r"mutant|mutation|[A-Za-z]\d{3}[A-Za-z]}"

def save(fig, name):
    fig.savefig(f"{FIGS}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{FIGS}/{name}.pdf", bbox_inches="tight")
    plt.close(fig); print(f"  + figs/{name}.png|.pdf")

def find_col(cols, *keys):
    for c in cols:
        if all(k in str(c).lower() for k in keys): return c

def parse_aff(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return None, None
    s = str(x).strip()
    if not s or s.lower() in ("-", "nan", "none", "n/a"): return None, None
    q = s[0] if s[0] in "<>" else None
    if q: s = s[1:]
    m = re.search(NUM, s)
    return (q, float(m.group())) if m else (None, None)

def to_pact(v): return -np.log10(v * 1e-9)

def largest_fragment(smi):
    if not isinstance(smi, str) or not smi.strip(): return None
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    fr = Chem.GetMolFrags(m, asMols=True)
    return Chem.MolToSmiles(max(fr, key=lambda f: f.GetNumAtoms())) if fr else None

def skel_of(smi):
    if not isinstance(smi, str) or not smi.strip(): return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m).split("-")[0] if m else None

def scaffold_of(s):
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
        return sc if sc else s
    except Exception:
        return s

def phase(title, fn):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
    try:
        fn()
    except Exception:
        traceback.print_exc()
        print("!! PHASE FAILED — continuing; paste this with the console")

# ── PHASE 0: inventory ──────────────────────────────────────────────────────
def _inventory():
    for d in ("data", "data/bindingdb", "models", "outputs"):
        print(f"  [{d}]")
        fs = sorted(f for f in glob.glob(f"{d}/*") if os.path.isfile(f))
        for f in fs: print(f"    {f:52s} {os.path.getsize(f)/1e3:8.1f} kB")
        if not fs: print("    (empty or missing)")

# ── PHASE 1: gold standard + labels ─────────────────────────────────────────
def _labels():
    if os.path.exists("data/manifest.json"):
        print("  manifest.json:")
        print("  " + json.dumps(json.load(open("data/manifest.json")),
                               indent=2).replace("\n", "\n  "))
    mat = pd.read_csv("data/matrix.csv"); S["mat"] = mat
    lab = {}
    for t in TARGETS:
        v = pd.to_numeric(mat[t], errors="coerce").fillna(-1)
        m = v != -1; vals = v[m]
        binary = set(np.unique(vals)) <= {0.0, 1.0}
        act = int((vals >= (1 if binary else ACTIVE_PACT)).sum())
        lab[t] = {"n_labeled": int(m.sum()), "actives": act,
                  "inactives": int(m.sum() - act),
                  "label_type": ("binary" if binary else
                                 f"continuous pAct (median {vals.median():.2f}, "
                                 f"range {vals.min():.2f}-{vals.max():.2f})")}
        print(f"  {t}: {lab[t]}")
    REPORT["labels"] = lab
    fig, ax = plt.subplots(figsize=(6.5, 3))
    x = np.arange(3)
    act = [lab[t]["actives"] for t in TARGETS]
    ina = [lab[t]["inactives"] for t in TARGETS]
    ax.bar(x, act, 0.55, color="#d65f5f", label="actives (pAct ≥ 6)")
    ax.bar(x, ina, 0.55, bottom=act, color="#4878d0", label="inactives")
    for i, t in enumerate(TARGETS):
        ax.text(i, act[i] + ina[i] + 12, f"n={lab[t]['n_labeled']}",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(TARGETS); ax.set_ylabel("compounds")
    ax.legend(frameon=False, fontsize=8); save(fig, "fig_labels")

    print("\n  LABEL DEFINITION ('active' = measured biochemical potency):")
    print("    • assay_type B (biochemical binding/inhibition) only — no cell/functional assays")
    print("    • wild-type human targets (mutant-construct assays excluded; organism human)")
    print("    • exact '=' Ki/IC50 in nM, 0.01–10^7 nM accepted   [CONFIRM censored handling vs step-7 console]")
    print("    • replicates → median pAct per compound; deduplicated by InChIKey skeleton")
    print("    • active ⇔ pAct ≥ 6 ⇔ Ki/IC50 ≤ 1 µM              [CONFIRM grey-zone handling vs step-7 console]")
    print("    • approved drugs excluded from training (ChEMBL ID + skeleton)")
    print("\n  GOLD STANDARD — pre-specified, independent of training data:")
    print(f"    {'drug':11s} {'ChEMBL id':14s} {'ABL1':>7s} {'c-KIT':>7s} {'PDGFRB':>7s}  (measured pAct, fixed before screening)")
    for name, (mid, ps) in GT.items():
        print(f"    {name:11s} {mid:14s} " + " ".join(f"{p:7.3f}" for p in ps))
    print(f"    kinase cohort ({len(KINASE)}): " + ", ".join(KINASE))
    print(f"    controls      ({len(CONTROLS)}): " + ", ".join(CONTROLS))
    print("    temporal cutoff: post-2016-01-01 BindingDB WT-human measurements held out of training")
    print("    endpoints: recovery percentile, EF5%, screen AUROC, cohort median gap,")
    print("               external AUC, external Spearman ρ")
    REPORT["gold_standard"] = {n: {"id": m, "pAct": dict(zip(TARGETS, ps))}
                               for n, (m, ps) in GT.items()}

# ── PHASE 2: internal validation ────────────────────────────────────────────
def _internal():
    ck = torch.load("models/best_model.pt", map_location=DEVICE)
    kw = dict(hidden_dim=ck.get("hidden_dim", 256), num_layers=ck.get("num_layers", 4),
              dropout=ck.get("dropout", 0.2))
    names = list(ck.get("targets", TARGETS))
    try:
        model = MultiTargetGNN(target_names=names, task="classification", **kw)
    except TypeError:                       # original v3 model.py: no task kwarg
        model = MultiTargetGNN(target_names=names, **kw)
    model.load_state_dict(ck["state_dict"]); model.to(DEVICE).eval()
    print(f"  checkpoint keys: {sorted(ck.keys())} | "
          f"params {sum(p.numel() for p in model.parameters()):,}")
    S["model"] = model

    def run(path):
        df = pd.read_csv(path); gs = []
        for _, r in df.iterrows():
            g = mol_to_graph(r["smiles"], [float(r[t]) for t in TARGETS])
            if g is not None: gs.append(g)
        ys, ps = [], []
        with torch.no_grad():
            for b in DataLoader(gs, batch_size=256):
                b = b.to(DEVICE)
                ps.append(torch.sigmoid(model(b)).cpu().numpy())
                ys.append(b.y.cpu().numpy())
        return np.vstack(ys), np.vstack(ps)

    y_va, p_va = run("data/val.csv"); y_te, p_te = run("data/test.csv")
    print(f"  val graphs: {len(y_va)} | test graphs: {len(y_te)}")

    def yb_of(v):
        u = set(np.unique(v[v != -1])) if (v != -1).any() else {0.0}
        return (v >= 1.0).astype(int) if u <= {0.0, 1.0} else (v >= ACTIVE_PACT).astype(int)

    internal, curves = {}, {}
    for k, t in enumerate(TARGETS):
        mv, m = (y_va[:, k] != -1), (y_te[:, k] != -1)
        if m.sum() == 0: continue
        yb, pp = yb_of(y_te[m, k]), p_te[m, k]
        rec = {"n": int(m.sum()), "n_active": int(yb.sum()),
               "n_inactive": int(len(yb) - yb.sum())}
        tstar = 0.5
        if mv.sum():
            ybv, pv = yb_of(y_va[mv, k]), p_va[mv, k]
            if 0 < ybv.sum() < len(ybv):
                best = -2.0
                for tt in np.linspace(0.02, 0.98, 97):
                    mcc = matthews_corrcoef(ybv, (pv >= tt).astype(int))
                    if mcc > best: best, tstar = mcc, float(tt)
        rec["threshold_MCC"] = round(tstar, 2)
        if 0 < yb.sum() < len(yb):
            rec["AUC"] = round(float(roc_auc_score(yb, pp)), 3)
            rec["AUPRC"] = round(float(average_precision_score(yb, pp)), 3)
            rec["MCC"] = round(float(matthews_corrcoef(yb, (pp >= tstar).astype(int))), 3)
            curves[t] = (yb, pp)
        internal[t] = rec; print(f"  {t}: {rec}")
    REPORT["internal"] = internal
    pd.DataFrame(internal).T.to_csv(f"{OUT}/internal_metrics.csv", index=False)
    print(f"  wrote {OUT}/internal_metrics.csv")

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for t, (yb, pp) in curves.items():
        fpr, tpr, _ = roc_curve(yb, pp)
        ax.plot(fpr, tpr, lw=1.5, label=f"{t} (AUC {internal[t]['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel="false positive rate", ylabel="true positive rate",
           title="Scaffold-split test — ROC")
    ax.legend(frameon=False, fontsize=8); save(fig, "fig_internal_roc")

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for t, (yb, pp) in curves.items():
        pr, rc, _ = precision_recall_curve(yb, pp)
        ax.plot(rc, pr, lw=1.5, label=f"{t} (AUPRC {internal[t]['AUPRC']:.3f})")
    ax.set(xlabel="recall", ylabel="precision",
           title="Scaffold-split test — precision-recall")
    ax.legend(frameon=False, fontsize=8); save(fig, "fig_internal_pr")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, t in zip(axes, TARGETS):
        if t not in curves: continue
        yb, pp = curves[t]
        ax.hist(pp[yb == 1], bins=25, alpha=0.65, color="#d65f5f", label="actives")
        ax.hist(pp[yb == 0], bins=25, alpha=0.65, color="#4878d0", label="inactives")
        ax.set(title=t, xlabel="predicted P(active)"); ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("test molecules")
    fig.tight_layout(); save(fig, "fig_internal_sep")

    for p in ("outputs/test_metrics_table.csv", "models/test_metrics_table.csv"):
        if os.path.exists(p):
            print(f"  recorded {p} (frozen comparison):")
            print(pd.read_csv(p).to_string(index=False)); break

# ── PHASE 3: screen, PS, recovery ───────────────────────────────────────────
def _screen():
    model = S.get("model")
    if model is None: print("  skip — no model"); return
    ap = pd.read_csv("data/approved_drugs.csv").dropna(subset=["smiles"])
    ap["pref_name"] = ap["pref_name"].fillna("(unnamed)")
    n0 = len(ap)
    graphs, keep = [], []
    for i, smi in enumerate(ap.smiles):
        g = mol_to_graph(smi, [-1.0] * 3)
        if g is not None: graphs.append(g); keep.append(i)
    ap = ap.iloc[keep].reset_index(drop=True)
    out = []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=256):
            out.append(torch.sigmoid(model(b.to(DEVICE))).cpu().numpy())
    P = np.vstack(out)
    for k, t in enumerate(TARGETS): ap[f"p_{t}"] = P[:, k].round(4)
    ap["PS"] = np.exp(np.mean(np.log(np.clip(P, 1e-6, 1.0)), axis=1)).round(4)
    ap = ap.sort_values("PS", ascending=False).reset_index(drop=True)
    ap.insert(0, "rank", np.arange(1, len(ap) + 1))
    print(f"  screened {len(ap)}/{n0} | PS = geometric mean of P(active) "
          f"| PS range {ap.PS.min():.3f}–{ap.PS.max():.3f}")

    def embed(s):
        m = Chem.MolFromSmiles(s)
        if m is None: return None
        bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.uint8); DataStructs.ConvertToNumpyArray(bv, arr)
        return arr, m.GetNumHeavyAtoms()

    tr = [e for e in (embed(s) for s in pd.read_csv("data/train.csv").smiles) if e]
    B = np.stack([e[0] for e in tr]).astype(np.float32)
    ha = np.array([e[1] for e in tr]); lo, hi = np.percentile(ha, [1, 99])
    emb = [embed(s) for s in ap.smiles]
    ok = [i for i, e in enumerate(emb) if e is not None]
    A = np.stack([emb[i][0] for i in ok]).astype(np.float32)
    ap = ap.iloc[ok].reset_index(drop=True)
    inter = A @ B.T
    sim = inter / (A.sum(1, keepdims=True) + B.sum(1, keepdims=True).T - inter + 1e-9)
    ap["max_train_sim"] = sim.max(1).round(3)

    def ha_ok(s):
        m = Chem.MolFromSmiles(s)
        return m is not None and lo <= m.GetNumHeavyAtoms() <= hi
    ap["in_ad"] = (ap.max_train_sim >= SIM_THRESHOLD) & ap.smiles.apply(ha_ok)
    ad = ap[ap.in_ad].sort_values("PS", ascending=False).copy()
    ad["ad_rank"] = np.arange(1, len(ad) + 1)
    ad["ad_rank_pct"] = (100 * ad.ad_rank / len(ad)).round(2)
    ap["ad_rank"] = ad["ad_rank"].reindex(ap.index)
    ap["ad_rank_pct"] = ad["ad_rank_pct"].reindex(ap.index)
    print(f"  AD: {len(ad)}/{len(ap)} (Tanimoto ≥ {SIM_THRESHOLD} vs train, "
          f"heavy atoms [{lo:.0f},{hi:.0f}])")

    rows = []
    for name, (mid, pacts) in GT.items():
        h = ad[ad.molecule_chembl_id == mid]; sub = ap[ap.molecule_chembl_id == mid]
        if sub.empty:
            print(f"  !! {name} not in screen — skipped"); continue
        old = int(sub.iloc[0]["rank"])
        if h.empty:
            print(f"  !! {name} outside AD (full rank {old}) — skipped"); continue
        h = h.iloc[0]
        rows.append({"drug": name, "full_rank": old, "ad_rank": int(h.ad_rank),
                     "ad_rank_pct": h.ad_rank_pct, "PS": h.PS,
                     "max_train_sim": h.max_train_sim,
                     **{f"pred_{t}": h[f"p_{t}"] for t in TARGETS},
                     **{f"meas_pAct_{t}": p for t, p in zip(TARGETS, pacts)}})
    ef5, h5, auroc = None, 0, None
    if rows:
        rec = pd.DataFrame(rows); S["rec"] = rec
        n, npos = len(ad), len(rec)
        top5 = max(1, round(0.05 * n))
        h5 = int((rec.ad_rank <= top5).sum())
        ef5 = round((h5 / top5) / (npos / n), 1)
        y = ad.molecule_chembl_id.astype(str).isin({m for m, _ in GT.values()}).astype(int).values
        line = f"  top-5% = top {top5} | hits {h5}/{npos} | EF5% {ef5}"
        if 0 < y.sum() < len(y):
            auroc = round(float(roc_auc_score(y, ad.PS.values)), 3)
            line += f" | screen AUROC {auroc}"
        print("\n  RECOVERY (gold standard, within AD):")
        print(rec.to_string(index=False)); print(line)
        rec.to_csv(f"{OUT}/recovery_table.csv", index=False)
    else:
        print("  !! no reference drugs recoverable — recovery metrics skipped")

    def med(names):
        r = ad[ad.pref_name.str.lower().isin(names)]
        return None if r.empty else round(float(r.ad_rank_pct.median()), 1)
    k_med, c_med = med(KINASE), med(CONTROLS)
    print(f"  kinase cohort median pct: {k_med} | controls: {c_med}")

    print("\n  Top-10 within AD:")
    print(ad.head(10)[["ad_rank", "pref_name", "PS", "p_ABL1", "p_c-KIT",
                       "p_PDGFRB", "max_train_sim"]].to_string(index=False))

    cols = ["rank", "ad_rank", "molecule_chembl_id", "pref_name", "smiles",
            "PS", "in_ad", "max_train_sim"] + [f"p_{t}" for t in TARGETS]
    ap[cols].to_csv(f"{OUT}/drug_screen.csv", index=False)
    ad.head(10)[["ad_rank", "pref_name", "PS"] + [f"p_{t}" for t in TARGETS]] \
      .to_csv(f"{OUT}/top10_drugs.csv", index=False)
    REPORT["screen"] = {"n_screened": int(len(ap)), "n_ad": int(len(ad)),
                        "EF5": ef5, "hits_top5pct": h5, "screen_AUROC": auroc,
                        "kinase_median_pct": k_med, "controls_median_pct": c_med,
                        "ps": "geometric mean of the three P(active)"}
    S["ap"], S["ad"] = ap, ad

    if rows:
        fig, ax = plt.subplots(figsize=(6.5, 2.6))
        y0 = np.arange(len(rec))
        ax.barh(y0, rec.ad_rank_pct, color="#4878d0", height=0.55)
        for yy, (_, r) in zip(y0, rec.iterrows()):
            ax.text(r.ad_rank_pct + 0.8, yy, f"AD rank {int(r.ad_rank)}",
                    va="center", fontsize=8)
        ax.axvline(5.0, color="crimson", ls="--", lw=1, label="top-5% (EF5% window)")
        ax.set_yticks(y0); ax.set_yticklabels(rec.drug); ax.invert_yaxis()
        ax.set_xlabel("percentile within AD (lower = recovered earlier)")
        ax.set_xlim(0, max(10.0, float(rec.ad_rank_pct.max()) * 1.25))
        ax.legend(frameon=False, fontsize=8); save(fig, "fig_recovery")

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    rng = np.random.default_rng(0)
    for i, (label, cohort, color) in enumerate(
            [("multi-kinase inhibitors", KINASE, "#d65f5f"),
             ("non-kinase controls", CONTROLS, "#4878d0")]):
        v = ad[ad.pref_name.str.lower().isin(cohort)].ad_rank_pct.dropna().values
        ax.scatter(v, np.full(len(v), i) + rng.uniform(-0.1, 0.1, len(v)),
                   s=42, color=color, alpha=0.85, edgecolor="white", lw=0.5)
        if len(v): ax.axvline(np.median(v), color=color, ls="--", lw=1)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["multi-kinase", "controls"])
    ax.set_xlabel("percentile within AD (lower = ranked higher)")
    ax.set_title("Cohort contrast — pre-specified positives vs controls")
    save(fig, "fig_cohorts")

    top = ad.head(10)
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    x, w, c3 = np.arange(len(top)), 0.27, ["#4878d0", "#d65f5f", "#6acc65"]
    for k, t in enumerate(TARGETS):
        ax.bar(x + (k - 1) * w, top[f"p_{t}"], w, label=t, color=c3[k])
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r.ad_rank)}. {r.pref_name} (PS {r.PS:.2f})"
                        for _, r in top.iterrows()], rotation=35, ha="right",
                       fontsize=8)
    ax.set_ylabel("P(active)"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Top-10 approved drugs by Polypharmacology Score")
    fig.tight_layout(); save(fig, "fig_top10")
    try:
        from rdkit.Chem.Draw import MolsToGridImage
        mols = [Chem.MolFromSmiles(s) for s in top.smiles]
        keepg = [i for i, m in enumerate(mols) if m is not None]
        MolsToGridImage([mols[i] for i in keepg], molsPerRow=5,
                        subImgSize=(320, 280),
                        legends=[f"{int(top.ad_rank.iloc[i])}. {top.pref_name.iloc[i]}"
                                 f"  PS={top.PS.iloc[i]:.2f}" for i in keepg]
                        ).save(f"{FIGS}/fig_top10_molecules.png")
        print("  + figs/fig_top10_molecules.png")
    except Exception as e:
        print(f"  - molecule grid skipped ({e})")

# ── PHASE 4: temporal external validation ───────────────────────────────────
def _external():
    model = S.get("model")
    if model is None: print("  skip — no model"); return
    mat = S.get("mat")
    pool_chembl = ({str(c) for c in mat.molecule_chembl_id.dropna() if str(c).strip()}
                   if "molecule_chembl_id" in mat.columns else set())
    pool_keys = {k for k in (skel_of(s) for s in mat.smiles) if k}
    apd = pd.read_csv("data/approved_drugs.csv").dropna(subset=["smiles"])
    ap_chem = set(apd.molecule_chembl_id)
    ap_skel = {k for k in (skel_of(s) for s in apd.smiles) if k}

    @torch.no_grad()
    def predict(smiles_list, k):
        graphs, idx = [], []
        for i, s in enumerate(smiles_list):
            g = mol_to_graph(s, [-1.0] * 3)
            if g is not None: graphs.append(g); idx.append(i)
        out = []
        for b in DataLoader(graphs, batch_size=256):
            out.append(torch.sigmoid(model(b.to(DEVICE))).cpu().numpy())
        return idx, (np.vstack(out)[:, k] if out else np.array([]))

    ext, chosen = {}, {}
    for tgt, (path, acc) in BDB.items():
        hdr = pd.read_csv(path, sep="\t", nrows=0)
        sep = "," if hdr.shape[1] < 5 else "\t"
        cols = list(hdr.columns)
        c_smiles = find_col(cols, "ligand", "smiles")
        c_ikey, c_chem = find_col(cols, "inchi key"), find_col(cols, "chembl id of ligand")
        c_ki, c_ic50, c_kd = (find_col(cols, "ki (nm)"), find_col(cols, "ic50 (nm)"),
                              find_col(cols, "kd (nm)"))
        c_org = find_col(cols, "source organism")
        c_uni = find_col(cols, "swissprot", "primary id", "chain 1")
        c_name = find_col(cols, "target name")
        use = [c for c in (c_smiles, c_ikey, c_chem, c_ki, c_ic50, c_kd,
                           c_org, c_uni, c_name) if c]
        df = pd.read_csv(path, sep=sep, dtype=str, on_bad_lines="skip",
                         usecols=use, engine="python", quoting=3)
        print(f"\n  {tgt}: rows {len(df)}")
        if c_org:
            df = df[df[c_org].fillna("").str.lower().str.contains("human|homo sapiens")]
        if c_uni:
            df = df[df[c_uni].fillna("").str.contains(acc, na=False)
                    | df[c_uni].fillna("").eq("")]
        if c_name:
            df = df[~df[c_name].fillna("").str.contains(MUT, regex=True)]
        recs = {}
        for _, r in df.iterrows():
            smi = largest_fragment(r[c_smiles]) if isinstance(r[c_smiles], str) else None
            if not smi: continue
            iw = r.get(c_ikey)
            skel = (str(iw).split("-")[0] if isinstance(iw, str) and iw else skel_of(smi))
            if not skel: continue
            rec = recs.setdefault(skel, {"smi": smi, "chembl": set(), "ex": [], "any": []})
            if c_chem and isinstance(r.get(c_chem), str) and r[c_chem].strip():
                rec["chembl"].add(r[c_chem].strip())
            primary = False
            for c in (c_ki, c_ic50):
                q, v = parse_aff(r.get(c))
                if v is not None and 0.01 <= v <= 1e7:
                    primary = True; rec["any"].append(v)
                    if q is None: rec["ex"].append(v)
            if not primary and c_kd:
                q, v = parse_aff(r.get(c_kd))
                if v is not None and 0.01 <= v <= 1e7:
                    rec["any"].append(v)
                    if q is None: rec["ex"].append(v)
        rows = [{"skel": k, "smi": v["smi"], "chembl": ";".join(v["chembl"]),
                 "pAct_exact": (float(np.median([to_pact(x) for x in v["ex"]]))
                                if v["ex"] else np.nan),
                 "pAct_any": (float(np.median([to_pact(x) for x in v["any"]]))
                              if v["any"] else np.nan)}
                for k, v in recs.items()]
        bdb = pd.DataFrame(rows)
        in_pool = (bdb.chembl.apply(lambda s: any(c in pool_chembl
                                                  for c in s.split(";") if c))
                   | bdb.skel.isin(pool_keys))
        is_appr = (bdb.chembl.apply(lambda s: any(c in ap_chem
                                                  for c in s.split(";") if c))
                   | bdb.skel.isin(ap_skel))
        bdb = bdb[~in_pool & ~is_appr].reset_index(drop=True)
        print(f"  external: {len(bdb)} ligands")
        idx, P = predict(list(bdb.smi), TARGETS.index(tgt))
        if not idx:
            print("  !! no valid structures — target skipped"); continue
        bdb = bdb.iloc[idx].copy(); bdb["pred"] = P
        res = {}
        for col in ("pAct_exact", "pAct_any"):
            lab = bdb.dropna(subset=[col])
            d = {"n_labeled": int(len(lab))}
            if len(lab) > 10:
                d["rho"] = round(float(spearmanr(lab[col], lab.pred)[0]), 3)
            yb = lab[col].apply(lambda p: 1 if p >= ACTIVE_PACT
                                else 0 if p <= 5 else None).dropna()
            if yb.nunique() == 2:
                d["AUC"] = round(float(roc_auc_score(
                    yb.astype(int), lab.loc[yb.index, "pred"])), 3)
                d["n_act"], d["n_inact"] = int((yb == 1).sum()), int((yb == 0).sum())
            d["pred_q05/50/95"] = [round(float(x), 3)
                                   for x in bdb.pred.quantile([0.05, 0.5, 0.95])]
            res[col] = d
        fz = V3_FROZEN[tgt]
        pick = min(("pAct_exact", "pAct_any"),
                   key=lambda c: abs(res[c]["n_labeled"] - fz["n"]))
        ext[tgt] = {"exact": res["pAct_exact"], "inclusive": res["pAct_any"],
                    "recorded_step6": fz, "figure_variant": pick}
        chosen[tgt] = bdb
        print(f"  exact: {res['pAct_exact']}")
        print(f"  inclusive: {res['pAct_any']}")
        print(f"  recorded step-6: {fz}")
    REPORT["external"] = ext
    for t, dfx in chosen.items():
        dfx.to_csv(f"{OUT}/external_{t.replace('-', '')}.csv", index=False)

    if len(chosen) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        for ax, t in zip(axes, TARGETS):
            col = ext[t]["figure_variant"]
            lab = chosen[t].dropna(subset=[col])
            rho = round(float(spearmanr(lab[col], lab.pred)[0]), 3)
            ax.scatter(lab[col], lab.pred, s=10, alpha=0.5,
                       color="#4878d0", edgecolor="none")
            ax.set(xlabel="measured pAct", ylabel="predicted P(active)",
                   title=t, xlim=(3, 11), ylim=(0, 1.02))
            ax.text(0.05, 0.05, f"ρ={rho:.3f} (n={len(lab)})\n"
                    f"recorded: ρ={V3_FROZEN[t]['rho']:.3f} (n={V3_FROZEN[t]['n']})",
                    transform=ax.transAxes, fontsize=7.5, va="bottom")
        fig.tight_layout(); save(fig, "fig_external")

        col = ext["c-KIT"]["figure_variant"]
        dfx, lab = chosen["c-KIT"], chosen["c-KIT"].dropna(subset=[col])
        fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
        ax[0].hist(dfx.pred, bins=30, color="#4878d0", edgecolor="white", lw=0.3)
        ax[0].axvline(0.90, color="crimson", ls="--", lw=1)
        share = float((dfx.pred >= 0.9).mean())
        ax[0].set(xlabel="predicted P(active)", ylabel="ligands",
                  title=f"predictions — {share*100:.1f}% ≥ 0.90 (saturated)")
        ax[1].hist(lab[col], bins=30, color="#d65f5f", edgecolor="white", lw=0.3)
        ax[1].set(xlabel="measured pAct",
                  title=f"measured — range {lab[col].min():.1f}–{lab[col].max():.1f}")
        fig.suptitle("c-KIT temporal external set: saturated predictions "
                     "vs wide measured potencies", fontsize=9)
        fig.tight_layout(); save(fig, "fig_kit_saturation")
        d = lab.copy(); d["scf"] = d.smi.apply(scaffold_of)
        groups = [g for _, g in d.groupby("scf") if len(g) >= 5]
        rhos = [spearmanr(g[col], g.pred)[0] for g in groups
                if np.std(g[col]) > 0 and np.std(g.pred) > 0]
        if rhos:
            print(f"\n  c-KIT within-scaffold ρ (≥5 ligands): median "
                  f"{np.median(rhos):.3f} over {len(groups)} groups / "
                  f"{sum(len(g) for g in groups)} ligands")
            print("  recorded step6b diagnostic: 0.096 over 32 groups / 368 ligands")

# ── PHASE 5: training curves ────────────────────────────────────────────────
def _curves():
    cands = set(glob.glob("models/*log*.json") + glob.glob("outputs/*log*.json")
                + glob.glob("models/*history*.json"))
    for path in sorted(cands):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        if not (isinstance(data, list) and data and isinstance(data[0], dict)): continue
        if "epoch" not in data[0]: continue
        keys = [k for k in data[0] if k != "epoch"
                and all(isinstance(r.get(k), (int, float)) for r in data)]
        if not keys: continue
        ep = [r["epoch"] for r in data]
        lossk = [k for k in keys if "loss" in k.lower()]
        metk = [k for k in keys if k not in lossk]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for k in (lossk or keys):
            axes[0].plot(ep, [r[k] for r in data], lw=1.2, label=k)
        axes[0].set_yscale("log"); axes[0].set(xlabel="epoch", title="training loss")
        axes[0].legend(fontsize=7, frameon=False)
        if metk:
            for k in metk:
                axes[1].plot(ep, [r[k] for r in data], lw=1.2, label=k)
            axes[1].set(xlabel="epoch", title="validation metric")
            axes[1].legend(fontsize=7, frameon=False)
        else:
            axes[1].axis("off")
        save(fig, "fig_traincurves")
        print(f"  training log: {path} ({len(data)} epochs)")
        REPORT["training_log"] = path
        return
    print("  no training log found — if your step-4 trainer saved one, "
          "tell me the filename")

# ── PHASE 6: UMAP of learned space ──────────────────────────────────────────
def _umap():
    ap, model = S.get("ap"), S.get("model")
    if ap is None or model is None: print("  skip — no model/screen"); return
    try:
        import umap
    except ImportError:
        print("  skip (pip install umap-learn)"); return
    graphs = [mol_to_graph(s, [-1.0] * 3) for s in ap.smiles]
    keep = [i for i, g in enumerate(graphs) if g is not None]
    d, gs = ap.iloc[keep].reset_index(drop=True), [graphs[i] for i in keep]
    E = []
    with torch.no_grad():
        for b in DataLoader(gs, batch_size=256):
            E.append(model.get_embedding(b.to(DEVICE)).cpu().numpy())
    E = np.vstack(E)
    Z = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                  random_state=42).fit_transform(E)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=d.PS, s=8, cmap="viridis",
                    alpha=0.85, edgecolor="none")
    plt.colorbar(sc, ax=ax, label="PS")
    gt = d.molecule_chembl_id.astype(str).isin({m for m, _ in GT.values()})
    if gt.any():
        ax.scatter(Z[gt.values, 0], Z[gt.values, 1], marker="*", s=260,
                   color="crimson", edgecolor="black", lw=0.5, zorder=3,
                   label="reference inhibitors")
        rev = {m: n for n, (m, _) in GT.items()}
        for z, (_, r) in zip(Z[gt.values], d[gt].iterrows()):
            ax.annotate(rev[str(r.molecule_chembl_id)], (z[0], z[1]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.legend(frameon=False, fontsize=8)
    ax.set(xticks=[], yticks=[],
           title="UMAP of GNN embeddings — approved-drug screen")
    save(fig, "fig_umap")

phase("PHASE 0 — ARTEFACT INVENTORY", _inventory)
phase("PHASE 1 — GOLD STANDARD & LABELS", _labels)
phase("PHASE 2 — INTERNAL VALIDATION", _internal)
phase("PHASE 3 — SCREEN, PS & RECOVERY", _screen)
phase("PHASE 4 — TEMPORAL EXTERNAL VALIDATION", _external)
phase("PHASE 5 — TRAINING CURVES", _curves)
phase("PHASE 6 — UMAP (learned space)", _umap)

print("\n" + "=" * 72 + "\nFINAL REPORT (also written to outputs/paper/report.json)\n" + "=" * 72)
json.dump(REPORT, open(f"{OUT}/report.json", "w"), indent=2, default=float)
print(json.dumps(REPORT, indent=2, default=float))
print("\nfiles written:")
for f in sorted(glob.glob(f"{OUT}/**/*.*", recursive=True)):
    print("  " + f)
print("\n[CONFIRM on paste]: (1) censored/grey-zone label handling vs your step-7 console;")
print("(2) PS = geometric mean of P(active) vs your step-5 PS; (3) paper external text =")
print("your recorded step-6 numbers (frozen above); figures carry recomputed n's.")