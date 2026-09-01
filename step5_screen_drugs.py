"""
Step 5: PRISM screening of approved drugs + retrospective recovery analysis.

Reads : data/approved_drugs.csv, models/best_model.pt, models/test_metrics.json
Uses  : step4_train.py (featuriser imported — single source of truth, no
        train/serve skew), model.py
Writes: outputs/drug_screen.csv, outputs/recovery_table.csv,
        outputs/top10_drugs.csv, outputs/step5_summary.json
Run   : python step5_screen_drugs.py   (project root; ~3-6 min on CPU)
"""
import json, os, sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from rdkit import RDLogger
from sklearn.metrics import roc_auc_score
from model import MultiTargetGNN
from step4_train import mol_to_graph, TARGETS   # exact same featurisation

RDLogger.DisableLog("rdApp.*")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = {"ABL1": 1/3, "c-KIT": 1/3, "PDGFRB": 1/3}   # configurable; sum = 1

# Ground truth: ChEMBL_37 wild-type assay medians (printed by step 3, WT-only)
GT = {
    "imatinib":  {"mid": "CHEMBL941",     "ABL1": 6.658, "c-KIT": 6.685, "PDGFRB": 6.517},
    "dasatinib": {"mid": "CHEMBL5416410", "ABL1": 8.509, "c-KIT": 7.983, "PDGFRB": 7.553},
    "nilotinib": {"mid": "CHEMBL255863",  "ABL1": 7.553, "c-KIT": 6.801, "PDGFRB": 7.185},
    "ponatinib": {"mid": "CHEMBL1171837", "ABL1": 9.432, "c-KIT": 8.770, "PDGFRB": 8.921},
}
KINASE_COHORT = ["sunitinib", "sorafenib", "pazopanib", "axitinib", "bosutinib",
                 "regorafenib", "vandetanib", "cabozantinib", "lenvatinib",
                 "midostaurin", "masitinib", "toceranib"]
CONTROLS = ["metformin", "atorvastatin", "warfarin", "ibuprofen", "amoxicillin",
            "oseltamivir", "fluoxetine", "loratadine"]

os.makedirs("outputs", exist_ok=True)

# --- 0. gate: refuse to screen with an underperforming model ----------------
tm = json.load(open("models/test_metrics.json"))["test_metrics"]
for t in TARGETS:
    auc = tm[t]["AUC"]
    if auc is None or auc < 0.70:
        sys.exit(f"GATE FAILED: {t} test AUC = {auc} (< 0.70). "
                 "Do not screen; paste metrics so we can retune (dropout 0.3, lr 5e-4).")
print("gate passed: all per-target scaffold-split test AUCs >= 0.70")

# --- 1. featurise the approved-drug library ----------------------------------
ap = pd.read_csv("data/approved_drugs.csv").dropna(subset=["smiles"])
ap["pref_name"] = ap["pref_name"].fillna("(unnamed)")
n0 = len(ap)
graphs, keep = [], []
for i, smi in enumerate(ap.smiles):
    g = mol_to_graph(smi, [-1.0] * len(TARGETS))     # labels unused at inference
    if g is not None:
        g.mol_id = ap.molecule_chembl_id.iloc[i]
        graphs.append(g); keep.append(i)
ap = ap.iloc[keep].reset_index(drop=True)
print(f"featurised {len(ap)}/{n0} approved drugs "
      f"({n0 - len(ap)} dropped: unparseable / no bonds)")

# --- 2. inference -------------------------------------------------------------
ck = torch.load("models/best_model.pt", map_location=DEVICE)
model = MultiTargetGNN(hidden_dim=ck["hidden_dim"], num_layers=ck["num_layers"],
                       dropout=ck["dropout"], target_names=ck["targets"]).to(DEVICE)
model.load_state_dict(ck["state_dict"]); model.eval()
probs = []
with torch.no_grad():
    for b in DataLoader(graphs, batch_size=256):
        probs.append(torch.sigmoid(model(b.to(DEVICE))).cpu().numpy())
P = np.vstack(probs)                                  # [N, 3]
for k, t in enumerate(TARGETS):
    ap[f"P_{t}"] = P[:, k]

# --- 3. Polypharmacology Score: weighted geometric mean ----------------------
#   PS = prod_k P_k^w_k   (uniform default; configurable weights, sum=1)
#   Unlike v1's x1.875 constant, weights here actually change the ranking.
w = np.array([WEIGHTS[t] for t in TARGETS])
ap["PS"] = np.exp(np.log(np.clip(P, 1e-6, 1.0)) @ w)
for t in TARGETS:   # single-target ranks for the analysis section
    ap[f"rank_{t}"] = ap[f"P_{t}"].rank(ascending=False, method="min").astype(int)
ap = ap.sort_values("PS", ascending=False).reset_index(drop=True)
ap.insert(0, "rank", np.arange(1, len(ap) + 1))
ap["rank_pct"] = (100 * ap["rank"] / len(ap)).round(2)

# --- 4. recovery of the four ground-truth drugs -------------------------------
rows = []
for name, info in GT.items():
    hit = ap[ap.molecule_chembl_id == info["mid"]]
    if hit.empty:
        print(f"!! WARNING: {name} ({info['mid']}) not in screened set"); continue
    r = hit.iloc[0]
    row = {"drug": name, "rank": int(r["rank"]), "rank_pct": r["rank_pct"],
           "PS": round(r["PS"], 3)}
    for t in TARGETS:
        row[f"P_{t}"] = round(r[f"P_{t}"], 3)
        row[f"rank_{t}"] = int(r[f"rank_{t}"])
        row[f"pAct_{t}"] = info[t]        # published ChEMBL_37 WT median
    rows.append(row)
rec = pd.DataFrame(rows)

n, n_act = len(ap), len(rec)
top1 = max(1, round(0.01 * n)); top5 = max(1, round(0.05 * n))
h1 = int((rec["rank"] <= top1).sum()); h5 = int((rec["rank"] <= top5).sum())
ef1 = (h1 / top1) / (n_act / n); ef5 = (h5 / top5) / (n_act / n)
y = ap.molecule_chembl_id.isin({v["mid"] for v in GT.values()}).astype(int).values
auroc = float(roc_auc_score(y, ap["PS"].values)) if n_act else None

print(f"\n===== RECOVERY (library N={n}, known multi-target drugs={n_act}) =====")
print(rec[["drug", "rank", "rank_pct", "PS", "P_ABL1", "P_c-KIT", "P_PDGFRB",
           "pAct_ABL1", "pAct_c-KIT", "pAct_PDGFRB" if "pAct_PDGFRB" in rec
           else "pAct_PDGFRB"]].to_string(index=False))
print(f"\ntop-1% = top {top1} | hits {h1} | EF1% {ef1:.1f}")
print(f"top-5% = top {top5} | hits {h5} | EF5% {ef5:.1f}")
print(f"screen AUROC (4 knowns as actives): {auroc:.3f}")
print(f"median rank of knowns: {rec['rank'].median():.0f} "
      f"({rec['rank_pct'].median():.1f} pct)")

# --- 5. specificity: approved kinase cohort vs non-kinase controls ------------
def locate(nm):
    r = ap[ap.pref_name.str.lower() == nm.lower()]
    return None if r.empty else {"name": nm, "rank": int(r.iloc[0]["rank"]),
                                 "rank_pct": r.iloc[0]["rank_pct"],
                                 "PS": round(r.iloc[0]["PS"], 3)}
kin = pd.DataFrame([x for x in map(locate, KINASE_COHORT) if x])
ctl = pd.DataFrame([x for x in map(locate, CONTROLS) if x])
if len(kin): print("\nkinase cohort:\n" + kin.to_string(index=False) +
                   f"\n  median rank_pct: {kin['rank_pct'].median():.1f}")
if len(ctl): print("\nnon-kinase controls:\n" + ctl.to_string(index=False) +
                   f"\n  median rank_pct: {ctl['rank_pct'].median():.1f}")

# --- 6. outputs ---------------------------------------------------------------
cols = ["rank", "molecule_chembl_id", "pref_name", "smiles", "PS",
        "P_ABL1", "P_c-KIT", "P_PDGFRB"] + [f"rank_{t}" for t in TARGETS]
ap[cols].to_csv("outputs/drug_screen.csv", index=False)
rec.to_csv("outputs/recovery_table.csv", index=False)
ap.head(10)[["rank", "pref_name", "molecule_chembl_id", "PS",
             "P_ABL1", "P_c-KIT", "P_PDGFRB"]].to_csv("outputs/top10_drugs.csv",
                                                     index=False)
print("\nTop 10 by PS:")
print(ap.head(10)[["rank", "pref_name", "PS", "P_ABL1", "P_c-KIT",
                   "P_PDGFRB"]].to_string(index=False))
json.dump({"n_screened": n, "weights": WEIGHTS,
           "ground_truth_source": "ChEMBL_37 wild-type assay medians",
           "recovery": rec.to_dict("records"),
           "EF1": ef1, "EF5": ef5, "top1_size": top1, "top5_size": top5,
           "hits_top1": h1, "hits_top5": h5, "screen_auroc": auroc,
           "kinase_cohort": kin.to_dict("records"),
           "controls": ctl.to_dict("records")},
          open("outputs/step5_summary.json", "w"), indent=2)
print("\nwrote outputs/drug_screen.csv, recovery_table.csv, top10_drugs.csv, "
      "step5_summary.json")