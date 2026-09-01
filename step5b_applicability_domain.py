"""
step5b_applicability_domain.py — re-rank the drug screen inside the AD.

AD rule (state verbatim in the paper):
  PASS iff max Tanimoto (Morgan r=2, 2048-bit) to any TRAINING molecule >= 0.30
      and heavy-atom count within [1st, 99th] percentile of the training set.
Threshold fixed a priori (literature kNN-AD standard); both unrestricted and
AD-restricted results are reported.

Reads : outputs/drug_screen.csv, data/train.csv
Writes: outputs/ad_drug_screen.csv, outputs/step5b_summary.json
"""
import json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
SIM_THRESHOLD = 0.30
TARGETS = ["ABL1", "c-KIT", "PDGFRB"]
GT = {"imatinib":  ("CHEMBL941",     (6.658, 6.685, 6.517)),
      "dasatinib": ("CHEMBL5416410", (8.509, 7.983, 7.553)),
      "nilotinib": ("CHEMBL255863",  (7.553, 6.801, 7.185)),
      "ponatinib": ("CHEMBL1171837", (9.432, 8.770, 8.921))}
KINASE = ["sunitinib", "sorafenib", "pazopanib", "axitinib", "bosutinib",
          "regorafenib", "vandetanib", "cabozantinib", "lenvatinib", "midostaurin"]
CONTROLS = ["metformin", "atorvastatin", "warfarin", "ibuprofen", "amoxicillin",
            "oseltamivir", "fluoxetine", "loratadine"]
ARTIFACTS = ["ANIDULAFUNGIN", "MIPOMERSEN", "MIPOMERSEN SODIUM", "REZAFUNGIN",
             "AMOXICILLIN", "CEFPIRAMIDE", "CEFPIRAMIDE SODIUM"]

def embed(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr, m.GetNumHeavyAtoms()

# training chemistry = the model's domain
tr = [e for e in (embed(s) for s in pd.read_csv("data/train.csv").smiles) if e]
B = np.stack([e[0] for e in tr]).astype(np.float32)
ha_tr = np.array([e[1] for e in tr])
ha_lo, ha_hi = np.percentile(ha_tr, [1, 99])
print(f"domain: {len(tr)} training mols | heavy atoms [{ha_lo:.0f}, {ha_hi:.0f}]")

dr = pd.read_csv("outputs/drug_screen.csv")
emb = [embed(s) for s in dr.smiles]
ok = [i for i, e in enumerate(emb) if e is not None]
A = np.stack([emb[i][0] for i in ok]).astype(np.float32)
dr = dr.iloc[ok].reset_index(drop=True)
inter = A @ B.T
tanim = inter / (A.sum(1, keepdims=True) + B.sum(1, keepdims=True).T - inter + 1e-9)
dr["max_train_sim"] = tanim.max(1)
dr["heavy_atoms"] = [emb[i][1] for i in ok]
dr["in_ad"] = (dr.max_train_sim >= SIM_THRESHOLD) & dr.heavy_atoms.between(ha_lo, ha_hi)

n_full = len(dr)
ad = dr[dr.in_ad].sort_values("PS", ascending=False).reset_index(drop=True)
ad["ad_rank"] = np.arange(1, len(ad) + 1)
ad["ad_rank_pct"] = (100 * ad.ad_rank / len(ad)).round(2)
print(f"AD: {len(ad)}/{n_full} molecules pass "
      f"(sim>={SIM_THRESHOLD}, heavy atoms in [{ha_lo:.0f},{ha_hi:.0f}])")

print("\n=== previously suspicious top-rankers, now diagnosed ===")
print(dr[dr.pref_name.str.upper().isin(ARTIFACTS)]
      [["pref_name", "rank", "max_train_sim", "heavy_atoms", "in_ad"]]
      .to_string(index=False))

print("\n=== recovery within AD (old rank -> AD rank) ===")
rows = []
for name, (mid, pacts) in GT.items():
    h = ad[ad.molecule_chembl_id == mid]
    if h.empty:
        print(f"!! {name} fell OUTSIDE the AD — inspect before writing anything"); continue
    h = h.iloc[0]
    old = int(dr[dr.molecule_chembl_id == mid].iloc[0]["rank"])
    rows.append({"drug": name, "old_rank": old, "ad_rank": int(h.ad_rank),
                 "ad_rank_pct": h.ad_rank_pct, "PS": round(h.PS, 3),
                 "max_train_sim": round(h.max_train_sim, 2),
                 **{f"P_{t}": round(h[f"P_{t}"], 3) for t in TARGETS},
                 **{f"pAct_{t}": p for t, p in zip(TARGETS, pacts)}})
rec = pd.DataFrame(rows)
print(rec[["drug", "old_rank", "ad_rank", "ad_rank_pct", "PS",
           "max_train_sim"]].to_string(index=False))
print("(max_train_sim ~0.6-0.9 for the knowns = non-approved analogs in training;")
print(" expected for repurposing, disclosed in the paper)")

n, n_pos = len(ad), len(rec)
top1, top5 = max(1, round(0.01 * n)), max(1, round(0.05 * n))
h1, h5 = int((rec.ad_rank <= top1).sum()), int((rec.ad_rank <= top5).sum())
ef1, ef5 = (h1 / top1) / (n_pos / n), (h5 / top5) / (n_pos / n)
y = ad.molecule_chembl_id.isin({m for m, _ in GT.values()}).astype(int).values
print(f"\nN={n} | top-1% = {top1} (hits {h1}, EF1% {ef1:.1f}) | "
      f"top-5% = {top5} (hits {h5}, EF5% {ef5:.1f})")
print(f"AD-restricted screen AUROC: {roc_auc_score(y, ad.PS.values):.3f}")

def med_pct(names):
    r = ad[ad.pref_name.str.lower().isin(names)]
    return None if r.empty else round(float(r.ad_rank_pct.median()), 1)
print(f"kinase cohort median pct: {med_pct(KINASE)} | "
      f"controls median pct: {med_pct(CONTROLS)}")

print("\n=== top 10 within AD (the paper's honest hit list) ===")
print(ad.head(10)[["ad_rank", "pref_name", "PS", "P_ABL1", "P_c-KIT",
                   "P_PDGFRB", "max_train_sim"]].to_string(index=False))

ad.to_csv("outputs/ad_drug_screen.csv", index=False)
json.dump({"n_full": n_full, "n_ad": n, "sim_threshold": SIM_THRESHOLD,
           "heavy_atom_bounds": [float(ha_lo), float(ha_hi)],
           "recovery": rec.to_dict("records"), "EF1": ef1, "EF5": ef5,
           "top1_size": top1, "top5_size": top5, "hits_top1": h1,
           "hits_top5": h5},
          open("outputs/step5b_summary.json", "w"), indent=2)
print("\nwrote outputs/ad_drug_screen.csv, outputs/step5b_summary.json")