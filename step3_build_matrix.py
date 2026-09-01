
# step3_build_matrix.py
"""
Step 3: build the PRISM multi-task training matrix for ABL1 / c-KIT / PDGFRB.
Reads : data/raw_*.csv            (from step 2, no re-query)
Writes: data/approved_drugs.csv   (Pool 4 screening library, max_phase=4)
        data/matrix.csv           (3-task labels, approved drugs REMOVED)
        data/train|val|test.csv   (Bemis-Murcko scaffold split)
        data/manifest.json        (cuts, filters, sizes, SHA-256)
Runtime: ~3-5 min (approved-drug pull dominates).
"""
import hashlib, json, os, re
import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from chembl_webresource_client.new_client import new_client

RDLogger.DisableLog("rdApp.*")
molecule_api = new_client.molecule

TARGETS   = ["ABL1", "c-KIT", "PDGFRB"]
ACTIVE_PACT, INACTIVE_PACT = 6.0, 5.0     # active <= 1 uM, inactive >= 10 uM
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1
KNOWN = {"imatinib": "CHEMBL941", "dasatinib": "CHEMBL5416410",
         "nilotinib": "CHEMBL255863", "ponatinib": "CHEMBL1171837"}
os.makedirs("data", exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# --- 1. per-target labels: biochemical, wild-type, exact-then-censored -------
def label_target(label):
    df = pd.read_csv(f"data/raw_{label}.csv"); n0 = len(df)
    df = df[df.assay_type == "B"]; n1 = len(df)
    mut = df.assay_description.fillna("").str.contains(
        r"mutant|mutation|[A-Za-z]\d{3}[A-Za-z]", case=False, regex=True)
    df = df[~mut]
    print(f"  {label}: {n0} rows -> biochemical {n1} -> wild-type {len(df)}")

    val  = pd.to_numeric(df.standard_value, errors="coerce")
    unit = df.standard_units == "nM"
    ex = df[unit & (df.standard_relation == "=") & val.between(0.01, 1e7)].copy()
    ex["pAct"] = -np.log10(val[ex.index] * 1e-9)
    gt = df[unit & (df.standard_relation == ">") & (val >= 10_000)]   # censored-inactive
    lt = df[unit & (df.standard_relation == "<") & (val <=  1_000)]   # censored-active

    out, wt = {}, {}
    for mid, g in ex.groupby("molecule_chembl_id"):
        p = g.pAct.median(); wt[mid] = p
        if   p >= ACTIVE_PACT:   out[mid] = 1
        elif p <= INACTIVE_PACT: out[mid] = 0           # else: gray -> unlabeled
    gt_ids, lt_ids = set(gt.molecule_chembl_id), set(lt.molecule_chembl_id)
    for mid in (gt_ids - lt_ids) - set(out): out[mid] = 0
    for mid in (lt_ids - gt_ids) - set(out): out[mid] = 1

    smiles = (df.dropna(subset=["canonical_smiles"])
                .groupby("molecule_chembl_id").canonical_smiles.first())
    tab = pd.DataFrame({"molecule_chembl_id": list(out), label: list(out.values())})
    tab = tab.merge(smiles.rename("smiles").reset_index(),
                    on="molecule_chembl_id", how="left")
    print(f"     labeled: {len(tab)} (act={int(tab[label].sum())}, "
          f"inact={int((tab[label]==0).sum())}, "
          f"censored-only={len((gt_ids|lt_ids) - set(wt))})")
    return tab, wt

print("=== labeling ===")
tabs, wts = {}, {}
for l in TARGETS:
    tabs[l], wts[l] = label_target(l)

# --- 2. Pool 4: approved drugs (cached) --------------------------------------
def fetch_approved():
    if os.path.exists("data/approved_drugs.csv"):
        return pd.read_csv("data/approved_drugs.csv")
    recs = []
    for m in molecule_api.filter(max_phase=4).only(
            ["molecule_chembl_id", "pref_name", "molecule_structures"]):
        smi = (m.get("molecule_structures") or {}).get("canonical_smiles")
        recs.append((m["molecule_chembl_id"], m.get("pref_name"), smi))
    ap = pd.DataFrame(recs, columns=["molecule_chembl_id", "pref_name", "smiles"])
    ap.to_csv("data/approved_drugs.csv", index=False)
    return ap

print("\n=== approved drugs (max_phase=4) ===")
approved = fetch_approved(); print(f"  {len(approved)} molecules")
approved_ids = set(approved.molecule_chembl_id)

# --- 3. Pool 1 matrix: outer join, drop drugs, validate ----------------------
per = [tabs[l].set_index("molecule_chembl_id") for l in TARGETS]
mat = pd.concat([per[i][TARGETS[i]] for i in range(3)], axis=1, join="outer")
smi = pd.concat([p["smiles"] for p in per]).groupby(level=0).first()
mat = mat.join(smi).reset_index()
mat[TARGETS] = mat[TARGETS].fillna(-1).astype(int)
mat = mat[(mat[TARGETS] != -1).any(axis=1)]
mat = mat[mat.smiles.apply(lambda s: isinstance(s, str)
                           and Chem.MolFromSmiles(s) is not None)]
n0 = len(mat)
mat = mat[~mat.molecule_chembl_id.isin(approved_ids)]
print(f"\n=== matrix ===\n{n0} -> {len(mat)} compounds "
      f"({n0-len(mat)} approved drugs removed)")
assert not set(KNOWN.values()) & set(mat.molecule_chembl_id), "DRUG LEAK"
print("known drugs confirmed absent from training pool")
print(f"labeled on >=2 targets: {(mat[TARGETS] != -1).sum(axis=1).ge(2).sum()}")
print(f"labeled on all 3:       {(mat[TARGETS] != -1).all(axis=1).sum()}")
print(f"active on all 3 (labeled): {(mat[TARGETS] == 1).all(axis=1).sum()}")

# --- 4. Bemis-Murcko scaffold split (deterministic, big groups -> train) -----
def scaffold_of(s):
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
        return sc if sc else s
    except Exception:
        return s

mat["scaffold"] = mat.smiles.apply(scaffold_of)
order = mat.groupby("scaffold").size().sort_values(ascending=False)
n = len(mat); n_tr, n_va = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
tr = va = te = 0; assign = {}
for sc, sz in order.items():
    r = [n_tr - tr, n_va - va, n - n_tr - n_va - te]
    pick = r.index(max(r))
    assign[sc] = ["train", "val", "test"][pick]
    tr, va, te = (tr + sz, va, te) if pick == 0 else \
                 (tr, va + sz, te) if pick == 1 else (tr, va, te + sz)
mat["split"] = mat.scaffold.map(assign)

for s in ("train", "val", "test"):
    sub = mat[mat.split == s]; sub.to_csv(f"data/{s}.csv", index=False)
    print(f"  {s}: {len(sub)} | per-target (act,inact): "
          f"{ {l: (int((sub[l]==1).sum()), int((sub[l]==0).sum())) for l in TARGETS} }")
mat.to_csv("data/matrix.csv", index=False)
a, b, c = (set(mat[mat.split == s].scaffold) for s in ("train", "val", "test"))
assert not (a & b or a & c or b & c), "SCAFFOLD LEAKAGE"

# --- 5. manifest + WT sanity check on known drugs -----------------------------
manifest = {
    "targets": dict(zip(TARGETS, ["CHEMBL1862", "CHEMBL1936", "CHEMBL1913"])),
    "uniprot": dict(zip(TARGETS, ["P00519", "P10721", "P09619"])),
    "active_cut_pAct": ACTIVE_PACT, "inactive_cut_pAct": INACTIVE_PACT,
    "filters": ["assay_type B only", "non-mutant descriptions",
                "exact medians; '>'>=10uM inactive; '<'<=1uM active",
                "approved drugs (max_phase=4) excluded"],
    "split": "Bemis-Murcko, largest-group-first greedy, deterministic",
    "sizes": {s: int((mat.split == s).sum()) for s in ("train", "val", "test")},
    "sha256": {f: sha256(f"data/{f}") for f in
               ["matrix.csv", "train.csv", "val.csv", "test.csv",
                "approved_drugs.csv"]},
}
try:
    st = requests.get("https://www.ebi.ac.uk/chembl/api/data/status.json",
                      timeout=10).json()
    print("ChEMBL status.json:", st)          # grab the version for the paper
    manifest["chembl_status"] = st
except Exception as e:
    print("status lookup failed:", e)
with open("data/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("manifest.json written")

print("\n=== known drugs, WT-only medians (sanity; official truth = BindingDB) ===")
print(f"{'drug':11s} " + " ".join(f"{l:>8s}" for l in TARGETS))
for name, mid in KNOWN.items():
    print(f"{name:11s} " + " ".join(
        f"{wts[l][mid]:8.3f}" if mid in wts[l] else "     -  " for l in TARGETS))
print("(pAct scale: 6 = 1 uM, 7 = 100 nM, 8 = 10 nM)")