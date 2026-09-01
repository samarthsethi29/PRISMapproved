# step2_validate_ligand_data.py
"""
Pull real bioactivity rows for ABL1 / c-KIT / PDGFRB, sanity-check,
deduplicate, and verify known-drug coverage. Saves ./data/raw_*.csv and
./data/clean_*.csv (training-ready snapshots). ~5-10 min, API-bound,
no GPU needed, just internet.
"""
import itertools, os
import numpy as np
import pandas as pd
import requests
from chembl_webresource_client.new_client import new_client

activity_api = new_client.activity
molecule_api = new_client.molecule
target_api   = new_client.target

TRIO = [("ABL1", "CHEMBL1862"), ("c-KIT", "CHEMBL1936"), ("PDGFRB", "CHEMBL1913")]
DRUGS = ["imatinib", "dasatinib", "nilotinib", "ponatinib"]
FIELDS = ["activity_id", "assay_chembl_id", "assay_type", "assay_description",
          "document_chembl_id", "molecule_chembl_id", "canonical_smiles",
          "standard_type", "standard_value", "standard_units", "standard_relation"]

os.makedirs("data", exist_ok=True)

# --- 1. identity lock -----------------------------------------------------
print("=== identity lock ===")
for label, tid in TRIO:
    r = list(target_api.filter(target_chembl_id=tid))
    print(f"  {label:7s} {tid} -> {r[0]['pref_name']} ({r[0]['organism']})")

# --- provenance for the paper ---------------------------------------------
try:
    ver = requests.get("https://www.ebi.ac.uk/chembl/api/data/status.json",
                       timeout=10).json()["chembl_version"]
    print(f"\nChEMBL release: {ver}  <- cite this + today's date in the paper")
except Exception as e:
    print("ChEMBL version lookup failed:", e)

# --- 2. pull raw rows (pagination = the slow part) -------------------------
raw = {}
for label, tid in TRIO:
    print(f"\n=== pulling {label} ({tid}) ===")
    frames = []
    for stype in ("IC50", "Ki"):
        rows = list(activity_api.filter(target_chembl_id=tid,
                                        standard_type=stype).only(FIELDS))
        print(f"   {stype:4s}: {len(rows)} rows")
        frames.append(pd.DataFrame(rows, columns=FIELDS))
    raw[label] = pd.concat(frames, ignore_index=True)
    raw[label].to_csv(f"data/raw_{label}.csv", index=False)

# --- 3. per-target cleaning + quality report -------------------------------
def clean_target(df, label):
    print(f"\n===== {label}: quality report =====")
    print("rows pulled      :", len(df))
    print("relation split   :", dict(df.standard_relation.value_counts(dropna=False)))
    print("units split      :", dict(df.standard_units.value_counts(dropna=False)))
    print("assay_type split :", dict(df.assay_type.value_counts(dropna=False)))

    d = df[(df.standard_relation == "=") & (df.standard_units == "nM")].copy()
    d["value_nM"] = pd.to_numeric(d.standard_value, errors="coerce")
    d = d.dropna(subset=["value_nM"])
    d = d[d.value_nM.between(0.01, 1e7)]          # sanity window: 10 pM .. 10 mM
    d = d[d.canonical_smiles.notna()]

    nmut = d.assay_description.fillna("").str.contains(
        r"mutant|t315|d816|v560", case=False, regex=True).sum()
    print(f"mutant-assay rows (desc. scan): {nmut} ({100*nmut/max(len(d),1):.1f}%) "
          "-> flag if >5%")

    d["pAct"] = -np.log10(d.value_nM * 1e-9)
    ded = (d.groupby("molecule_chembl_id")
             .agg(pAct=("pAct", "median"),
                  value_nM=("value_nM", "median"),
                  n_meas=("pAct", "size"),
                  smiles=("canonical_smiles", "first"),
                  std_type=("standard_type", "first"))
             .reset_index())
    print(f"distinct compounds (dedup, median pAct): {len(ded)}")

    q = ded.pAct.describe()[["min", "25%", "50%", "75%", "max"]]
    print("pAct spread      :", {k: round(v, 2) for k, v in q.items()})
    for a_cut, i_cut, desc in [(7, 5, "active<=100nM, inactive>=10uM"),
                               (6, 5, "active<=1uM,   inactive>=10uM")]:
        na, ni = (ded.pAct >= a_cut).sum(), (ded.pAct <= i_cut).sum()
        print(f"labels [{desc}]: actives={na}, inactives={ni}, gray={len(ded)-na-ni}")

    ded.to_csv(f"data/clean_{label}.csv", index=False)
    return ded

cleaned = {lab: clean_target(raw[lab], lab) for lab, _ in TRIO}

# --- 4. multi-task overlap --------------------------------------------------
labs = [l for l, _ in TRIO]
print("\n===== overlap between targets =====")
sets = {l: set(cleaned[l].molecule_chembl_id) for l in labs}
for a, b in itertools.combinations(labs, 2):
    print(f"  {a:7s} n {b:7s}: {len(sets[a] & sets[b])}")
print(f"  all three        : {len(set.intersection(*sets.values()))}")

m = cleaned[labs[0]][["molecule_chembl_id", "pAct"]].rename(columns={"pAct": labs[0]})
for l in labs[1:]:
    m = m.merge(cleaned[l][["molecule_chembl_id", "pAct"]].rename(columns={"pAct": l}),
                on="molecule_chembl_id", how="outer")
comp = m.dropna()
print(f"compounds measured on all 3: {len(comp)}")
for cut in (6, 7):
    print(f"   active on all 3 at pAct>={cut} ({'<=1uM' if cut==6 else '<=100nM'}): "
          f"{(comp[labs] >= cut).all(axis=1).sum()}")

# --- 5. ground-truth preview: known drugs ------------------------------------
print("\n===== known drugs in OUR pulled data (median of exact measurements) =====")
print(f"{'drug':11s} {'ChEMBL ID':14s} {'ABL1':>10s} {'c-KIT':>10s} {'PDGFRB':>10s}")
for name in DRUGS:
    hits = list(molecule_api.search(name))[:10]
    mol = next((h for h in hits if (h.get("pref_name") or "").upper() == name.upper()),
               hits[0] if hits else None)
    mid = mol["molecule_chembl_id"] if mol else "NOT FOUND"
    cells = []
    for l in labs:
        sub = cleaned[l].loc[cleaned[l].molecule_chembl_id == mid]
        cells.append(f"{sub.value_nM.iloc[0]:.3g} nM" if len(sub) else "-")
    print(f"{name:11s} {mid:14s} {cells[0]:>10s} {cells[1]:>10s} {cells[2]:>10s}")

print("\nDone. data/clean_*.csv are training-ready snapshots for Step 3.")