"""
step7_merge_bdb.py — Option B: ChEMBL+BindingDB merged training, temporal
external holdout. One-shot design: cutoff fixed by data volume before metrics.

Reads : data/matrix.csv, data/approved_drugs.csv, data/bindingdb/*.tsv
Writes: data/matrix.csv (merged), data/train|val|test.csv, data/manifest.json
        (v3 appended), outputs/step7_summary.json
"""
import json, os, re, shutil
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
TARGETS = ["ABL1", "c-KIT", "PDGFRB"]
ACTIVE_PACT, INACTIVE_PACT = 6.0, 5.0
CUTOFF = pd.Timestamp("2016-01-01")     # adjust by VOLUME only (see printout)
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1
FILES = {"ABL1": ("data/bindingdb/ABL1.tsv", "P00519"),
         "c-KIT": ("data/bindingdb/KIT.tsv", "P10721"),
         "PDGFRB": ("data/bindingdb/PDGFRB.tsv", "P09619")}
KNOWN = {"CHEMBL941", "CHEMBL5416410", "CHEMBL255863", "CHEMBL1171837"}
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
os.makedirs("outputs", exist_ok=True)

def find_col(cols, *keys):
    for c in cols:
        if all(k in str(c).lower() for k in keys):
            return c
    return None

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
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    fr = Chem.GetMolFrags(m, asMols=True)
    return Chem.MolToSmiles(max(fr, key=lambda f: f.GetNumAtoms())) if fr else None

def skel_of(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m).split("-")[0] if m else None

def parse_target(tgt, path, acc):
    hdr = pd.read_csv(path, sep="\t", nrows=0)
    sep = "," if hdr.shape[1] < 5 else "\t"
    cols = list(hdr.columns)
    c_smiles = find_col(cols, "ligand", "smiles")
    c_ikey  = find_col(cols, "inchi key")
    c_chem  = find_col(cols, "chembl id of ligand")
    c_ki, c_ic50, c_kd = (find_col(cols, "ki (nm)"), find_col(cols, "ic50 (nm)"),
                          find_col(cols, "kd (nm)"))
    c_org, c_uni = find_col(cols, "source organism"), find_col(cols, "swissprot", "primary id", "chain 1")
    c_name, c_date = find_col(cols, "target name"), find_col(cols, "date of publication")
    use = [c for c in (c_smiles, c_ikey, c_chem, c_ki, c_ic50, c_kd,
                       c_org, c_uni, c_name, c_date) if c]
    df = pd.read_csv(path, sep=sep, dtype=str, on_bad_lines="skip",
                     usecols=use, engine="python", quoting=3)
    if c_org:
        df = df[df[c_org].fillna("").str.lower().str.contains("human|homo sapiens")]
    if c_uni:
        df = df[df[c_uni].fillna("").str.contains(acc, na=False)
                | df[c_uni].fillna("").eq("")]
    if c_name:
        mut = df[c_name].fillna("").str.contains(
            r"mutant|mutation|[A-Za-z]\d{3}[A-Za-z]", regex=True)
        df = df[~mut]
    lig = {}
    for _, r in df.iterrows():
        smi = largest_fragment(r[c_smiles]) if isinstance(r[c_smiles], str) else None
        if not smi: continue
        iw = r.get(c_ikey)
        skel = (str(iw).split("-")[0] if isinstance(iw, str) and iw else skel_of(smi))
        if not skel: continue
        rec = lig.setdefault(skel, {"smi": smi, "chembl": set(),
                                    "ex": [], "kd": [], "cens": [], "date": None})
        if c_chem and isinstance(r.get(c_chem), str) and r[c_chem].strip():
            rec["chembl"].add(r[c_chem].strip())
        d = pd.to_datetime(r.get(c_date), errors="coerce")
        if pd.notna(d):
            rec["date"] = d if rec["date"] is None else min(rec["date"], d)
        for c in (c_ki, c_ic50):
            q, v = parse_aff(r.get(c))
            if v is not None and 0.01 <= v <= 1e7:
                (rec["ex"] if q is None else rec["cens"]).append(v if q is None else (q, v))
        if c_kd:
            q, v = parse_aff(r.get(c_kd))
            if v is not None and 0.01 <= v <= 1e7 and q is None:
                rec["kd"].append(v)
    return lig

def label(rec):
    if rec["ex"]:
        p = float(np.median([to_pact(v) for v in rec["ex"]]))
        return (1 if p >= ACTIVE_PACT else 0 if p <= INACTIVE_PACT else None)
    if rec["kd"]:
        p = float(np.median([to_pact(v) for v in rec["kd"]]))
        return (1 if p >= ACTIVE_PACT else 0 if p <= INACTIVE_PACT else None)
    for q, v in rec["cens"]:
        if q == ">" and v >= 10_000: return 0
        if q == "<" and v <=  1_000: return 1
    return None

# ── pools ─────────────────────────────────────────────────────────────────────
base = pd.read_csv("data/matrix.csv")
base["skel"] = [skel_of(s) for s in base.smiles]
base_skels = set(base.skel.dropna())
base_chem  = set(base.molecule_chembl_id.dropna())
ap = pd.read_csv("data/approved_drugs.csv").dropna(subset=["smiles"])
ap_chem = set(ap.molecule_chembl_id)
ap_skel = {k for k in (skel_of(s) for s in ap.smiles) if k}
print(f"base matrix: {len(base)} | approved: {len(ap_chem)} ids / {len(ap_skel)} skeletons")

# ── per-target merge decisions ───────────────────────────────────────────────
new_labels, heldout, stats = {}, {}, {}
for tgt, (path, acc) in FILES.items():
    lig = parse_target(tgt, path, acc)
    n_in_base = n_appr = n_undated = n_pre = n_post = 0
    pre_act = post_act = 0
    for skel, rec in lig.items():
        if skel in base_skels or (rec["chembl"] & base_chem):
            n_in_base += 1; continue                      # ChEMBL label wins
        if skel in ap_skel or (rec["chembl"] & ap_chem):
            n_appr += 1; continue                         # approved: never train
        y = label(rec)
        if y is None: continue
        if rec["date"] is None:
            n_undated += 1                                # conservative -> train
        elif rec["date"] > CUTOFF:
            n_post += 1; post_act += y
            heldout.setdefault(tgt, set()).add(skel)
            continue
        else:
            n_pre += 1; pre_act += y
        new_labels[(skel, tgt)] = y
    stats[tgt] = {"bdb_ligands": len(lig), "skipped_in_base": n_in_base,
                  "skipped_approved": n_appr, "undated_to_train": n_undated,
                  "pre_cutoff_added": n_pre, "pre_actives": pre_act,
                  "post_cutoff_heldout": n_post, "post_actives": post_act}
    print(f"{tgt}: {json.dumps(stats[tgt])}")
    # volume guide for the ONE cutoff decision (pick before any metrics):
    if n_post < 250:
        print(f"   !! {tgt}: only {n_post} post-cutoff labeled — consider an "
              f"earlier CUTOFF for volume (e.g. 2013) BEFORE proceeding")

# ── build merged matrix ───────────────────────────────────────────────────────
new_skels = sorted({s for s, _ in new_labels})
new_df = pd.DataFrame({"molecule_chembl_id": "", "smiles": "",
                       "skel": new_skels})
smi_map = {}
for tgt, (path, acc) in FILES.items():   # recover a SMILES per new skeleton
    pass  # (filled during parse below via lig dict retained)
# simpler: re-derive from files once more is wasteful; capture during parse:
# -> we capture smiles inside new_labels pass instead:
smi_map = {}
for tgt, (path, acc) in FILES.items():
    for skel, rec in parse_target(tgt, path, acc).items():
        if (skel, tgt) in new_labels and skel not in smi_map:
            smi_map[skel] = rec["smi"]
new_df["smiles"] = new_df.skel.map(smi_map)
for t in TARGETS:
    new_df[t] = [new_labels.get((s, t), -1) for s in new_df.skel]
merged = pd.concat([base[["molecule_chembl_id", "smiles", "skel"] + TARGETS],
                    new_df[["molecule_chembl_id", "smiles", "skel"] + TARGETS]],
                   ignore_index=True)
merged = merged[merged.smiles.notna()]
merged = merged[merged.smiles.apply(lambda s: Chem.MolFromSmiles(s) is not None)]
assert not (set(KNOWN) & set(merged.molecule_chembl_id.dropna())), "DRUG LEAK"
print(f"\nmerged matrix: {len(merged)} compounds "
      f"(+{len(merged) - len(base)} from BindingDB pre-cutoff)")

# ── scaffold split (deterministic, largest groups first) ─────────────────────
def scaffold_of(s):
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
        return sc if sc else s
    except Exception:
        return s
merged["scaffold"] = merged.smiles.apply(scaffold_of)
order = merged.groupby("scaffold").size().sort_values(ascending=False)
n = len(merged); n_tr, n_va = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
tr = va = te = 0; assign = {}
for sc, sz in order.items():
    r = [n_tr - tr, n_va - va, n - n_tr - n_va - te]
    pick = r.index(max(r)); assign[sc] = ["train", "val", "test"][pick]
    tr += sz if pick == 0 else 0; va += sz if pick == 1 else 0; te += sz if pick == 2 else 0
merged["split"] = merged.scaffold.map(assign)
a, b, c = (set(merged[merged.split == s].scaffold) for s in ("train", "val", "test"))
assert not (a & b or a & c or b & c), "SCAFFOLD LEAKAGE"
for s in ("train", "val", "test"):
    sub = merged[merged.split == s]; sub.to_csv(f"data/{s}.csv", index=False)
    print(f"  {s}: {len(sub)} | " + str({t: (int((sub[t]==1).sum()),
          int((sub[t]==0).sum())) for t in TARGETS}))
# temporal integrity: held-out skeletons carry no label in the matrix
for t in TARGETS:
    labeled = {s for s in merged.skel if merged.loc[merged.skel == s, t].iloc[0] != -1} \
        if False else {row.skel for _, row in merged.iterrows() if row[t] != -1}
    assert not (heldout.get(t, set()) & labeled), f"TEMPORAL LEAK {t}"
merged.to_csv("data/matrix.csv", index=False)

# ── manifest v3 ───────────────────────────────────────────────────────────────
man = json.load(open("data/manifest.json"))
man["v3_merge"] = {"cutoff": str(CUTOFF.date()),
                   "training_sources": "ChEMBL_37 + BindingDB pre-cutoff",
                   "external_design": "BindingDB-only, earliest publication > cutoff",
                   "undated_policy": "assigned to training (conservative)",
                   "per_target": stats,
                   "matrix_size": int(len(merged)),
                   "split_sizes": {s: int((merged.split == s).sum())
                                   for s in ("train", "val", "test")}}
json.dump(man, open("data/manifest.json", "w"), indent=2)
json.dump({"cutoff": str(CUTOFF.date()), "per_target": stats,
           "matrix_size": int(len(merged)),
           "heldout_counts": {t: len(v) for t, v in heldout.items()}},
          open("outputs/step7_summary.json", "w"), indent=2)
print(f"\nmanifest v3 + step7_summary.json written | external holdout sizes: "
      f"{ {t: len(v) for t, v in heldout.items()} }")
print("NEXT: retrain -> step4_train.py (models/ already archived to models_v2)")