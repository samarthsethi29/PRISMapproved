"""
step6_external_validation.py — TEMPORAL external validation (v3).

Training pool (from step 7) = ChEMBL_37 + BindingDB ligands published
before the cutoff (+ undated, conservatively assigned to training).
This script therefore evaluates ONLY on BindingDB ligands that are:
  (i)   not in the merged training pool (ChEMBL ID or InChIKey skeleton),
  (ii)  not approved drugs (max_phase=4 — recovery experiment's territory),
  (iii) human, wild-type (Target-Name regex, matching training curation).
External set is post-cutoff by construction.

Reads : data/bindingdb/{ABL1,KIT,PDGFRB}.tsv, data/matrix.csv (merged v3),
        data/approved_drugs.csv, models/best_model.pt (RETRAINED v3),
        outputs/drug_screen.csv (new step 5), outputs/step7_summary.json
Writes: outputs/step6_external.json, outputs/bindingdb_drug_check.csv
"""
import json, os, re, sys
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import DataLoader
from model import MultiTargetGNN
from step4_train import mol_to_graph, TARGETS

RDLogger.DisableLog("rdApp.*")
ACTIVE_PACT, INACTIVE_PACT = 6.0, 5.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILES = {"ABL1": ("data/bindingdb/ABL1.tsv", "P00519"),
         "c-KIT": ("data/bindingdb/KIT.tsv", "P10721"),
         "PDGFRB": ("data/bindingdb/PDGFRB.tsv", "P09619")}
GT = {"imatinib":  ("CHEMBL941",     (6.658, 6.685, 6.517)),
      "dasatinib": ("CHEMBL5416410", (8.509, 7.983, 7.553)),
      "nilotinib": ("CHEMBL255863",  (7.553, 6.801, 7.185)),
      "ponatinib": ("CHEMBL1171837", (9.432, 8.770, 8.921))}
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
MUTANT_RE = r"mutant|mutation|[A-Za-z]\d{3}[A-Za-z]"   # same as step 7

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

def auc_ci(y, p, n_boot=1000, seed=42):
    y, p = np.asarray(y, int), np.asarray(p)
    rng = np.random.default_rng(seed); aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            aucs.append(roc_auc_score(y[i], p[i]))
    return f"{np.percentile(aucs,2.5):.2f}-{np.percentile(aucs,97.5):.2f}"

# ── pools ─────────────────────────────────────────────────────────────────────
mat = pd.read_csv("data/matrix.csv")
if len(mat) <= 6400:
    sys.exit("data/matrix.csv looks like the PRE-MERGE (v2) matrix — "
             "run step7_merge_bdb.py first, then retrain (step4), then step5.")
pool_chembl = {c for c in mat.molecule_chembl_id.dropna().astype(str) if c}
pool_keys = {k for k in (skel_of(s) for s in mat.smiles) if k}
ap = pd.read_csv("data/approved_drugs.csv").dropna(subset=["smiles"])
ap_chem = {c for c in ap.molecule_chembl_id.dropna().astype(str)}
ap_skel = {k for k in (skel_of(s) for s in ap.smiles) if k}
try:
    cut = json.load(open("outputs/step7_summary.json"))["cutoff"]
    print(f"temporal design confirmed: external = BindingDB-only, first "
          f"published after {cut} | pool: {len(mat)} merged mols, "
          f"{len(pool_keys)} skeletons | approved excluded: {len(ap_chem)} ids")
except Exception:
    print("warning: step7_summary.json missing — proceeding with merged-matrix pool")

ds = pd.read_csv("outputs/drug_screen.csv")          # PRISM probs for drug table
P_by_mid = {str(r.molecule_chembl_id): r for _, r in ds.iterrows()}

# ── model (must be the RETRAINED one) ─────────────────────────────────────────
ck = torch.load("models/best_model.pt", map_location=DEVICE)
model = MultiTargetGNN(hidden_dim=ck["hidden_dim"], num_layers=ck["num_layers"],
                       dropout=ck["dropout"], target_names=ck["targets"]).to(DEVICE)
model.load_state_dict(ck["state_dict"]); model.eval()

def predict(smiles_list, k):
    graphs, idx = [], []
    for i, s in enumerate(smiles_list):
        g = mol_to_graph(s, [-1.0] * len(TARGETS))
        if g is not None: graphs.append(g); idx.append(i)
    probs = []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=256):
            probs.append(torch.sigmoid(model(b.to(DEVICE))).cpu().numpy())
    return idx, (np.vstack(probs)[:, k] if probs else np.array([]))

results, drug_rows = {}, []
for tgt, (path, acc) in FILES.items():
    if not os.path.exists(path):
        sys.exit(f"MISSING {path}")
    hdr = pd.read_csv(path, sep="\t", nrows=0)
    sep = "," if hdr.shape[1] < 5 else "\t"
    cols = list(hdr.columns)
    c_smiles = find_col(cols, "ligand", "smiles")
    c_ikey, c_chem = find_col(cols, "inchi key"), find_col(cols, "chembl id of ligand")
    c_ki, c_ic50, c_kd = (find_col(cols, "ki (nm)"), find_col(cols, "ic50 (nm)"),
                          find_col(cols, "kd (nm)"))
    c_org = find_col(cols, "source organism")
    c_uni = find_col(cols, "swissprot", "primary id", "chain 1")
    c_name, c_lname = find_col(cols, "target name"), find_col(cols, "ligand name")
    use = [c for c in (c_smiles, c_ikey, c_chem, c_ki, c_ic50, c_kd,
                       c_org, c_uni, c_name, c_lname) if c]
    df = pd.read_csv(path, sep=sep, dtype=str, on_bad_lines="skip",
                     usecols=use, engine="python", quoting=3)
    print(f"\n=== {tgt} ({acc}) === rows: {len(df)}")
    if c_org:
        df = df[df[c_org].fillna("").str.lower().str.contains("human|homo sapiens")]
        print(f"  human rows: {len(df)}")
    if c_uni:
        off = int((df[c_uni].fillna("").ne("") &
                   ~df[c_uni].fillna("").str.contains(acc, na=False)).sum())
        df = df[df[c_uni].fillna("").str.contains(acc, na=False)
                | df[c_uni].fillna("").eq("")]
        print(f"  UniProt filter: {off} off-target rows dropped -> {len(df)}")
    if c_name:   # CHANGE: mutant filter — training data is WT-only (step 7)
        mut = df[c_name].fillna("").str.contains(MUTANT_RE, regex=True)
        print(f"  mutant-assay rows dropped: {int(mut.sum())} -> {len(df) - int(mut.sum())}")
        df = df[~mut]

    recs = {}
    for _, r in df.iterrows():
        smi = largest_fragment(r[c_smiles]) if isinstance(r[c_smiles], str) else None
        if not smi: continue
        iw = r.get(c_ikey)
        skel = (str(iw).split("-")[0] if isinstance(iw, str) and iw else skel_of(smi))
        if not skel: continue
        rec = recs.setdefault(skel, {"smi": smi, "chembl": set(), "name": "",
                                     "ex": [], "kd": [], "cens": []})
        if c_chem and isinstance(r.get(c_chem), str) and r[c_chem].strip():
            rec["chembl"].add(r[c_chem].strip())
        if c_lname and isinstance(r.get(c_lname), str) and not rec["name"]:
            rec["name"] = r[c_lname]
        for c in (c_ki, c_ic50):
            q, v = parse_aff(r.get(c))
            if v is not None and 0.01 <= v <= 1e7:
                (rec["ex"] if q is None else rec["cens"]).append(
                    v if q is None else (q, v))
        if c_kd:
            q, v = parse_aff(r.get(c_kd))
            if v is not None and 0.01 <= v <= 1e7 and q is None:
                rec["kd"].append(v)
    n0 = len(recs)

    def label(rec):
        if rec["ex"]:
            p = float(np.median([to_pact(v) for v in rec["ex"]]))
            return (1 if p >= ACTIVE_PACT else 0 if p <= INACTIVE_PACT else None), p, "Ki/IC50"
        if rec["kd"]:
            p = float(np.median([to_pact(v) for v in rec["kd"]]))
            return (1 if p >= ACTIVE_PACT else 0 if p <= INACTIVE_PACT else None), p, "Kd"
        for q, v in rec["cens"]:
            if q == ">" and v >= 10_000: return 0, None, "censored"
            if q == "<" and v <=  1_000:  return 1, None, "censored"
        return None, None, None

    rows = [{"skel": k, "smi": v["smi"], "chembl": ";".join(v["chembl"]),
             "y": label(v)[0], "pAct": label(v)[1], "type": label(v)[2]}
            for k, v in recs.items()]
    bdb = pd.DataFrame(rows)
    in_pool = (bdb.chembl.apply(lambda s: any(c in pool_chembl for c in s.split(";") if c))
               | bdb.skel.isin(pool_keys))
    is_appr = (bdb.chembl.apply(lambda s: any(c in ap_chem for c in s.split(";") if c))
               | bdb.skel.isin(ap_skel))
    print(f"  distinct ligands: {n0} | pool overlap removed: {int(in_pool.sum())} "
          f"| approved removed: {int((~in_pool & is_appr).sum())} "
          f"| external: {int((~in_pool & ~is_appr).sum())}")
    bdb = bdb[~in_pool & ~is_appr].reset_index(drop=True)

    idx, P = predict(list(bdb.smi), TARGETS.index(tgt))
    bdb = bdb.iloc[idx].copy(); bdb["p"] = P

    def auc_stats(sub):
        lab = sub.dropna(subset=["y"])
        if lab.y.nunique() < 2: return None
        y, p = lab.y.astype(int).values, lab.p.values
        return {"n": int(len(lab)), "act": int(y.sum()), "inact": int((y == 0).sum()),
                "pos_rate": round(float(y.mean()), 3),
                "AUC": round(float(roc_auc_score(y, p)), 3),
                "AUC_95CI": auc_ci(y, p),
                "AUPRC": round(float(average_precision_score(y, p)), 3)}
    out = {"n_external": int(len(bdb)),
           "type_counts": bdb.type.value_counts().to_dict()}
    out["AUC_full"] = auc_stats(bdb)
    out["AUC_KiIC50_only"] = auc_stats(bdb[bdb.type == "Ki/IC50"])
    ex = bdb.dropna(subset=["pAct"])
    if len(ex) >= 20:
        out["Spearman_rho"] = round(float(spearmanr(ex.pAct, ex.p)[0]), 3)
        out["Spearman_n"] = int(len(ex))
    ex_k = bdb[bdb.type == "Ki/IC50"]
    if len(ex_k) >= 20:
        out["Spearman_rho_KiIC50_only"] = round(float(spearmanr(ex_k.pAct, ex_k.p)[0]), 3)
    results[tgt] = out
    print("  " + json.dumps(out, default=str))

    # drug cross-check: BDB values from file, PRISM P from the step-5 screen
    for name, (mid, pacts) in GT.items():
        key = next((k for k, v in recs.items() if mid in v["chembl"]), None)
        if key is None:      # CHANGE: name fallback -> catches dasatinib
            key = next((k for k, v in recs.items()
                        if name.upper() in v.get("name", "").upper()), None)
        if key is None:
            drug_rows.append({"drug": name, "target": tgt, "BindingDB": "not in file",
                              "PRISM_P": None, "ChEMBL37_pAct": dict(zip(TARGETS, pacts))[tgt]})
            continue
        rec = recs[key]
        if rec["ex"]:   meas, typ = f"{np.median(rec['ex']):.3g} nM", "Ki/IC50"
        elif rec["kd"]: meas, typ = f"{np.median(rec['kd']):.3g} nM", "Kd"
        else:           meas, typ = "-", "-"
        P_val = (round(float(P_by_mid[mid][f"P_{tgt}"]), 3)
                 if mid in P_by_mid else None)
        drug_rows.append({"drug": name, "target": tgt,
                          "BindingDB": f"{meas} ({typ})", "PRISM_P": P_val,
                          "ChEMBL37_pAct": dict(zip(TARGETS, pacts))[tgt]})

print("\n===== TEMPORAL EXTERNAL VALIDATION (BindingDB, post-cutoff, "
      "no approved drugs, no training overlap) =====")
print(pd.DataFrame(results).T.to_string())
dr = pd.DataFrame(drug_rows)
if not dr.empty:
    dr.to_csv("outputs/bindingdb_drug_check.csv", index=False)
    print("\nCross-database check (BindingDB vs ChEMBL_37 vs PRISM):")
    print(dr.to_string(index=False))
json.dump(results, open("outputs/step6_external.json", "w"), indent=2, default=str)
print("\nwrote outputs/step6_external.json, outputs/bindingdb_drug_check.csv")