# step6b_diagnose_kit.py — evidence for the c-KIT ranking boundary.
# python step6b_diagnose_kit.py   (~2 min, data already on disk)
import re
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader
from model import MultiTargetGNN
from step4_train import mol_to_graph, TARGETS

RDLogger.DisableLog("rdApp.*")
ACC, PATH = "P10721", "data/bindingdb/KIT.tsv"
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
MUT = r"mutant|mutation|[A-Za-z]\d{3}[A-Za-z]"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_col(cols, *keys):
    for c in cols:
        if all(k in str(c).lower() for k in keys):
            return c

def parse_aff(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return None, None
    s = str(x).strip()
    if not s or s.lower() in ("-", "nan", "none", "n/a"): return None, None
    q = s[0] if s[0] in "<>" else None
    if q: s = s[1:]
    m = re.search(NUM, s)
    return (q, float(m.group())) if m else (None, None)

def largest_fragment(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    fr = Chem.GetMolFrags(m, asMols=True)
    return Chem.MolToSmiles(max(fr, key=lambda f: f.GetNumAtoms())) if fr else None

def skel_of(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m).split("-")[0] if m else None

mat = pd.read_csv("data/matrix.csv")
pool_chembl = {c for c in mat.molecule_chembl_id.dropna().astype(str) if c}
pool_keys = {k for k in (skel_of(s) for s in mat.smiles) if k}
ap = pd.read_csv("data/approved_drugs.csv").dropna(subset=["smiles"])
ap_chem = {c for c in ap.molecule_chembl_id.dropna().astype(str)}
ap_skel = {k for k in (skel_of(s) for s in ap.smiles) if k}

hdr = pd.read_csv(PATH, sep="\t", nrows=0)
sep = "," if hdr.shape[1] < 5 else "\t"
cols = list(hdr.columns)
c_smiles = find_col(cols, "ligand", "smiles")
c_ikey, c_chem = find_col(cols, "inchi key"), find_col(cols, "chembl id of ligand")
c_ki, c_ic50 = find_col(cols, "ki (nm)"), find_col(cols, "ic50 (nm)")
c_org, c_uni, c_name = (find_col(cols, "source organism"),
                         find_col(cols, "swissprot", "primary id", "chain 1"),
                         find_col(cols, "target name"))
use = [c for c in (c_smiles, c_ikey, c_chem, c_ki, c_ic50, c_org, c_uni, c_name) if c]
df = pd.read_csv(PATH, sep=sep, dtype=str, on_bad_lines="skip", usecols=use,
                 engine="python", quoting=3)
df = df[df[c_org].fillna("").str.lower().str.contains("human|homo sapiens")]
df = df[df[c_uni].fillna("").str.contains(ACC, na=False) | df[c_uni].fillna("").eq("")]
df = df[~df[c_name].fillna("").str.contains(MUT, regex=True)]

recs = {}
for _, r in df.iterrows():
    smi = largest_fragment(r[c_smiles]) if isinstance(r[c_smiles], str) else None
    if not smi: continue
    iw = r.get(c_ikey)
    skel = (str(iw).split("-")[0] if isinstance(iw, str) and iw else skel_of(smi))
    if not skel: continue
    rec = recs.setdefault(skel, {"smi": smi, "chembl": set(), "ex": []})
    if c_chem and isinstance(r.get(c_chem), str) and r[c_chem].strip():
        rec["chembl"].add(r[c_chem].strip())
    for c in (c_ki, c_ic50):
        q, v = parse_aff(r.get(c))
        if q is None and v is not None and 0.01 <= v <= 1e7:
            rec["ex"].append(v)
bdb = pd.DataFrame([{"skel": k, "smi": v["smi"], "chembl": ";".join(v["chembl"]),
                     "pAct": (float(np.median([-np.log10(x * 1e-9) for x in v["ex"]]))
                              if v["ex"] else np.nan)}
                    for k, v in recs.items()])
bdb = bdb[~(bdb.chembl.apply(lambda s: any(c in pool_chembl for c in s.split(";") if c))
             | bdb.skel.isin(pool_keys)
             | bdb.chembl.apply(lambda s: any(c in ap_chem for c in s.split(";") if c))
             | bdb.skel.isin(ap_skel))].reset_index(drop=True)
print(f"c-KIT external set (same rule as step 6): {len(bdb)} ligands, "
      f"{int(bdb.pAct.notna().sum())} with exact affinity")

ck = torch.load("models/best_model.pt", map_location=DEVICE)
model = MultiTargetGNN(hidden_dim=ck["hidden_dim"], num_layers=ck["num_layers"],
                       dropout=ck["dropout"], target_names=ck["targets"]).to(DEVICE)
model.load_state_dict(ck["state_dict"]); model.eval()
graphs, idx = [], []
for i, s in enumerate(bdb.smi):
    g = mol_to_graph(s, [-1.0] * len(TARGETS))
    if g is not None: graphs.append(g); idx.append(i)
probs = []
with torch.no_grad():
    for b in DataLoader(graphs, batch_size=256):
        probs.append(torch.sigmoid(model(b.to(DEVICE))).cpu().numpy())
bdb = bdb.iloc[idx].copy()
bdb["p"] = np.vstack(probs)[:, TARGETS.index("c-KIT")]

print(f"\n[1] prediction quantiles: "
      f"{bdb.p.quantile([0.05,0.25,0.5,0.75,0.95]).round(3).to_dict()}")
print(f"    share of predictions >= 0.90: {float((bdb.p>=0.9).mean()*100):.1f}%")
ex = bdb.dropna(subset=["pAct"]).copy()
print(f"[2] measured pAct range: {ex.pAct.min():.2f} - {ex.pAct.max():.2f} "
      f"(median {ex.pAct.median():.2f})")
print(f"[3] overall Spearman rho: {spearmanr(ex.pAct, ex.p)[0]:.3f} (n={len(ex)})")
ex["scaffold"] = ex.smi.apply(lambda s: MurckoScaffold.MurckoScaffoldSmiles(
    smiles=s, includeChirality=False) or s)
groups = [g for _, g in ex.groupby("scaffold") if len(g) >= 5
          and g.pAct.nunique() > 1 and g.p.nunique() > 1]
within = [spearmanr(g.pAct, g.p)[0] for g in groups]
gm = ex.groupby("scaffold").agg(p=("p", "mean"), pAct=("pAct", "mean"))
across = spearmanr(gm.pAct, gm.p)[0] if len(gm) >= 5 else float("nan")
print(f"[4] scaffold groups >=5 ligands: {len(groups)} "
      f"(covering {sum(len(g) for g in groups)} ligands)")
if within:
    print(f"    within-scaffold rho (analog ranking): median "
          f"{np.nanmedian(within):.3f}")
print(f"    across-scaffold rho (chemotype ranking): {across:.3f}")
print("\nInterpretation: high saturation + ~0 within-scaffold rho = the binary")
print("classifier has no within-active resolution (data volume cannot fix this);")
print("if across-scaffold rho is decent, chemotype-level triage still works.")