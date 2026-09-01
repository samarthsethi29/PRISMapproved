# PRISM: Polypharmacology Ranking through Integrated Screening of Approved Molecules

PRISM is an open-source tool that ranks approved drugs by **balanced multi-target activity** against a user-defined panel of protein targets. Given published bioactivity data (ChEMBL + BindingDB), it trains a single multi-task graph neural network and screens any compound library — here, 3,386 approved drugs — returning a ranked list scored by a Polypharmacology Score that rewards simultaneous engagement of all targets.

**Validated, not asserted.** PRISM was evaluated on three independent axes against published measurements: scaffold-split internal validation, temporal external validation on post-2016 BindingDB data, and a pre-specified recovery experiment in which the model — which never saw any approved drug during training — had to re-discover the four clinically validated multi-target inhibitors of ABL1 / c-KIT / PDGFRβ. It recovered ponatinib at rank 2 of 3,386.

**Case-study panel (this repo):** ABL1 (CHEMBL1862 / P00519), c-KIT (CHEMBL1936 / P10721), PDGFRβ (CHEMBL1913 / P09619) — the clinically coupled kinase family behind CML (imatinib, nilotinib, dasatinib, ponatinib), GIST, and PDGFRβ-driven proliferative disease. The target panel is a configuration, not code: swapping targets means changing three IDs.

---

## Results at a glance

**1. Internal validation — Bemis–Murcko scaffold split** (test molecules share no Murcko scaffold with training; n = 750):

| Target | n labeled | AUC-ROC | AUPRC | Pos. rate (random AUPRC) | MCC |
|---|---|---|---|---|---|
| ABL1 | 367 | 0.890 | 0.951 | 0.670 | 0.590 |
| c-KIT | 223 | 0.868 | 0.956 | 0.807 | 0.482 |
| PDGFRβ | 235 | 0.843 | 0.912 | 0.660 | 0.502 |
| **Average** | — | **0.867** | **0.940** | — | — |

**2. Retrospective recovery — zero-shot screen of 3,386 approved drugs.** Gold standard fixed *before* training: four approved inhibitors whose clinical indications rest on exactly these targets. No approved drug was in training (asserted programmatically).

| Drug | Full-screen rank | AD rank | AD percentile | PS | Measured pAct (A/K/P) |
|---|---|---|---|---|---|
| Ponatinib | 2 | 2 | 0.4% | 0.996 | 9.43 / 8.77 / 8.92 |
| Imatinib | 72 | 41 | 7.2% | 0.783 | 6.66 / 6.69 / 6.52 |
| Nilotinib | 105 | 58 | 9.4% | 0.710 | 7.55 / 6.80 / 7.19 |
| Dasatinib | 121 | 66 | 10.6% | 0.659 | 8.51 / 7.98 / 7.55 |

EF5% = 5.0 · screen AUROC = 0.935 · all four references in the top 11% of the applicability domain. (Salt forms appear as separate entries — ponatinib and its HCl salt occupy adjacent ranks, an internal consistency check.) Ablation note: the v1 model, trained on ChEMBL alone, ranked imatinib at the 10.5th percentile; the expanded v2 training data moved all references upward.

**3. Temporal external validation — post-2016 BindingDB** (independent database; ligands absent from training by ChEMBL ID and InChIKey; approved drugs excluded; wild-type only). n (AUC) = ligands with active/inactive labels (censored included, gray zone excluded); n (ρ) = ligands with exact measured affinities:

| Target | n (AUC) | % act. | AUC (95% CI) | n (ρ) | Spearman ρ |
|---|---|---|---|---|---|
| ABL1 | 362 | 79.3 | 0.627 (0.55–0.70) | 682 | 0.443 |
| c-KIT | 1,533 | 98.1 | 0.873 (0.79–0.94) | 1,219 | 0.12 |
| PDGFRβ | 745 | 98.3 | 0.719 (0.58–0.83) | 1,165 | 0.679 |

Every target passes at least one external axis. c-KIT's within-active ordering failure is diagnosed (binary-label saturation: median predicted probability 0.90, 50% of external ligands ≥ 0.90, 95th percentile 0.996; within-scaffold ρ = 0.096) — a structural property of binary training, not a data-volume deficiency. A continuous-affinity pilot (same architecture, MSE loss) eliminated saturation and lifted internal ranking to mean ρ = 0.66 but did not transfer externally; see [Limitations](#limitations--roadmap).

**4. Top-10 approved drugs within the applicability domain (by PS):**

| AD rank | Drug | PS | P(ABL1) | P(c-KIT) | P(PDGFRβ) | Max train sim. |
|---|---|---|---|---|---|---|
| 1 | Ponatinib hydrochloride | 0.9967 | 0.9997 | 0.9987 | 0.9916 | 0.933 |
| 2 | Ponatinib | 0.9965 | 0.9996 | 0.9987 | 0.9912 | 0.946 |
| 3 | Sonidegib | 0.9949 | 0.9997 | 0.9935 | 0.9914 | 0.472 |
| 4 | Sonidegib phosphate | 0.9904 | 0.9995 | 0.9924 | 0.9794 | 0.453 |
| 5 | Asciminib hydrochloride | 0.9861 | 1.0000 | 0.9932 | 0.9655 | 0.750 |
| 6 | Asciminib | 0.9847 | 1.0000 | 0.9922 | 0.9622 | 0.761 |
| 7 | Ribociclib | 0.9673 | 0.9433 | 0.9887 | 0.9703 | 0.427 |
| 8 | Olmutinib | 0.9580 | 0.9778 | 0.9697 | 0.9274 | 0.363 |
| 9 | Ribociclib succinate | 0.9512 | 0.8876 | 0.9811 | 0.9882 | 0.398 |
| 10 | Pazopanib hydrochloride | 0.9440 | 0.9859 | 0.9269 | 0.9204 | 0.683 |

Sonidegib (ranks 3–4) is the tool's genuinely novel prediction: a non-kinase drug predicted as a pan-inhibitor of all three targets at low training similarity (0.47). We are not aware of published measurements against these kinases — a testable repurposing hypothesis.

**5. Cross-database ground-truth agreement** (two independent published databases + PRISM predictions on the same drugs):

| Drug | Target | BindingDB (measured) | ChEMBL 37 pAct | PRISM P |
|---|---|---|---|---|
| Imatinib | ABL1 | 300 nM | 6.658 | 0.608 |
| Dasatinib | ABL1 | 1.6 nM | 8.509 | 0.999 |
| Ponatinib | ABL1 | 0.74 nM | 9.432 | 1.000 |
| Imatinib | c-KIT | 385 nM | 6.685 | 0.913 |
| Ponatinib | c-KIT | 8 nM | 8.770 | 0.998 |
| Imatinib | PDGFRβ | 240 nM | 6.517 | 0.865 |
| Dasatinib | PDGFRβ | 28 nM | 7.553 | 0.288 |
| Nilotinib | PDGFRβ | 42.5 nM | 7.185 | 0.471 |
| Ponatinib | PDGFRβ | 1.2 nM | 8.921 | 0.991 |

## Repository structure

```
PRISM/
├── model.py                     # GINEConv multi-task GNN + masked loss (y = [1, K])
├── step2_validate_ligand_data.py    # target verification + ChEMBL pull + QC report
├── step3_build_matrix.py            # labels, approved-drug holdout, scaffold split, manifest
├── step4_train.py                   # training loop (imports model.py; featurizer lives here)
├── step4b_plot_curves.py            # training-curve figure with random baseline
├── step4c_paper_table.py            # paper-ready internal metrics table
├── step5_screen_drugs.py            # approved-drug screen + recovery + EF/AUROC (AUC gate)
├── step5b_applicability_domain.py   # kNN Tanimoto AD module + AD-restricted re-ranking
├── step6_external_validation.py     # temporal external validation on BindingDB
├── step6b_diagnose_kit.py           # c-KIT saturation diagnostic
├── step7_merge_bdb.py               # BindingDB merge + temporal split (v2 data design)
├── data/
│   ├── manifest.json            # curation rules, splits, per-target counts, SHA-256
│   ├── matrix.csv               # 7,495-compound multi-task matrix (approved drugs absent)
│   ├── train.csv / val.csv / test.csv
│   ├── approved_drugs.csv       # 4,225 approved molecules (max_phase = 4)
│   └── raw_ABL1.csv, raw_c-KIT.csv, raw_PDGRB.csv   # raw ChEMBL pulls (see note)
│   └── bindingdb/               # NOT committed — see Setup below (download instructions)
├── models/
│   ├── best_model.pt            # trained checkpoint (1,453,315 parameters)
│   ├── thresholds.json          # per-target MCC-maximizing decision thresholds
│   ├── test_metrics.json        # internal test metrics + config
│   └── train_log.json           # per-epoch training log
├── outputs/
│   ├── drug_screen.csv          # full 3,386-drug screen, ranked
│   ├── ad_drug_screen.csv       # AD-restricted ranking with similarity
│   ├── recovery_table.csv       # gold-standard recovery
│   ├── top10_drugs.csv
│   ├── step5_summary.json, step5b_summary.json, step7_summary.json
│   ├── step6_external.json      # temporal external validation metrics
│   ├── bindingdb_drug_check.csv # cross-database drug check
│   └── test_metrics_table.csv
├── figs/
│   ├── training_curves.png      # loss + validation AUPRC with random baseline
│   ├── umap_embeddings.png      # UMAP of GNN embeddings, colored by PS
│   └── architecture.png         # model diagram (paper Fig. 1)
├── requirements.txt
├── LICENSE (MIT)
└── README.md
```

## Setup

```bash
git clone https://github.com/samarthsethi29/PRISM.git
cd PRISM
pip install -r requirements.txt
```

Python 3.10. Key pins: torch 2.11.0, torch_geometric 2.7.0, rdkit, scikit-learn, scipy, pandas, numpy, chembl-webresource-client, matplotlib. No GPU required — the full pipeline trains and screens on a laptop CPU (~10 min train, minutes for the screen).

**BindingDB files (not committed; ~15 min to obtain):** on bindingdb.org, search UniProt accessions `P00519`, `P10721`, `P09619`, open each human target entry, add all ligand pages, and export the affinity data as TSV. Save as `data/bindingdb/ABL1.tsv`, `KIT.tsv`, `PDGFRB.tsv`.

## Usage

### Run the full pipeline (reproduces every result)

```bash
python step2_validate_ligand_data.py     # verify targets, pull ChEMBL, QC (~10 min)
python step3_build_matrix.py             # labels, drug holdout, scaffold split (~5 min)
python step7_merge_bdb.py                # merge BindingDB, temporal split (~10 min)
python step4_train.py                    # train (CPU ~10 min–2 h; T4 ~10 min)
python step4b_plot_curves.py             # training figures
python step4c_paper_table.py             # internal metrics table
python step5_screen_drugs.py             # screen 3,386 approved drugs (~5 min)
python step5b_applicability_domain.py    # AD module + recovery re-run (~2 min)
python step6_external_validation.py      # temporal external validation (~5 min)
python step6b_diagnose_kit.py            # c-KIT saturation diagnostic (optional)
```

Every split and matrix is SHA-256 checksummed in `data/manifest.json`; a mismatch aborts the run.

### Screen a new compound library with the trained model

```python
from model import MultiTargetGNN
from step4_train import mol_to_graph
import torch
from torch_geometric.loader import DataLoader
import pandas as pd

lib = pd.read_csv("my_library.csv")          # columns: smiles [, name]
graphs = [g for g in (mol_to_graph(s, [-1.]*3) for s in lib.smiles) if g is not None]
ck = torch.load("models/best_model.pt", map_location="cpu")
model = MultiTargetGNN(ck["hidden_dim"], ck["num_layers"], ck["dropout"], ck["targets"])
model.load_state_dict(ck["state_dict"]); model.eval()
P = []
with torch.no_grad():
    for b in DataLoader(graphs, batch_size=256):
        P.append(torch.sigmoid(model(b)).numpy())
P = __import__("numpy").vstack(P)
lib["PS"] = (P.prod(axis=1)) ** (1/3)        # Polypharmacology Score
print(lib.sort_values("PS", ascending=False).head(20))
```

### Change the target panel

Edit the `TARGETS` / ChEMBL-ID / UniProt mapping in `step2`–`step7` (one dict per script) and re-run. The architecture, loss, featurizer, AD module, and scoring are target-agnostic. Note per-target data availability varies — validate any new panel as done here (internal split + external set + a pre-specified recovery standard if one exists).

## Key methods

- **Data curation:** biochemical assays only; wild-type only (mutant-construct assays excluded by description matching); active pAct ≥ 6 (≤1 µM), inactive ≤ 5 (≥10 µM); exact-measurement medians, with censored bounds binarized only when unambiguous; ChEMBL 37 (via API) + BindingDB merged under a temporal rule (pre-2016 → training, post-2016 → frozen external, undated → training, conservative).
- **Model:** 4× GINEConv (edge features), 256 hidden, mean+max pooling, shared MLP, per-target heads; masked multi-task BCE with per-target positive-class weights (0.28/0.23/0.52); 1,453,315 parameters.
- **Polypharmacology Score:** PS = (P_ABL1 · P_c-KIT · P_PDGFRβ)^⅓ — geometric mean; a molecule must be predicted active on all targets to score high.
- **Applicability domain:** Tanimoto ≥ 0.30 (Morgan r=2, 2048-bit) to some training molecule + heavy-atom count within the training 1st–99th percentile. Fixed a priori; both full and AD-restricted rankings reported.

## Limitations & roadmap

1. **c-KIT within-active ordering (ρ = 0.12).** Binary labels carry boundary distance, not potency; probabilities saturate in the active regime. Diagnosed quantitatively (`step6b`): median prediction 0.90, 50% ≥ 0.90, within-scaffold ρ = 0.096. Fix: continuous-affinity (pAct) regression heads — a pilot eliminated saturation (internal mean ρ 0.66) but external transfer still requires potency-diverse training data, which public c-KIT measurements (dominated by mutant and confirmatory assays) currently lack.
2. **Analog presence.** Reference drugs share high Tanimoto similarity (0.93–0.95) with non-approved training analogs — realistic for repurposing, and disclosed; scaffold-split internal validation is the analog-free evidence.
3. **Salt duplicates** occupy adjacent ranks (internal consistency check, small rank inflation).
4. **2D graphs** carry no stereochemical pharmacophores; 3D descriptors and GNNExplainer attribution are planned.
5. **Future:** active-learning loop with experimentally validated candidates (sonidegib first in line); expanding the panel to additional clinically coupled kinases.

## Citation

If you use PRISM, please cite the accompanying paper:

> S. Sethi, D. Shukla, N. Kumar, and V. Ramakrishnan, "PRISM: Polypharmacology Ranking through Integrated Screening of Approved Molecules," 2026.

## References

[1] E. Jabbour and H. Kantarjian, "Chronic myeloid leukemia: 2020 update on diagnosis, therapy, and monitoring," *Am. J. Hematol.*, vol. 95, no. 5, pp. 691–709, 2020.

[2] T. O'Hare, W. C. Shakespeare, X. Zhu *et al.*, "AP24534, a pan-BCR-ABL inhibitor for chronic myeloid leukemia, potently inhibits the T315I mutant and overcomes resistance," *Cancer Cell*, vol. 16, no. 5, pp. 401–412, 2009.

[3] C. L. Corless, J. A. Fletcher, and M. C. Heinrich, "Biology of gastrointestinal stromal tumors," *J. Clin. Oncol.*, vol. 22, no. 18, pp. 3813–3825, 2004.

[4] A. S. Reddy and S. Zhang, "Polypharmacology: Drug discovery for the future," *Expert Rev. Clin. Pharmacol.*, vol. 6, no. 1, pp. 41–47, 2013.

[5] P. de Sena Murteira Pinheiro, L. S. Franco *et al.*, "Molecular hybridization: A powerful tool for multitarget drug discovery," *Expert Opin. Drug Discov.*, 2024. doi: 10.1080/17460441.2024.2322990.

[6] C. McInnes, "Virtual screening strategies in drug discovery," *Curr. Opin. Chem. Biol.*, vol. 11, no. 5, pp. 494–502, 2007.

[7] J. Deng, J. Chen, J. Wang *et al.*, "Retention time prediction of emerging contaminants via transfer learning with graph neural networks," *J. Hazard. Mater.*, 2026.

[8] Y. Wang, J. Wang, Z. Cao *et al.*, "Molecular contrastive learning of representations via graph neural networks," *Nat. Mach. Intell.*, vol. 4, pp. 279–287, 2022.

[9] A. S. Redkar, A. Surendran, B. Rajdev *et al.*, "Multi-omics mechanistic investigation of Yograj Guggulu, an Ayurvedic polyherbal formulation, against MIA-induced osteoarthritis in rats," *J. Ethnopharmacol.*, vol. 370, p. 122058, 2026.

[10] A. Rai, V. Kumar, G. Jerath *et al.*, "Mapping drug-target interactions and synergy in multi-molecular therapeutics for pressure-overload cardiac hypertrophy," *npj Syst. Biol. Appl.*, vol. 7, no. 1, pp. 1–11, 2021.

[11] S. Pushpakom, F. Iorio, P. A. Eyers *et al.*, "Drug repurposing: progress, challenges and recommendations," *Nat. Rev. Drug Discov.*, vol. 18, pp. 41–58, 2019.

[12] A. Gaulton, L. J. Bellis, A. P. Bento, J. Chambers *et al.*, "ChEMBL: A large-scale bioactivity database for drug discovery," *Nucleic Acids Res.*, vol. 40, no. D1, pp. D1100–D1107, 2012.

[13] The UniProt Consortium, "UniProt: the Universal Protein Knowledgebase in 2023," *Nucleic Acids Res.*, vol. 51, no. D1, pp. D523–D531, 2023.

[14] A. P. Bento, A. Hersey, E. Félix, G. Landrum *et al.*, "An open source chemical structure curation pipeline using RDKit," *J. Cheminform.*, vol. 12, p. 51, 2020.

[15] J. Simm, L. Humbeck, A. Zalewski, N. Sturm *et al.*, "Splitting chemical structure data sets for federated privacy-preserving machine learning," *J. Cheminform.*, vol. 13, p. 96, 2021.

[16] M. K. Gilson, T. Liu, M. Baitaluk, G. Nicola, L. Hwang, and J. Chong, "BindingDB in 2015: A public database for medicinal chemistry," *Nucleic Acids Res.*, vol. 44, no. D1, pp. D1045–D1053, 2016.

[17] G. W. Bemis and M. A. Murcko, "The properties of known drugs. 1. Molecular frameworks," *J. Med. Chem.*, vol. 39, no. 15, pp. 2887–2893, 1996.

[18] M. Fey and J. E. Lenssen, "Fast graph representation learning with PyTorch Geometric," *arXiv:1903.02428*, 2019.

[19] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals *et al.*, "Message passing neural networks," in *Machine Learning Meets Quantum Physics*, Springer, 2020, ch. 10.

[20] Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes *et al.*, "MoleculeNet: A benchmark for molecular machine learning," *Chem. Sci.*, vol. 9, pp. 513–530, 2018.

[21] J. Zhou, G. Cui, S. Hu, Z. Zhang *et al.*, "Graph neural networks: A review of methods and applications," *AI Open*, vol. 1, pp. 57–81, 2020.

[22] L. Prechelt, "Early stopping—but when?" in *Neural Networks: Tricks of the Trade*, Springer, 2002, pp. 55–69.

[23] D. Chicco and G. Jurman, "The Matthews correlation coefficient (MCC) should replace the ROC AUC as the standard metric for assessing binary classification," *BioData Mining*, vol. 16, p. 4, 2023.

[24] J. Davis and M. Goadrich, "The relationship between Precision-Recall and ROC curves," in *Proc. 23rd Int. Conf. Machine Learning (ICML)*, pp. 233–240, 2006.

[25] K. K. Mak, Y. H. Wong, and M. R. Pichika, "Artificial intelligence in drug discovery and development," in *Drug Discovery and Evaluation: Safety and Pharmacokinetic Assays*, Springer, 2024.

[26] Z. Ying, D. Bourgeois, J. You, M. Zitnik *et al.*, "GNNExplainer: Generating explanations for graph neural networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

## License

MIT — see [LICENSE](LICENSE).

## Citing PRISM Software:
If you use PRISM software in your research, please cite:
```
@software{prism_sethi_2026,  author = {Sethi, Samarth and Shukla, Dev},  title = {PRISM: Polypharmacology Ranking through Integrated Screening of Approved Molecules},  year = {2026}}
```