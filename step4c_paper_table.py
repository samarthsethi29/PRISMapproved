# step4c_paper_table.py
import json
import numpy as np, pandas as pd

tm = json.load(open("models/test_metrics.json"))["test_metrics"]
th = json.load(open("models/thresholds.json"))
rows = []
for t, m in tm.items():
    rows.append({"Target": t, "n_labeled": m["n_labeled"],
                 "AUC-ROC": round(m["AUC"], 3), "AUPRC": round(m["AUPRC"], 3),
                 "Pos. rate (=random AUPRC)": round(m["n_active"]/m["n_labeled"], 3),
                 "MCC": round(m["MCC"], 3), "Thr": th[t]})
df = pd.DataFrame(rows)
avg = {"Target": "Average", "n_labeled": "-",
       "AUC-ROC": round(df["AUC-ROC"].mean(), 3),
       "AUPRC": round(df["AUPRC"].mean(), 3),
       "Pos. rate (=random AUPRC)": round(df["Pos. rate (=random AUPRC)"].mean(), 3),
       "MCC": "-", "Thr": "-"}
df.loc[len(df)] = avg
df.to_csv("outputs/test_metrics_table.csv", index=False)
print(df.to_markdown(index=False))