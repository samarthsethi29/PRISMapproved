# step4b_plot_curves.py  — regenerates the paper's training-curve figure
import json, os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)
log = json.load(open("models/train_log.json"))
df = pd.DataFrame(log)
df.to_csv("outputs/training_log.csv", index=False)   # old-repo parity

RANDOM_AUPRC = 0.732   # mean val positive rate (Step 3: 187/261, 181/239, 154/213)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(df.epoch, df.train_loss, color="tab:green")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("masked multi-task loss (per batch)")
ax[0].set_title("Training loss")

ax[1].plot(df.epoch, df.val_avg_auprc, color="tab:blue")
ax[1].axhline(RANDOM_AUPRC, ls="--", color="grey",
              label=f"random baseline ({RANDOM_AUPRC})")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("avg val AUPRC")
ax[1].set_ylim(0.5, 1.0); ax[1].set_title("Validation AUPRC")
ax[1].legend()
fig.tight_layout()
fig.savefig("outputs/training_curves.png", dpi=300)
print("wrote outputs/training_curves.png and outputs/training_log.csv")