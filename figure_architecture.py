"""
figure_architecture.py — PRISM v3 architecture diagram (two-row, orthogonal
routing), matching the previous paper's Fig. 1, relabelled for
ABL1 / c-KIT / PDGFRβ. Pure schematic — no inputs required.
Text size is controlled by SHRINK (1.0 = original sizes; 0.75 = current).
Writes: outputs/paper/figs/figure_architecture_2row_orthogonal.png / .pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

SHRINK = 0.75   # ← global text scale: lower = smaller text

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "font.size": 8 * SHRINK})

EDGE, GRAY = "#1e293b", "#64748b"
FC = {"feat": "#fef3c7", "proj": "#fee2e2", "gin": "#dcfce7",
      "pool": "#ede9fe", "mlp": "#fef9c3", "head": "#ffedd5", "out": "#f1f5f9"}

fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.set_xlim(0, 35); ax.set_ylim(0, 22.5); ax.axis("off"); ax.set_aspect("equal")

def box(x, y, w, h, title, sub=None, fc="#f1f5f9", fs=8, subfs=6.2,
        bold=False, dashed=False):
    fs, subfs = fs * SHRINK, subfs * SHRINK          # ← scaled here
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.15",
                 fc=fc, ec=EDGE, lw=1.0, ls=("--" if dashed else "-"), zorder=3))
    if sub:
        ax.text(x, y + h*0.18, title, ha="center", va="center", fontsize=fs,
                zorder=5, fontweight=("bold" if bold else "normal"))
        ax.text(x, y - h*0.22, sub, ha="center", va="center", fontsize=subfs,
                color="#334155", zorder=5)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=fs, zorder=5,
                fontweight=("bold" if bold else "normal"))

def polyline(pts, lw=1.1, ls="-"):
    for a, b in zip(pts[:-1], pts[1:]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=EDGE, lw=lw,
                           ls=ls, zorder=4))

def arrow(pts, lw=1.1, ls="-"):
    polyline(pts, lw, ls)
    a, b = pts[-2], pts[-1]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                 color=EDGE, lw=lw, shrinkA=0, shrinkB=0, zorder=4))

# ── ROW 1: molecule → features → projections → GINE encoder ───────────────
ring = [(2.4, 17.9), (1.19, 17.2), (1.19, 15.8), (2.4, 15.1),
        (3.61, 15.8), (3.61, 17.2)]
subs = [((1.10, 19.4), ring[1]), ((3.70, 19.4), ring[5])]
for i, j in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)]:
    ax.add_line(Line2D([ring[i][0], ring[j][0]], [ring[i][1], ring[j][1]],
                       color=EDGE, lw=1.2, zorder=1))
for p, q in subs:
    ax.add_line(Line2D([p[0], q[0]], [p[1], q[1]], color=EDGE, lw=1.2, zorder=1))
for (px, py) in ring + [p for p, _ in subs]:
    ax.add_patch(Circle((px, py), 0.25, fc="#334155", ec="none", zorder=2))
ax.text(2.4, 13.9, "molecule (SMILES)", ha="center",
        fontsize=6 * SHRINK, color=GRAY)

box(6.5, 17.5, 3.6, 1.7, "atom feats", "[N × 7]",  FC["feat"])
box(6.5, 15.5, 3.6, 1.7, "bond feats", "[2E × 8]", FC["feat"])
box(11.0, 17.5, 3.2, 1.7, "Linear", "7 → 256", FC["proj"])
box(11.0, 15.5, 3.2, 1.7, "Linear", "8 → 256", FC["proj"])
box(17.2, 16.5, 6.4, 4.4, "GINEConv × 4",
    "256 → 512 → 256 MLP\nBN · ReLU · Dropout 0.2", FC["gin"],
    fs=9.5, subfs=6.4, bold=True)

for yy in (17.5, 15.5):
    arrow([(4.1, yy), (4.7, yy)])
    arrow([(8.3, yy), (9.4, yy)])
    arrow([(12.6, yy), (14.0, yy)])

# wrap: encoder output → pooling (row 2)
arrow([(20.4, 16.5), (21.8, 16.5), (21.8, 12.6), (1.5, 12.6),
       (1.5, 8.9), (3.0, 8.9)])
arrow([(1.5, 12.6), (1.5, 7.1), (3.0, 7.1)])
ax.text(11.5, 13.15, "node embeddings [N × 256] after 4 message-passing rounds",
        ha="center", fontsize=6 * SHRINK, color=GRAY)

# ── ROW 2: pooling → shared MLP → task heads → outputs ────────────────────
box(5.2, 8.9, 4.4, 1.7, "mean pool", "[256]", FC["pool"])
box(5.2, 7.1, 4.4, 1.7, "max pool",  "[256]", FC["pool"])
box(10.0, 8.0, 2.6, 3.6, "concat", "[512]", FC["pool"])
box(14.2, 8.0, 4.2, 2.6, "shared MLP", "512 → 256\nReLU · Dropout 0.2", FC["mlp"])

arrow([(7.4, 8.9), (8.7, 8.9)])
arrow([(7.4, 7.1), (8.7, 7.1)])
arrow([(11.3, 8.0), (12.1, 8.0)])

polyline([(16.3, 8.0), (17.4, 8.0)])
polyline([(17.4, 4.6), (17.4, 10.6)])
for yy, name in [(10.6, "ABL1 head"), (7.6, "c-KIT head"), (4.6, "PDGFRβ head")]:
    arrow([(17.4, yy), (17.8, yy)])
    box(19.6, yy, 3.6, 1.6, name, "Linear 256 → 1", FC["head"], fs=8, subfs=6)
    arrow([(21.4, yy), (24.7, yy)])
    ax.text(23.0, yy + 0.55, r"$\sigma$", ha="center", fontsize=7.5 * SHRINK)

box(26.4, 10.6, 3.4, 1.6, r"$P_{\mathrm{ABL1}}$", None, FC["out"], fs=9)
box(26.4, 7.6,  3.4, 1.6, r"$P_{\mathrm{cKIT}}$",  None, FC["out"], fs=9)
box(26.4, 4.6,  3.4, 1.6, r"$P_{\mathrm{PDGFR\beta}}$", None, FC["out"], fs=9)

# optional: downstream Polypharmacology Score (delete this block for an
# architecture-only figure identical in scope to the previous paper's Fig. 1)
polyline([(28.1, 10.6), (29.5, 10.6)], ls="--")
polyline([(28.1, 7.6),  (29.5, 7.6)],  ls="--")
polyline([(28.1, 4.6),  (29.5, 4.6)],  ls="--")
polyline([(29.5, 4.6), (29.5, 10.6)], ls="--")
arrow([(29.5, 7.6), (30.5, 7.6)], ls="--")
box(32.6, 7.6, 4.2, 2.6, "PS", r"$(P_1 P_2 P_3)^{1/3}$" + "\ngeometric mean",
    "white", fs=9.5, subfs=6, bold=True, dashed=True)

ax.text(17.5, 1.0, "1,453,315 trainable parameters · shared graph encoder, "
        "three task-specific heads · sigmoid applied at inference",
        ha="center", fontsize=6 * SHRINK, color=GRAY)

os.makedirs("outputs/paper/figs", exist_ok=True)
fig.savefig("outputs/paper/figs/figure_architecture_2row_orthogonal.png",
            dpi=300, bbox_inches="tight")
fig.savefig("outputs/paper/figs/figure_architecture_2row_orthogonal.pdf",
            bbox_inches="tight")
plt.close(fig)
print("wrote outputs/paper/figs/figure_architecture_2row_orthogonal.png/.pdf")