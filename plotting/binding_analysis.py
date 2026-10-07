from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILE = (
    ROOT / "reduced_data" / "binding" /
    "dg_bound_unbound_analysis_4mer_-6_fixed_regions_3replicas.txt"
)
OUT = ROOT / "reproduced_figures" / "figure7_binding_analysis.pdf"

x = np.loadtxt(FILE)

q = x[:, 0]

# Columns from repository header
bc, bc_sd = x[:, 2], x[:, 3]
uc, uc_sd = x[:, 4], x[:, 5]
bo, bo_sd = x[:, 6], x[:, 7]
uo, uo_sd = x[:, 8], x[:, 9]

fig, ax = plt.subplots(figsize=(5.2, 4.0))

ax.errorbar(q, bc, yerr=bc_sd,
            marker="o", capsize=2,
            label="Bound, central")

ax.errorbar(q, uc, yerr=uc_sd,
            marker="s", capsize=2,
            label="Unbound, central")

ax.errorbar(q, bo, yerr=bo_sd,
            marker="^", capsize=2,
            label="Bound, outer")

ax.errorbar(q, uo, yerr=uo_sd,
            marker="D", capsize=2,
            label="Unbound, outer")

ax.set_xlabel(r"$q_{\mathrm{M}}^*$")
ax.set_ylabel("DG count")
ax.legend(frameon=False)

ax.tick_params(direction="in", top=True, right=True)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")
