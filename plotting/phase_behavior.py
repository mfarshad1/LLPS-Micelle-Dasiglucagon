from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reduced_data" / "phase_behavior"
OUT = ROOT / "reproduced_figures" / "figure5_phase_behavior.pdf"

mean = pd.read_csv(
    DATA / "phase_data_4mer_-6_direct_L6_3replicas_mean.csv"
)
std = pd.read_csv(
    DATA / "phase_data_4mer_-6_direct_L6_3replicas_std.csv"
)

q = mean["Charge"].to_numpy()

fig, ax = plt.subplots(1, 2, figsize=(6.8, 4.1), sharey=True)

# Central = coacervate; outer = supernatant
ax[0].errorbar(
    mean["Micelle phi (supernatant)"], q,
    xerr=std["Micelle phi (supernatant)"],
    marker="o", linestyle="--", capsize=2,
    label="Outer"
)
ax[0].errorbar(
    mean["Micelle phi (coacervate)"], q,
    xerr=std["Micelle phi (coacervate)"],
    marker="s", linestyle="--", capsize=2,
    label="Central"
)

ax[1].errorbar(
    mean["DG phi (supernatant)"], q,
    xerr=std["DG phi (supernatant)"],
    marker="o", linestyle="--", capsize=2,
    label="Outer"
)
ax[1].errorbar(
    mean["DG phi (coacervate)"], q,
    xerr=std["DG phi (coacervate)"],
    marker="s", linestyle="--", capsize=2,
    label="Central"
)

ax[0].set_xlabel(r"$\phi_{\mathrm{M}}$")
ax[1].set_xlabel(r"$\phi_{\mathrm{DG}}$")
ax[0].set_ylabel(r"$q_{\mathrm{M}}^*$")

ax[0].legend(frameon=False)

for a in ax:
    a.tick_params(direction="in", top=True, right=True)

# q* ~ 2.03 zeta from the charge mapping
sec = ax[1].secondary_yaxis(
    "right",
    functions=(lambda q: q / 2.03,
               lambda zeta: zeta * 2.03)
)
sec.set_ylabel(r"$\zeta$ (mV)")

ax[0].text(0.04, 0.95, "(a)",
           transform=ax[0].transAxes, va="top")
ax[1].text(0.04, 0.95, "(b)",
           transform=ax[1].transAxes, va="top")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")
