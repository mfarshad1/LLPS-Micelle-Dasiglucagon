from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MSDDIR = ROOT / "reduced_data" / "msd"
MOBFILE = (
    ROOT / "reduced_data" / "mobility" /
    "mobility_4mer_-6_3replicas.txt"
)
OUT = ROOT / "reproduced_figures" / "figure9_msd_mobility.pdf"

charges = [8, 12, 16, 20, 24]

fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.3))

for q in charges:

    curves = []

    for rep in [1, 2, 3]:

        f = MSDDIR / f"q{q}_rep{rep}_msd_region_com.dat"

        if not f.exists():
            raise RuntimeError(
                f"Missing repository MSD data: {f}"
            )

        a = np.loadtxt(f, comments="#")

        if a.ndim == 1:
            a = a[None, :]

        # first column = time;
        # final numeric column = total MSD
        t = a[:, 0]
        msd = a[:, -1]

        curves.append((t, msd))

    tref = curves[0][0]

    yy = []
    for t, y in curves:
        yy.append(np.interp(tref, t, y))

    yy = np.asarray(yy)

    mean = yy.mean(axis=0)
    std = yy.std(axis=0, ddof=1)

    ax[0].plot(
        tref, mean,
        label=rf"$q_{{\mathrm{{M}}}}^*={q:.2f}$"
    )
    ax[0].fill_between(
        tref,
        mean - std,
        mean + std,
        alpha=0.15
    )

mob = np.loadtxt(MOBFILE)

ax[1].errorbar(
    mob[:, 0],
    mob[:, 1],
    yerr=mob[:, 2],
    marker="o",
    capsize=2
)

ax[0].set_xlabel(r"$t^*$ ($\tau^*$)")
ax[0].set_ylabel(
    r"$\langle \Delta r^{*2}\rangle$ ($\sigma^{*2}$)"
)
ax[0].legend(frameon=False, fontsize=8)

ax[1].set_xlabel(r"$q_{\mathrm{M}}^*$")
ax[1].set_ylabel(
    r"$\mu^*$ ($\sigma^{*2}/(\epsilon^*\tau^*)$)"
)

ax[0].text(
    0.04, 0.95, "(a)",
    transform=ax[0].transAxes,
    va="top"
)
ax[1].text(
    0.04, 0.95, "(b)",
    transform=ax[1].transAxes,
    va="top"
)

for a in ax:
    a.tick_params(direction="in", top=True, right=True)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")
