from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT /
    "reduced_data" /
    "density_profiles" /
    "figure6_dg_density_q8_16_24_32.csv"
)

OUT = (
    ROOT /
    "reproduced_figures" /
    "figure6b_density_profiles.pdf"
)


# ============================================================
# Original manuscript style
# ============================================================

rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{bm}")
rc("ps", usedistiller="xpdf")
rc(
    "font",
    **{
        "family": "serif",
        "serif": ["Computer Modern Roman"],
    }
)

rc("axes", labelsize="28")
rc("xtick", labelsize="24")
rc("ytick", labelsize="24")


def rho_model(x, rp, rm, xmx, xmn, L):
    f1 = (x - xmn) / L
    f2 = (xmx - x) / L

    return (
        rm
        + 0.5
        * (rp - rm)
        * (np.tanh(f1) + np.tanh(f2))
    )


charges = [8, 16, 24, 32]

df = pd.read_csv(DATA)

fig, axes = plt.subplots(
    4,
    1,
    figsize=(7.0, 18.0),
)

DENS_TICK_FONTSIZE = 48
DENS_LEGEND_FONTSIZE = 48


for ax, q in zip(axes, charges):

    d = (
        df[df["Charge"] == q]
        .sort_values("x")
        .copy()
    )

    if d.empty:
        raise RuntimeError(
            f"Repository reduced density data missing q={q}"
        )

    xvals = d["x"].to_numpy()
    ave_dens = d["mean_density"].to_numpy()
    std_dens = d["std_density"].to_numpy()

    x_fit = np.linspace(
        np.min(xvals),
        np.max(xvals),
        2400
    )

    sdev = np.std(ave_dens)

    if sdev == 0:
        sdev = 1e-8

    sigma = np.full_like(
        ave_dens,
        sdev
    )

    popt, _ = curve_fit(
        rho_model,
        xvals,
        ave_dens,
        p0=[
            max(ave_dens),
            min(ave_dens),
            max(xvals),
            min(xvals),
            20,
        ],
        sigma=sigma,
        maxfev=10000,
    )

    y_fit = rho_model(
        x_fit,
        *popt
    )

    ax.plot(
        xvals[::20],
        ave_dens[::20],
        "o",
        color="cyan",
        markersize=8,
        fillstyle="none",
    )

    ax.plot(
        x_fit,
        y_fit,
        "--",
        color="cyan",
        label=rf"$q^*_{{\mathrm{{M}}}} = \mathrm{{{q}}}$",
        lw=3,
    )

    ax.fill_between(
        xvals,
        ave_dens - std_dens,
        ave_dens + std_dens,
        color="cyan",
        alpha=0.5,
    )

    ax.set_xlim(0, 300)
    ax.set_xticks(
        np.arange(0, 301, 100)
    )

    ax.set_yticks(
        np.arange(0, 0.061, 0.02)
    )

    ax.set_ylim(-0.004, 0.06)

    ax.set_ylabel(
        r"$\mathbf{{\rho}^{*}_{DG}}$",
        fontsize=70,
        labelpad=15,
    )

    ax.set_xlabel(
        r"$\mathbf{\textit{z}^{*}}$",
        fontsize=70,
        labelpad=0,
    )

    ax.legend(
        frameon=False,
        borderpad=0.1,
        labelspacing=0.2,
        columnspacing=0.2,
        borderaxespad=0.4,
        handletextpad=0.4,
        fontsize=DENS_LEGEND_FONTSIZE,
        loc="upper left",
        handlelength=0.5,
    )

    ax.xaxis.set_minor_locator(
        AutoMinorLocator()
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator()
    )

    ax.tick_params(
        which="major",
        length=5,
        pad=4,
        labelsize=DENS_TICK_FONTSIZE,
    )

    ax.tick_params(
        which="minor",
        length=3,
    )


fig.tight_layout(
    h_pad=0.8
)

fig.savefig(
    OUT,
    bbox_inches="tight",
)

print(f"Saved {OUT}")
