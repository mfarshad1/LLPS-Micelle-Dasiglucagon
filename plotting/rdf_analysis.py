from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator


ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT /
    "reduced_data" /
    "rdf" /
    "figure8_rdf_q8_16_24_32.csv"
)

OUT = (
    ROOT /
    "reproduced_figures" /
    "figure8_rdf.pdf"
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


charges = [8, 16, 24, 32]

df = pd.read_csv(DATA)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(8.0, 8.0),
    sharex=True,
    sharey=True,
)


mapping = {
    8: (0, 0),
    16: (0, 1),
    24: (1, 0),
    32: (1, 1),
}


for q in charges:

    d = (
        df[df["Charge"] == q]
        .sort_values("r")
        .copy()
    )

    if d.empty:
        raise RuntimeError(
            f"Repository reduced RDF data missing q={q}"
        )

    ax = axes[
        mapping[q]
    ]

    x = d["r"].to_numpy()

    r11 = d["rdf_1_1_mean"].to_numpy()
    s11 = d["rdf_1_1_std"].to_numpy()

    r12 = d["rdf_1_2_mean"].to_numpy()
    s12 = d["rdf_1_2_std"].to_numpy()

    r22 = d["rdf_2_2_mean"].to_numpy()
    s22 = d["rdf_2_2_std"].to_numpy()

    nrep = int(
        d["n_replicas"].iloc[0]
    )

    sl = slice(0, 100)

    ax.plot(
        x[sl],
        r11[sl],
        "-",
        color="blue",
        lw=2.5,
        label="M--M",
    )

    ax.plot(
        x[sl],
        r22[sl],
        "-",
        color="cyan",
        lw=2.5,
        label="DG--DG",
    )

    ax.plot(
        x[sl],
        r12[sl],
        "-",
        color="red",
        lw=2.5,
        label="M--DG",
    )

    if nrep > 1:

        ax.fill_between(
            x[sl],
            r11[sl] - s11[sl],
            r11[sl] + s11[sl],
            color="blue",
            alpha=0.18,
            linewidth=0,
        )

        ax.fill_between(
            x[sl],
            r22[sl] - s22[sl],
            r22[sl] + s22[sl],
            color="cyan",
            alpha=0.18,
            linewidth=0,
        )

        ax.fill_between(
            x[sl],
            r12[sl] - s12[sl],
            r12[sl] + s12[sl],
            color="red",
            alpha=0.18,
            linewidth=0,
        )

    ax.text(
        0.05,
        0.95,
        rf"$q^*_{{\mathrm{{M}}}} = {q:.0f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24,
    )

    ax.set_ylim(-1, 12)
    ax.set_xlim(-1, 25)

    ax.set_xticks(
        np.arange(0, 21, 10)
    )

    ax.set_yticks(
        np.arange(0, 21, 10)
    )

    if q == 16:
        ax.legend(
            frameon=False,
            borderpad=0.1,
            labelspacing=0.2,
            columnspacing=0.2,
            borderaxespad=0.4,
            handletextpad=0.4,
            fontsize=22,
            loc="upper right",
            handlelength=0.7,
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
    )

    ax.tick_params(
        which="minor",
        length=3,
    )


# Exact manuscript layout
RDF_LEFT = 0.16
RDF_RIGHT = 0.98
RDF_BOTTOM = 0.16
RDF_TOP = 0.98

fig.subplots_adjust(
    left=RDF_LEFT,
    right=RDF_RIGHT,
    bottom=RDF_BOTTOM,
    top=RDF_TOP,
    hspace=0.10,
    wspace=0.10,
)

xlab = fig.supxlabel(
    r"$\mathbf{\textit{r}^{*}\ (\sigma^{*})}$",
    fontsize=40,
)

xlab.set_position(
    (0.57, 0.02)
)

ylab = fig.supylabel(
    r"RDF",
    fontsize=40,
)

RDF_YCENTER = (
    0.5 *
    (RDF_BOTTOM + RDF_TOP)
)

ylab.set_position(
    (0.03, RDF_YCENTER)
)


fig.savefig(
    OUT,
    bbox_inches="tight",
)

print(f"Saved {OUT}")
