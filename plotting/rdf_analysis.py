from pathlib import Path
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

# ============================================================
# STYLE — original manuscript settings
# ============================================================
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{bm}')
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

rc('axes', labelsize='28')
rc('xtick', labelsize='24')
rc('ytick', labelsize='24')

REPO = Path(__file__).resolve().parents[1]

data_file = (
    REPO
    / "reduced_data"
    / "rdf"
    / "figure8_rdf_q8_16_24_32.csv"
)

output_pdf = (
    REPO
    / "reproduced_figures"
    / "figure8_rdf.pdf"
)
output_pdf.parent.mkdir(parents=True, exist_ok=True)

data = np.genfromtxt(
    data_file,
    delimiter=",",
    names=True,
    dtype=None,
    encoding=None
)

charges_to_plot = [8, 16, 24, 32]

rdf_map = {
    '8.00': (0, 0),
    '16.00': (0, 1),
    '24.00': (1, 0),
    '32.00': (1, 1)
}

# Exact manuscript RDF figure geometry
figR, rdf_axes_grid = plt.subplots(
    2, 2,
    figsize=(8.0, 8.0),
    sharex=True,
    sharey=True
)

for q in charges_to_plot:
    q_formatted = "{:.2f}".format(q)

    mask = data["Charge"] == q
    rows = data[mask]

    if len(rows) == 0:
        print(f"Skipping q = {q_formatted}: no repository RDF data")
        continue

    xvals = np.asarray(rows["r"], dtype=float)

    ave_rdf_1_1 = np.asarray(rows["rdf_1_1_mean"], dtype=float)
    std_rdf_1_1 = np.asarray(rows["rdf_1_1_std"], dtype=float)

    ave_rdf_1_2 = np.asarray(rows["rdf_1_2_mean"], dtype=float)
    std_rdf_1_2 = np.asarray(rows["rdf_1_2_std"], dtype=float)

    ave_rdf_2_2 = np.asarray(rows["rdf_2_2_mean"], dtype=float)
    std_rdf_2_2 = np.asarray(rows["rdf_2_2_std"], dtype=float)

    r_i, r_j = rdf_map[q_formatted]
    axR = rdf_axes_grid[r_i, r_j]

    # ---- exact manuscript plotting order ----
    axR.plot(
        xvals[0:100],
        ave_rdf_1_1[0:100],
        '-',
        color='blue',
        lw=2.5,
        label='M--M'
    )

    axR.plot(
        xvals[0:100],
        ave_rdf_2_2[0:100],
        '-',
        color='cyan',
        lw=2.5,
        label='DG--DG'
    )

    axR.plot(
        xvals[0:100],
        ave_rdf_1_2[0:100],
        '-',
        color='red',
        lw=2.5,
        label='M--DG'
    )

    axR.fill_between(
        xvals[0:100],
        ave_rdf_1_1[0:100] - std_rdf_1_1[0:100],
        ave_rdf_1_1[0:100] + std_rdf_1_1[0:100],
        color='blue',
        alpha=0.18,
        linewidth=0
    )

    axR.fill_between(
        xvals[0:100],
        ave_rdf_2_2[0:100] - std_rdf_2_2[0:100],
        ave_rdf_2_2[0:100] + std_rdf_2_2[0:100],
        color='cyan',
        alpha=0.18,
        linewidth=0
    )

    axR.fill_between(
        xvals[0:100],
        ave_rdf_1_2[0:100] - std_rdf_1_2[0:100],
        ave_rdf_1_2[0:100] + std_rdf_1_2[0:100],
        color='red',
        alpha=0.18,
        linewidth=0
    )

    axR.text(
        0.05,
        0.95,
        rf'$q^*_{{\mathrm{{M}}}} = {q:.0f}$',
        transform=axR.transAxes,
        ha='left',
        va='top',
        fontsize=24
    )

    axR.set_ylim(-1, 12)
    axR.set_xlim(-1, 25)

    axR.set_xticks(np.arange(0, 21, 10))
    axR.set_yticks(np.arange(0, 21, 10))

    if q_formatted == '16.00':
        axR.legend(
            frameon=False,
            borderpad=0.1,
            labelspacing=0.2,
            columnspacing=0.2,
            borderaxespad=0.4,
            handletextpad=0.4,
            fontsize='22',
            loc='upper right',
            handlelength=0.7
        )

    axR.xaxis.set_minor_locator(AutoMinorLocator())
    axR.yaxis.set_minor_locator(AutoMinorLocator())

    axR.tick_params(which='major', length=5, pad=4)
    axR.tick_params(which='minor', length=3)

# -------------------- ORIGINAL LAYOUT --------------------
RDF_LEFT = 0.16
RDF_RIGHT = 0.98
RDF_BOTTOM = 0.16
RDF_TOP = 0.98

figR.subplots_adjust(
    left=RDF_LEFT,
    right=RDF_RIGHT,
    bottom=RDF_BOTTOM,
    top=RDF_TOP,
    hspace=0.10,
    wspace=0.10
)

RDF_YCENTER = 0.5 * (RDF_BOTTOM + RDF_TOP)

xlab = figR.supxlabel(
    r'$\mathbf{\textit{r}^{*}\ (\sigma^{*})}$',
    fontsize=40
)
xlab.set_position((0.57, 0.02))

ylab = figR.supylabel(
    r'RDF',
    fontsize=40
)

ylab.set_position((-0.02, 0.5))
ylab.set_position((0.03, RDF_YCENTER))

figR.savefig(
    output_pdf,
    bbox_inches='tight'
)

print("Saved RDF figure to:", output_pdf)

plt.show()
