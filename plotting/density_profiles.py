from pathlib import Path
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit

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

def rho_model(x, rp, rm, xmx, xmn, L):
    f1 = (x - xmn) / L
    f2 = (xmx - x) / L
    return rm + 0.5 * (rp - rm) * ((np.tanh(f1) + np.tanh(f2)))

REPO = Path(__file__).resolve().parents[1]

data_file = (
    REPO
    / "reduced_data"
    / "density_profiles"
    / "figure6_dg_density_q8_16_24_32.csv"
)

output_pdf = (
    REPO
    / "reproduced_figures"
    / "figure6b_density_profiles.pdf"
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

dens_map = {
    '8.00': 0,
    '16.00': 1,
    '24.00': 2,
    '32.00': 3
}

# -------------------- FIGURE --------------------
# Exact manuscript figure geometry
figD, dens_axes = plt.subplots(4, 1, figsize=(7.0, 18.0))

# Density-only font changes requested in the manuscript code
DENS_TICK_FONTSIZE = 48
DENS_LEGEND_FONTSIZE = 48

for q in charges_to_plot:
    q_formatted = "{:.2f}".format(q)

    mask = data["Charge"] == q
    rows = data[mask]

    if len(rows) == 0:
        print(f"Skipping q = {q_formatted}: no repository density data")
        continue

    xvals = np.asarray(rows["x"], dtype=float)
    ave_dens = np.asarray(rows["mean_density"], dtype=float)
    standard_deviations = np.asarray(rows["std_density"], dtype=float)

    # Same fit used by manuscript analysis
    sdev = np.std(ave_dens)
    if sdev == 0:
        sdev = 1e-8

    sigma = np.full_like(ave_dens, sdev)

    try:
        popt, _ = curve_fit(
            rho_model,
            xvals,
            ave_dens,
            p0=[
                max(ave_dens),
                min(ave_dens),
                max(xvals),
                min(xvals),
                20
            ],
            sigma=sigma,
            maxfev=10000
        )

        x_fit = np.linspace(min(xvals), max(xvals), 2400)
        y_fit = rho_model(x_fit, *popt)

    except Exception as exc:
        print(f"Fit failed for q = {q_formatted}: {exc}")
        x_fit = xvals
        y_fit = np.full_like(xvals, np.nan)

    # ---- original manuscript plotting block ----
    axD = dens_axes[dens_map[q_formatted]]

    axD.plot(
        xvals[::20],
        ave_dens[::20],
        'o',
        color='cyan',
        markersize=8,
        fillstyle='none'
    )

    axD.plot(
        x_fit,
        y_fit,
        '--',
        color='cyan',
        label=rf'$q^*_{{\mathrm{{M}}}} = \mathrm{{{q}}}$',
        lw=3
    )

    axD.fill_between(
        xvals,
        ave_dens - standard_deviations,
        ave_dens + standard_deviations,
        color='cyan',
        alpha=0.5
    )

    axD.set_xlim(0, 300)
    axD.set_xticks(np.arange(0, 301, 100))

    axD.set_yticks(np.arange(0, 0.061, 0.02))
    axD.set_ylim(-.004, 0.06)

    axD.set_ylabel(
        r'$\mathbf{{\rho}^{*}_{DG}}$',
        fontsize=70,
        labelpad=15
    )

    axD.set_xlabel(
        r'$\mathbf{\textit{z}^{*}}$',
        fontsize=70,
        labelpad=0
    )

    axD.legend(
        frameon=False,
        borderpad=0.1,
        labelspacing=0.2,
        columnspacing=0.2,
        borderaxespad=0.4,
        handletextpad=0.4,
        fontsize=DENS_LEGEND_FONTSIZE,
        loc='upper left',
        handlelength=0.5
    )

    axD.xaxis.set_minor_locator(AutoMinorLocator())
    axD.yaxis.set_minor_locator(AutoMinorLocator())

    axD.tick_params(
        which='major',
        length=5,
        pad=4,
        labelsize=DENS_TICK_FONTSIZE
    )
    axD.tick_params(which='minor', length=3)

figD.tight_layout(h_pad=0.8)

figD.savefig(
    output_pdf,
    bbox_inches='tight'
)

print("Saved density figure to:", output_pdf)

plt.show()
