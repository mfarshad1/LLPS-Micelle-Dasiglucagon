import os
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
import seaborn as sns
from scipy.signal import find_peaks

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

# ============================================================
# STYLE
# ============================================================
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{bm}')
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

# Match the phase/volume-fraction figure style
rc('axes', labelsize='28')
rc('xtick', labelsize='24')
rc('ytick', labelsize='24')


def rho_model(x, rp, rm, xmx, xmn, L):
    f1 = (x - xmn) / L
    f2 = (xmx - x) / L
    return rm + 0.5 * (rp - rm) * ((np.tanh(f1) + np.tanh(f2)))


# ============================================================
# USER SETTINGS
# ============================================================

number = '4mer_-6'

base_dir = "/groups/jwhitme1/Data-MF-01/LLPS/llps-data/biphasic/dg-1-7_2/4mer_-6"
output_dir = "/groups/jwhitme1/Data-MF-01/LLPS/"

replicas = [1, 2, 3]

# Only calculate and plot these charges.
# This skips all in-between charges.
charges_to_plot = [8, 16, 24, 32]

number_bins = 1000
colors = sns.color_palette("bright", 20)

dens_map = {'8.00': 0, '16.00': 1, '24.00': 2, '32.00': 3}
rdf_map  = {'8.00': (0, 0), '16.00': (0, 1), '24.00': (1, 0), '32.00': (1, 1)}

density_pdf = os.path.join(output_dir, 'density.pdf')
rdf_pdf = os.path.join(output_dir, 'rdf.pdf')


# -------------------- FIGURES --------------------
# Density figure enlarged only for readability.
figD, dens_axes = plt.subplots(4, 1, figsize=(7.0, 18.0))

figR, rdf_axes_grid = plt.subplots(
    2, 2, figsize=(8.0, 8.0),
    sharex=True, sharey=True
)

# Density-only font changes requested by Jon.
# Axis labels are kept unchanged below.
DENS_TICK_FONTSIZE = 48
DENS_LEGEND_FONTSIZE = 48


# ============================================================
# DENSITY HELPERS
# ============================================================

def read_density_blocks(density_file):
    all_density = []

    with open(density_file, "r") as file:
        for line in file:
            s = line.strip()

            if not s:
                continue

            if s.startswith("#"):
                continue

            try:
                values = [float(v) for v in s.split()]
            except ValueError:
                continue

            if len(values) < 4:
                continue

            if values[0] > number_bins:
                continue

            all_density.append(values)

    block_size = number_bins
    num_blocks = len(all_density) // block_size

    blocks = []

    for i in range(num_blocks):
        if i > 50:
            start = i * block_size
            end = start + block_size
            block = all_density[start:end]

            if len(block) == number_bins:
                blocks.append(block)

    return blocks


def process_density_one_replica(q_formatted, rep):
    density = os.path.join(
        base_dir,
        f"q-{q_formatted}",
        f"rep{rep}",
        f"density_polymer.q-{q_formatted}.data"
    )

    if not os.path.exists(density):
        print(f"Skipping q = {q_formatted}, rep = {rep}: density file not found")
        return None

    blocks = read_density_blocks(density)

    if not blocks:
        print(f"Skipping q = {q_formatted}, rep = {rep}: no density blocks")
        return None

    ave_block = np.zeros((number_bins, 4))
    low_dens, high_dens = [], []

    for block in blocks:
        dens = np.array([row[3] for row in block])
        xvals = np.array([row[1] for row in block])

        sdev = np.std(dens)

        if sdev == 0:
            sdev = 1e-8

        sigma = np.full_like(dens, sdev)

        try:
            popt, _ = curve_fit(
                rho_model, xvals, dens,
                p0=[max(dens), min(dens), max(xvals), min(xvals), 20],
                sigma=sigma,
                maxfev=10000
            )

            low_dens.append(abs(popt[1]))
            high_dens.append(abs(popt[0]))

        except Exception as e:
            print(f"Fit failed for q = {q_formatted}, rep = {rep}: {e}")

        ave_block += np.array(block)

    ave_block /= len(blocks)

    ave_dens = np.array([row[3] for row in ave_block])
    xvals = np.array([row[1] for row in ave_block])

    xvals_shifted = xvals - np.min(xvals)

    return {
        "xvals": xvals_shifted,
        "ave_dens": ave_dens,
        "low_dens": low_dens,
        "high_dens": high_dens,
        "n_blocks": len(blocks),
        "file": density
    }


def process_density_three_replicas(q_formatted):
    rep_results = []

    for rep in replicas:
        result = process_density_one_replica(q_formatted, rep)

        if result is not None:
            print(
                f"q = {q_formatted}, rep = {rep}: "
                f"density blocks used = {result['n_blocks']}"
            )
            rep_results.append(result)

    if len(rep_results) == 0:
        return None

    x_ref = rep_results[0]["xvals"]

    dens_reps = []

    for result in rep_results:
        x = result["xvals"]
        dens = result["ave_dens"]

        if len(x) != len(x_ref) or not np.allclose(x, x_ref):
            dens_reps.append(np.interp(x_ref, x, dens))
        else:
            dens_reps.append(dens)

    dens_reps = np.array(dens_reps)

    ave_dens = np.mean(dens_reps, axis=0)
    std_dens = np.std(dens_reps, axis=0)

    x_fit = np.linspace(min(x_ref), max(x_ref), 2400)

    try:
        sdev = np.std(ave_dens)

        if sdev == 0:
            sdev = 1e-8

        sigma = np.full_like(ave_dens, sdev)

        popt, _ = curve_fit(
            rho_model,
            x_ref,
            ave_dens,
            p0=[max(ave_dens), min(ave_dens), max(x_ref), min(x_ref), 20],
            sigma=sigma,
            maxfev=10000
        )

        y_fit = rho_model(x_fit, *popt)

    except Exception as e:
        print(f"Mean density fit failed for q = {q_formatted}: {e}")
        y_fit = np.full_like(x_fit, np.nan)

    low_all = []
    high_all = []

    for result in rep_results:
        low_all.extend(result["low_dens"])
        high_all.extend(result["high_dens"])

    return {
        "xvals": x_ref,
        "ave_dens": ave_dens,
        "std_dens": std_dens,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "low_dens": low_all,
        "high_dens": high_all,
        "n_replicas": len(rep_results)
    }


# ============================================================
# RDF HELPERS
# ============================================================

def read_rdf_blocks(rdf_file):
    all_rdf = []

    with open(rdf_file, "r") as file:
        for line in file:
            s = line.strip()

            if not s:
                continue

            if s.startswith("#"):
                continue

            try:
                values = [float(v) for v in s.split()]
            except ValueError:
                continue

            if len(values) < 9:
                continue

            if values[0] > number_bins:
                continue

            all_rdf.append(values)

    block_size = number_bins
    num_blocks = len(all_rdf) // block_size

    blocks = []

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block = all_rdf[start:end]

        if len(block) == number_bins:
            blocks.append(block)

    return blocks


def process_rdf_one_replica(q_formatted, rep):
    rdf_file = os.path.join(
        base_dir,
        f"q-{q_formatted}",
        f"rep{rep}",
        f"out.q-{q_formatted}.rdf"
    )

    if not os.path.exists(rdf_file):
        print(f"Skipping q = {q_formatted}, rep = {rep}: RDF file not found")
        return None

    blocks = read_rdf_blocks(rdf_file)

    if not blocks:
        print(f"Skipping q = {q_formatted}, rep = {rep}: no RDF blocks")
        return None

    ave_block = np.zeros((number_bins, 10))

    for block in blocks:
        ave_block += np.array(block)

    ave_block /= len(blocks)

    ave_rdf_1_1 = np.array([row[2] for row in ave_block])
    ave_rdf_1_2 = np.array([row[4] for row in ave_block])
    ave_rdf_2_2 = np.array([row[8] for row in ave_block])
    xvals = np.array([row[1] for row in ave_block])

    return {
        "xvals": xvals,
        "ave_rdf_1_1": ave_rdf_1_1,
        "ave_rdf_1_2": ave_rdf_1_2,
        "ave_rdf_2_2": ave_rdf_2_2,
        "n_blocks": len(blocks),
        "file": rdf_file
    }


def process_rdf_three_replicas(q_formatted):
    rep_results = []

    for rep in replicas:
        result = process_rdf_one_replica(q_formatted, rep)

        if result is not None:
            print(
                f"q = {q_formatted}, rep = {rep}: "
                f"RDF blocks used = {result['n_blocks']}"
            )
            rep_results.append(result)

    if len(rep_results) == 0:
        return None

    x_ref = rep_results[0]["xvals"]

    rdf_1_1_reps = []
    rdf_1_2_reps = []
    rdf_2_2_reps = []

    for result in rep_results:
        x = result["xvals"]

        if len(x) != len(x_ref) or not np.allclose(x, x_ref):
            rdf_1_1_reps.append(np.interp(x_ref, x, result["ave_rdf_1_1"]))
            rdf_1_2_reps.append(np.interp(x_ref, x, result["ave_rdf_1_2"]))
            rdf_2_2_reps.append(np.interp(x_ref, x, result["ave_rdf_2_2"]))
        else:
            rdf_1_1_reps.append(result["ave_rdf_1_1"])
            rdf_1_2_reps.append(result["ave_rdf_1_2"])
            rdf_2_2_reps.append(result["ave_rdf_2_2"])

    rdf_1_1_reps = np.array(rdf_1_1_reps)
    rdf_1_2_reps = np.array(rdf_1_2_reps)
    rdf_2_2_reps = np.array(rdf_2_2_reps)

    return {
        "xvals": x_ref,

        "ave_rdf_1_1": np.mean(rdf_1_1_reps, axis=0),
        "ave_rdf_1_2": np.mean(rdf_1_2_reps, axis=0),
        "ave_rdf_2_2": np.mean(rdf_2_2_reps, axis=0),

        "std_rdf_1_1": np.std(rdf_1_1_reps, axis=0),
        "std_rdf_1_2": np.std(rdf_1_2_reps, axis=0),
        "std_rdf_2_2": np.std(rdf_2_2_reps, axis=0),

        "n_replicas": len(rep_results)
    }


# -------------------- DENSITY PASS --------------------
ave_low_dens, ave_high_dens = [], []

for q in charges_to_plot:
    q_formatted = "{:.2f}".format(q)

    print("\n" + "=" * 70)
    print(f"Processing density q = {q_formatted}")
    print("=" * 70)

    result = process_density_three_replicas(q_formatted)

    if result is None:
        print(f"Skipping q = {q_formatted}: no replica density data")
        continue

    xvals = result["xvals"]
    ave_dens = result["ave_dens"]
    standard_deviations = result["std_dens"]
    x_fit = result["x_fit"]
    y_fit = result["y_fit"]

    # ---- plot density ----
    if q_formatted in dens_map:
        axD = dens_axes[dens_map[q_formatted]]

        axD.plot(
            xvals[::20], ave_dens[::20],
            'o', color='cyan', markersize=8, fillstyle='none'
        )

        axD.plot(
            x_fit, y_fit, '--', color='cyan',
            label=rf'$q^*_{{\mathrm{{M}}}} = \mathrm{{{q}}}$',
            lw=3
        )

        axD.fill_between(
            xvals,
            ave_dens - standard_deviations,
            ave_dens + standard_deviations,
            color='cyan', alpha=0.5
        )

        axD.set_xlim(0, 300)
        axD.set_xticks(np.arange(0, 301, 100))

        axD.set_yticks(np.arange(0, 0.061, 0.02))
        axD.set_ylim(-.004, 0.06)

        # Axis labels unchanged.
        axD.set_ylabel(r'$\mathbf{{\rho}^{*}_{DG}}$', fontsize=70, labelpad=15)
        axD.set_xlabel(r'$\mathbf{\textit{z}^{*}}$', fontsize=70, labelpad=0)

        # Density legend enlarged only here.
        axD.legend(
            frameon=False, borderpad=0.1, labelspacing=0.2,
            columnspacing=0.2, borderaxespad=0.4,
            handletextpad=0.4, fontsize=DENS_LEGEND_FONTSIZE,
            loc='upper left', handlelength=0.5
        )

        axD.xaxis.set_minor_locator(AutoMinorLocator())
        axD.yaxis.set_minor_locator(AutoMinorLocator())

        # Density tick/data font enlarged only here.
        axD.tick_params(which='major', length=5, pad=4,
                        labelsize=DENS_TICK_FONTSIZE)
        axD.tick_params(which='minor', length=3)

    high_dens = result["high_dens"]
    low_dens = result["low_dens"]

    if len(high_dens) > 0 and np.all(np.isfinite(y_fit)):
        y_max = 2 * np.nanmax(y_fit)

        filtered_high = []
        filtered_low = []

        for lo, hi in zip(low_dens, high_dens):
            if hi <= y_max:
                filtered_low.append(lo)
                filtered_high.append(hi)

        ave_low_dens.append(np.mean(filtered_low) if filtered_low else 0.0)
        ave_high_dens.append(np.mean(filtered_high) if filtered_high else 0.0)


# -------------------- RDF PASS --------------------
charge1 = []
peaks_1_1_xvals_1, peaks_1_2_xvals_1, peaks_2_2_xvals_1 = [], [], []
peaks_1_1_rdf_1, peaks_1_2_rdf_1, peaks_2_2_rdf_1 = [], [], []

for q in charges_to_plot:
    q_formatted = "{:.2f}".format(q)

    print("\n" + "=" * 70)
    print(f"Processing RDF q = {q_formatted}")
    print("=" * 70)

    result = process_rdf_three_replicas(q_formatted)

    if result is None:
        print(f"Skipping q = {q_formatted}: no replica RDF data")
        continue

    ave_rdf_1_1 = result["ave_rdf_1_1"]
    ave_rdf_1_2 = result["ave_rdf_1_2"]
    ave_rdf_2_2 = result["ave_rdf_2_2"]

    std_rdf_1_1 = result["std_rdf_1_1"]
    std_rdf_1_2 = result["std_rdf_1_2"]
    std_rdf_2_2 = result["std_rdf_2_2"]

    xvals = result["xvals"]

    if q_formatted in rdf_map:
        r_i, r_j = rdf_map[q_formatted]
        axR = rdf_axes_grid[r_i, r_j]
        charge1.append(q)

        axR.plot(
            xvals[0:100], ave_rdf_1_1[0:100],
            '-', color='blue', lw=2.5, label='M--M'
        )

        axR.plot(
            xvals[0:100], ave_rdf_2_2[0:100],
            '-', color='cyan', lw=2.5, label='DG--DG'
        )

        axR.plot(
            xvals[0:100], ave_rdf_1_2[0:100],
            '-', color='red', lw=2.5, label='M--DG'
        )

        if result["n_replicas"] > 1:
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
            0.05, 0.95,
            rf'$q^*_{{\mathrm{{M}}}} = {q:.0f}$',
            transform=axR.transAxes,
            ha='left', va='top', fontsize=24
        )

        axR.set_ylim(-1, 12)
        axR.set_xlim(-1, 25)
        axR.set_xticks(np.arange(0, 21, 10))
        axR.set_yticks(np.arange(0, 21, 10))

        if q_formatted == '16.00':
            axR.legend(
                frameon=False, borderpad=0.1, labelspacing=0.2,
                columnspacing=0.2, borderaxespad=0.4,
                handletextpad=0.4, fontsize='22',
                loc='upper right', handlelength=0.7
            )

        axR.xaxis.set_minor_locator(AutoMinorLocator())
        axR.yaxis.set_minor_locator(AutoMinorLocator())

        axR.tick_params(which='major', length=5, pad=4)
        axR.tick_params(which='minor', length=3)

        peaks_1_1, _ = find_peaks(ave_rdf_1_1)
        peaks_2_2, _ = find_peaks(ave_rdf_2_2)
        peaks_1_2, _ = find_peaks(ave_rdf_1_2)

        def report_peaks(name, peaks, rdf, xvals):
            if len(peaks) >= 1:
                print(
                    f"First peak of {name} at r = {xvals[peaks[0]]:.2f}, "
                    f"g(r) = {rdf[peaks[0]]:.2f}"
                )
            if len(peaks) >= 2:
                print(
                    f"Second peak of {name} at r = {xvals[peaks[1]]:.2f}, "
                    f"g(r) = {rdf[peaks[1]]:.2f}"
                )
            else:
                print(f"{name} has less than two peaks.")

        report_peaks("ave_rdf_1_1", peaks_1_1, ave_rdf_1_1, xvals)
        report_peaks("ave_rdf_2_2", peaks_2_2, ave_rdf_2_2, xvals)
        report_peaks("ave_rdf_1_2", peaks_1_2, ave_rdf_1_2, xvals)

        if len(peaks_1_1) >= 1:
            peaks_1_1_xvals_1.append(xvals[peaks_1_1[0]])
            peaks_1_1_rdf_1.append(ave_rdf_1_1[peaks_1_1[0]])

        if len(peaks_1_2) >= 1:
            peaks_1_2_xvals_1.append(xvals[peaks_1_2[0]])
            peaks_1_2_rdf_1.append(ave_rdf_1_2[peaks_1_2[0]])

        if len(peaks_2_2) >= 1:
            peaks_2_2_xvals_1.append(xvals[peaks_2_2[0]])
            peaks_2_2_rdf_1.append(ave_rdf_2_2[peaks_2_2[0]])


# -------------------- LAYOUT & SAVE --------------------
figD.tight_layout(h_pad=0.8)

# RDF margins
RDF_LEFT = 0.16
RDF_RIGHT = 0.98
RDF_BOTTOM = 0.16
RDF_TOP = 0.98

figR.subplots_adjust(
    left=RDF_LEFT, right=RDF_RIGHT,
    bottom=RDF_BOTTOM, top=RDF_TOP,
    hspace=0.10, wspace=0.10
)

# Center shared labels relative to the axis block, not the full figure
RDF_XCENTER = 0.5 * (RDF_LEFT + RDF_RIGHT)
RDF_YCENTER = 0.5 * (RDF_BOTTOM + RDF_TOP)

# Bold shared RDF labels
xlab = figR.supxlabel(r'$\mathbf{\textit{r}^{*}\ (\sigma^{*})}$',
                      fontsize=40)
xlab.set_position((0.57, 0.02))   # (x, y) in figure coords

ylab = figR.supylabel(r'RDF', fontsize=40)
ylab.set_position((-0.02, 0.5))   # (x, y) in figure coords

ylab.set_position((0.03, RDF_YCENTER))

figD.savefig(density_pdf, bbox_inches='tight')
figR.savefig(rdf_pdf, bbox_inches='tight')

print("Saved density figure to:", density_pdf)
print("Saved RDF figure to:", rdf_pdf)

plt.show()