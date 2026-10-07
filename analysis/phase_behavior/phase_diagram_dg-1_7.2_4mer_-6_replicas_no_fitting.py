import os
import numpy as np
import pandas as pd

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


# ============================================================
# Plot style
# ============================================================

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize='28')
rc('xtick', labelsize='24')
rc('ytick', labelsize='24')

colors = ['k'] * 12


# ============================================================
# User settings
# ============================================================

number = '4mer_-6'

base_dir = "/groups/jwhitme1/Data-MF-01/LLPS/llps-data/biphasic/dg-1-7_2/4mer_-6"
output_dir = "/groups/jwhitme1/Data-MF-01/LLPS/llps-data/biphasic/dg-1-7_2"

replicas = [1, 2, 3]

# This gives q = 8, 9, ..., 28.
q_start = 8
n_charges = 21

dx = 0.3
number_bins = 1000

DISCARD_INITIAL_BLOCKS = 0

# Physical volumes
dg_length_nm = 4
dg_radius_nm = 0.5
dg_volume_nm3 = np.pi * dg_radius_nm**2 * dg_length_nm

micelle_radius_nm = 5
micelle_volume_nm3 = (4 / 3) * np.pi * micelle_radius_nm**3


# ============================================================
# Density-file reading
# ============================================================

def read_density_blocks(path):
    """
    Read LAMMPS fix ave/chunk density file.
    Keeps only bin/chunk lines where the first column is 1...number_bins.
    Skips timestep header lines and comment lines.
    """
    rows = []

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            try:
                first_col = float(parts[0])
            except ValueError:
                continue

            if first_col <= number_bins:
                rows.append([
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3])
                ])

    all_density = np.array(rows)

    if all_density.size == 0:
        return []

    block_size = number_bins
    num_blocks = len(all_density) // block_size

    blocks = []

    for i in range(num_blocks):
        if i <= DISCARD_INITIAL_BLOCKS:
            continue

        block = all_density[i * block_size:(i + 1) * block_size]

        if block.shape[0] == number_bins:
            blocks.append(block)

    return blocks


def average_density_profile(path):
    """
    Average density profile over selected time blocks.
    """
    blocks = read_density_blocks(path)

    if len(blocks) == 0:
        return None

    ave_block = np.mean(blocks, axis=0)

    xvals = ave_block[:, 1]
    dens = ave_block[:, 3]

    return {
        "xvals": xvals,
        "dens": dens,
        "n_blocks": len(blocks)
    }


# ============================================================
# Direct high/low density extraction without fitting
# ============================================================

def direct_region_densities(xvals, dens):
    """
    Estimate phase densities directly from fixed spatial regions.

    Dense phase:
        central plateau-like region

    Dilute phase:
        outer regions

    Interfacial regions are ignored.
    """
    xvals = np.asarray(xvals)
    dens = np.asarray(dens)

    x_left = np.min(xvals) - 0.5 * dx
    x_right = np.max(xvals) + 0.5 * dx
    Lbox = x_right - x_left

    s = (xvals - x_left) / Lbox

    low_mask = (s < 1.5 / 6.0) | (s >= 4.5 / 6.0)
    high_mask = (s >= 2.0 / 6.0) & (s < 4.0 / 6.0)

    rho_low = np.mean(dens[low_mask])
    rho_high = np.mean(dens[high_mask])

    low_width = (3.0 / 6.0) * Lbox
    high_width = (2.0 / 6.0) * Lbox
    transition_width = Lbox - low_width - high_width

    return {
        "rho_low": rho_low,
        "rho_high": rho_high,
        "low_width": low_width,
        "high_width": high_width,
        "transition_width": transition_width,
        "Lbox": Lbox,
        "n_low_bins": int(np.sum(low_mask)),
        "n_high_bins": int(np.sum(high_mask))
    }


def process_one_replica(q, rep):
    """
    Process one charge and one replica using direct region averaging.
    """
    q_formatted = f"{q:.2f}"

    repdir = os.path.join(base_dir, f"q-{q_formatted}", f"rep{rep}")

    path_micelle = os.path.join(
        repdir,
        f"density_micelle.q-{q_formatted}.data"
    )

    path_dg = os.path.join(
        repdir,
        f"density_polymer.q-{q_formatted}.data"
    )

    if not os.path.exists(path_micelle) or not os.path.exists(path_dg):
        print(f"Skipping q={q_formatted}, rep={rep}: missing density file")
        return None

    result_dg = average_density_profile(path_dg)
    result_micelle = average_density_profile(path_micelle)

    if result_dg is None or result_micelle is None:
        print(f"Skipping q={q_formatted}, rep={rep}: no density blocks")
        return None

    dg_regions = direct_region_densities(
        result_dg["xvals"],
        result_dg["dens"]
    )

    micelle_regions = direct_region_densities(
        result_micelle["xvals"],
        result_micelle["dens"]
    )

    vol_coacervate = dg_regions["high_width"]
    vol_supernatant = dg_regions["low_width"]
    vol_transition = dg_regions["transition_width"]
    vol_total = dg_regions["Lbox"]

    phi_coacervate = vol_coacervate / vol_total
    phi_supernatant = vol_supernatant / vol_total
    phi_transition = vol_transition / vol_total

    # DG density file counts DG beads; divide by 4 to get DG molecules.
    rho_dg_low = dg_regions["rho_low"] / 4.0
    rho_dg_high = dg_regions["rho_high"] / 4.0

    rho_micelle_low = micelle_regions["rho_low"]
    rho_micelle_high = micelle_regions["rho_high"]

    # Convert number density to volume fraction.
    phi_dg_super = rho_dg_low * dg_volume_nm3
    phi_dg_dense = rho_dg_high * dg_volume_nm3

    phi_micelle_super = rho_micelle_low * micelle_volume_nm3
    phi_micelle_dense = rho_micelle_high * micelle_volume_nm3

    vol_dg_dense = phi_dg_dense * vol_coacervate
    vol_micelle_dense = phi_micelle_dense * vol_coacervate

    return {
        "Charge": q,
        "Replica": rep,

        "DG vol (dense)": vol_dg_dense,
        "Micelle vol (dense)": vol_micelle_dense,
        "DG/Micelle ratio": (
            vol_dg_dense / vol_micelle_dense
            if vol_micelle_dense > 0 else np.nan
        ),

        "Coacervate frac": phi_coacervate,
        "Supernatant frac": phi_supernatant,
        "Transition frac": phi_transition,

        "DG phi (coacervate)": phi_dg_dense,
        "DG phi (supernatant)": phi_dg_super,

        "Micelle phi (coacervate)": phi_micelle_dense,
        "Micelle phi (supernatant)": phi_micelle_super,

        "low dens DG": dg_regions["rho_low"],
        "high dens DG": dg_regions["rho_high"],

        "low dens micelle": micelle_regions["rho_low"],
        "high dens micelle": micelle_regions["rho_high"],

        "DG blocks used": result_dg["n_blocks"],
        "Micelle blocks used": result_micelle["n_blocks"],

        "Dense bins used": dg_regions["n_high_bins"],
        "Dilute bins used": dg_regions["n_low_bins"],
        "Lbox": dg_regions["Lbox"]
    }


# ============================================================
# Main analysis
# ============================================================

rows = []

for n in range(n_charges):
    q = q_start + n
    q_formatted = f"{q:.2f}"

    print("\n" + "=" * 70)
    print(f"Processing q = {q_formatted}")
    print("=" * 70)

    for rep in replicas:
        row = process_one_replica(q, rep)

        if row is not None:
            print(
                f"q={q_formatted}, rep={rep}: "
                f"DG blocks={row['DG blocks used']}, "
                f"micelle blocks={row['Micelle blocks used']}, "
                f"phi_DG_dense={row['DG phi (coacervate)']:.5f}, "
                f"phi_M_dense={row['Micelle phi (coacervate)']:.5f}"
            )

            rows.append(row)

df_rep = pd.DataFrame(rows)

if df_rep.empty:
    raise RuntimeError("No replica data were processed. Check density file paths.")


# ============================================================
# Save replica data and compute mean/std
# ============================================================

output_replica_csv = os.path.join(
    output_dir,
    f"phase_data_{number}_direct_L6_3replicas_raw.csv"
)

df_rep.to_csv(output_replica_csv, index=False)

df = df_rep.groupby("Charge", as_index=False).mean(numeric_only=True)
df_std = df_rep.groupby("Charge", as_index=False).std(numeric_only=True).fillna(0)

output_mean_csv = os.path.join(
    output_dir,
    f"phase_data_{number}_direct_L6_3replicas_mean.csv"
)

output_std_csv = os.path.join(
    output_dir,
    f"phase_data_{number}_direct_L6_3replicas_std.csv"
)

df.to_csv(output_mean_csv, index=False)
df_std.to_csv(output_std_csv, index=False)

print("\nSaved raw replica data to:", output_replica_csv)
print("Saved mean data to:", output_mean_csv)
print("Saved std data to:", output_std_csv)


# ============================================================
# Convert to arrays for plotting
# ============================================================

charge = df["Charge"].to_numpy()

phi_dg_coacervate = df["DG phi (coacervate)"].to_numpy()
phi_dg_supernatant = df["DG phi (supernatant)"].to_numpy()

phi_micelle_coacervate = df["Micelle phi (coacervate)"].to_numpy()
phi_micelle_supernatant = df["Micelle phi (supernatant)"].to_numpy()

err_phi_dg_coacervate = df_std["DG phi (coacervate)"].to_numpy()
err_phi_dg_supernatant = df_std["DG phi (supernatant)"].to_numpy()

err_phi_micelle_coacervate = df_std["Micelle phi (coacervate)"].to_numpy()
err_phi_micelle_supernatant = df_std["Micelle phi (supernatant)"].to_numpy()


# ============================================================
# Plotting
# ============================================================

output_pdf = os.path.join(
    output_dir,
    f"phase_and_volume_fraction_{number}_direct_L6_3replicas_common_y.pdf"
)

print("Saving figure to:", output_pdf)
print("Can write to output directory?", os.access(output_dir, os.W_OK))

pp = PdfPages(output_pdf)

# Minimal change: share the y-axis between the two panels.
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)


# ============================================================
# Zeta potential calculation
# ============================================================

e = 1.602e-19
epsilon_0 = 8.854e-12
epsilon_r = 78.5
k_B = 1.381e-23
T = 300

sigma_real = 10e-9
sigma_lj = 10
a_i_star = sigma_lj / 2

sigma_scale = sigma_real / sigma_lj

epsilon_real = 0.596
epsilon_LJ = 1.0
epsilon_scale = epsilon_real / epsilon_LJ
kT_kcal = 0.596

lambda_B = (e**2) / (4 * np.pi * epsilon_0 * epsilon_r * k_B * T)
prefactor = np.sqrt(sigma_scale * epsilon_r / lambda_B)
kT_factor = k_B * T * 1000 / e
squared_root_eps_scale_to_kT = (epsilon_scale / kT_kcal) ** 0.5

charge_array = np.array(charge)

zeta_array_mV = (
    charge_array * kT_factor * squared_root_eps_scale_to_kT
) / (prefactor * a_i_star)


def q_to_zeta(q):
    return (q * kT_factor * squared_root_eps_scale_to_kT) / (prefactor * a_i_star)


def sync_right_ylim(ax_left, ax_right):
    ymin, ymax = ax_left.get_ylim()
    ax_right.set_ylim(q_to_zeta(ymin), q_to_zeta(ymax))


def style_left_and_x(ax):
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", length=3)


def add_zeta_axis_only(ax):
    """
    Add only the right-side zeta axis.
    No duplicate data are plotted on the twin axis.
    """
    ax2 = ax.twinx()

    ax2.set_ylabel(r"$\mathbf{\zeta}$ (mV)", fontsize=40, color="black")
    ax2.tick_params(axis="y", colors="black", labelcolor="black")
    ax2.spines["right"].set_color("black")

    ax2.yaxis.set_major_locator(MultipleLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.tick_params(which="minor", length=3, colors="black")

    sync_right_ylim(ax, ax2)
    ax.callbacks.connect("ylim_changed", lambda a: sync_right_ylim(a, ax2))

    return ax2


# ============================================================
# Panel 1: Micelle
# ============================================================

ax_m = axes[0]

ax_m.errorbar(
    phi_micelle_supernatant,
    charge,
    xerr=err_phi_micelle_supernatant,
    fmt="o--",
    color=colors[2],
    markersize=12,
    mfc="none",
    lw=2,
    capsize=3
)

ax_m.errorbar(
    phi_micelle_coacervate,
    charge,
    xerr=err_phi_micelle_coacervate,
    fmt="s--",
    color=colors[3],
    markersize=12,
    mfc="none",
    lw=2,
    capsize=3
)

ax_m.set_xlabel(r"$\mathbf{\phi_{M}}$", fontsize=40)

# q-axis only on the left panel.
ax_m.set_ylabel(
    r"$\mathbf{\mathit{q}^{*}_{M}}$",
    fontsize=40,
    color="black",
    labelpad=10
)

ax_m.set_xlim(0.0, 0.4)
ax_m.set_ylim(2, 32)

ax_m.tick_params(axis="y", colors="black", labelcolor="black")
ax_m.spines["left"].set_color("black")

style_left_and_x(ax_m)


# ============================================================
# Panel 2: DG
# ============================================================

ax_dg = axes[1]

ax_dg.errorbar(
    phi_dg_supernatant,
    charge,
    xerr=err_phi_dg_supernatant,
    fmt="o--",
    color=colors[2],
    markersize=12,
    mfc="none",
    lw=2,
    capsize=3
)

ax_dg.errorbar(
    phi_dg_coacervate,
    charge,
    xerr=err_phi_dg_coacervate,
    fmt="s--",
    color=colors[3],
    markersize=12,
    mfc="none",
    lw=2,
    capsize=3
)

ax_dg.set_xlabel(r"$\mathbf{\phi_{DG}}$", fontsize=40)

# Remove duplicate q-axis labels/ticks from the right panel.
ax_dg.set_ylabel("")
ax_dg.tick_params(axis="y", which="both", left=False, labelleft=False)
ax_dg.spines["left"].set_visible(False)

ax_dg.set_xlim(-0.003, 0.02)
ax_dg.set_ylim(2, 32)

style_left_and_x(ax_dg)

# zeta-axis only on the right side of the right panel.
ax_zeta = add_zeta_axis_only(ax_dg)


plt.tight_layout()

pp.savefig(fig, bbox_inches="tight")
pp.close()

print("Saved figure to:", output_pdf)

plt.show()