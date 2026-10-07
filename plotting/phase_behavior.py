from pathlib import Path
import os
import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize='28')
rc('xtick', labelsize='24')
rc('ytick', labelsize='24')

REPO = Path(__file__).resolve().parents[1]
number = "4mer_-6"

mean_csv = REPO / "reduced_data" / "phase_behavior" / "phase_data_4mer_-6_direct_L6_3replicas_mean.csv"
std_csv  = REPO / "reduced_data" / "phase_behavior" / "phase_data_4mer_-6_direct_L6_3replicas_std.csv"

output_dir = REPO / "reproduced_figures"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(mean_csv)
df_std = pd.read_csv(std_csv)

colors = ['k'] * 12

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
output_pdf = output_dir / "figure5_phase_behavior.pdf"

print("Saving figure to:", output_pdf)

pp = PdfPages(str(output_pdf))

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
