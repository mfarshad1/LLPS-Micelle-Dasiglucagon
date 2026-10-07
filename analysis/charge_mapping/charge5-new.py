import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from Bio.PDB import PDBIO, StructureBuilder
import MDAnalysis as mda
import os
import subprocess
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ---------------- Global formatting (match original look) ----------------
rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize='28')
rc('xtick', labelsize='24')
rc('ytick', labelsize='24')

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Plain (non-math) tick formatter -> keeps the hyphen-minus narrow
def fmt_int_plain(x, pos):
    ix = int(round(x))
    return f"{ix}" if abs(x - ix) < 1e-9 else f"{x:.0f}"

# ---------------- Chemistry / charge model ----------------
sequence = "HSQGTFTSDYSKYLDAARAEGFVKWLEST"
PKA = {'N_term': 9.0, 'C_term': 2.0, 'D': 3.9, 'E': 4.2,
       'H': 6.0, 'K': 10.5, 'R': 12.5, 'Y': 10.1}
ACIDIC = {'D', 'E', 'Y'}
BASIC  = {'H', 'K', 'R'}

def fraction_deprotonated(pKa, pH):
    return 1 / (1 + 10**(pKa - pH))

def fraction_protonated(pKa, pH):
    return 1 / (1 + 10**(pH - pKa))

def calculate_net_charge(seq, pH):
    cnt = Counter(seq)
    q = 0.0
    q += fraction_protonated(PKA['N_term'], pH)
    q -= fraction_deprotonated(PKA['C_term'], pH)

    for aa, n in cnt.items():
        if aa in ACIDIC and aa in PKA:
            q -= n * fraction_deprotonated(PKA[aa], pH)
        elif aa in BASIC and aa in PKA:
            q += n * fraction_protonated(PKA[aa], pH)

    return q

# ---------------- Units & conversion ----------------
e_charge        = 1.602e-19
epsilon_0       = 8.854e-12
epsilon_r       = 78.5
epsilon_scale_J = 4.14e-21   # approximately kBT at 300 K
sigma_scale     = 1e-9       # 1 nm
a_star          = 0.5

def convert_to_reduced_unit(q_net, a_star):
    """
    Dielectric-included reduced charge for DG:
    q* = q / sqrt(4*pi*epsilon_0*epsilon_r*sigma*epsilon_LJ)
    """
    denom = np.sqrt(
        4 * np.pi * epsilon_0 * epsilon_r * sigma_scale * epsilon_scale_J
    )
    return q_net * e_charge / denom

# ---------------- Build arrays for final twin-axes plot (from .pka) ----------------
pka_file = "model2.pka"
pH_list, folded_charges = [], []

with open(pka_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            try:
                pH = float(parts[0])
                folded = float(parts[2])
                pH_list.append(pH)
                folded_charges.append(folded)
            except ValueError:
                pass

pH_array = np.array(pH_list)
charge_array = np.array(folded_charges)

idx = np.argsort(pH_array)
pH_array = pH_array[idx]
charge_array = charge_array[idx]

# ---------------- Plotting ----------------
pp = PdfPages("Charge_pH.pdf")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ===== Left panel: zeta vs q*_M =====
e = 1.602e-19
k_B = 1.381e-23
T = 300

sigma_real = 10e-9
sigma_lj = 10
a_i_star = sigma_lj / 2
sigma_scale_lr = sigma_real / sigma_lj

epsilon_real = 0.596
epsilon_LJ = 1.0
epsilon_scale_kcal_mol = epsilon_real / epsilon_LJ
kT_kcal = 0.596

lambda_B = (e**2) / (4 * np.pi * epsilon_0 * epsilon_r * k_B * T)

# Original micelle mapping convention used in the manuscript/phase diagram
# This keeps zeta = 10 mV -> q_M* ~ 20.3
prefactor = np.sqrt(sigma_scale_lr * epsilon_r / lambda_B)

charge_array1 = np.linspace(0, 50, 200)

kT_factor = k_B * T * 1000 / e
sroot = (epsilon_scale_kcal_mol / kT_kcal)**0.5

zeta_array_mV = (
    charge_array1 * kT_factor * sroot
) / (prefactor * a_i_star)

axes[0].plot(zeta_array_mV, charge_array1, color='black', linewidth=3)
axes[0].set_ylabel(r'$\mathbf{\textit{q}^{*}_{M}}$', fontsize=40)
axes[0].set_xlabel(r'$\mathbf{\zeta}$ (mV)', fontsize=40)

axes[0].set_xticks(np.arange(0, 25.1, 5))
axes[0].set_yticks(np.arange(0, 50.1, 10))

axes[0].set_xlim(0, 25)
axes[0].set_ylim(0, 50)

axes[0].xaxis.set_minor_locator(AutoMinorLocator())
axes[0].yaxis.set_minor_locator(AutoMinorLocator())

zeta_value = 10

qi = float(
    interp1d(
        zeta_array_mV,
        charge_array1,
        kind='linear',
        bounds_error=False,
        fill_value="extrapolate"
    )(zeta_value)
)

axes[0].axvline(zeta_value, linestyle='--', color='gray', linewidth=2)
axes[0].axhline(qi, linestyle='--', color='gray', linewidth=2)

axes[0].annotate(
    rf'$\zeta = {zeta_value}$\;mV',
    xy=(zeta_value + 0.25, 1.),
    xytext=(zeta_value + 3, 8),
    fontsize=20,
    color='black',
    arrowprops=dict(arrowstyle='->', color='black')
)

axes[0].annotate(
    rf'$q^*_{{M}} = {qi:.1f}$',
    xy=(0.2, qi+0.5),
    xytext=(1, qi + 10),
    fontsize=20,
    color='black',
    arrowprops=dict(arrowstyle='->', color='black')
)

# ===== Right panel: q*_{DG, bead} (left) vs q_DG (right), synced =====
s_linear = convert_to_reduced_unit(1.0, a_star)  # dielectric-included q* per 1e molecular charge

def left_to_right(y):  # bead reduced charge -> molecular real charge
    return (4.0 / s_linear) * y

def right_to_left(y):  # molecular real charge -> bead reduced charge
    return (s_linear / 4.0) * y

qstar_bead = right_to_left(charge_array)

axes[1].plot(pH_array, qstar_bead, color='black', linewidth=3, zorder=2)

axes[1].set_xlabel(r'$\mathbf{pH}$', fontsize=40)
axes[1].set_ylabel(
    r'$\mathbf{\textit{q}^{*}_{\mathrm{DG,\,bead}}}$',
    fontsize=40,
    color='black'
)

axes[1].tick_params(axis='y', colors='black')
axes[1].spines['left'].set_color('black')

ax2 = axes[1].secondary_yaxis('right', functions=(left_to_right, right_to_left))
ax2.set_ylabel(
    r'$\mathbf{\textit{q}_{\mathrm{DG}}}$',
    fontsize=40,
    color='black'
)

ax2.tick_params(axis='y', colors='black')
ax2.spines['right'].set_color('black')

axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(fmt_int_plain))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_int_plain))

axes[1].set_ylim(right_to_left(-9), right_to_left(7))

axes[1].xaxis.set_major_locator(ticker.MultipleLocator(2))
axes[1].xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
axes[1].yaxis.set_minor_locator(AutoMinorLocator(2))
axes[1].minorticks_on()

axes[1].tick_params(
    axis='y',
    which='minor',
    left=True,
    right=False,
    width=1.2,
    color='black'
)

ax2.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# Characteristic points
dq_dpH_smooth = gaussian_filter1d(np.gradient(charge_array, pH_array), sigma=1)
peaks, _ = find_peaks(np.abs(dq_dpH_smooth), prominence=0.5)

pk_pHs = sorted(pH_array[peaks])
first_eq_pH, second_eq_pH = pk_pHs[0], pk_pHs[1]

i_pI = np.argmin(np.abs(charge_array))
pI = pH_array[i_pI]

axes[1].plot(
    pI,
    right_to_left(0),
    's',
    color='black',
    markersize=12,
    zorder=10
)

axes[1].plot(
    first_eq_pH,
    right_to_left(charge_array[peaks[0]]),
    'o',
    color='black',
    markersize=12,
    zorder=10
)

axes[1].plot(
    second_eq_pH,
    right_to_left(charge_array[peaks[1]]),
    'o',
    color='black',
    markersize=12,
    zorder=10
)

# -------------------------------------------------------------------
# Dashed-line target: find pH where q_DG = -6e
# and report corresponding reduced DG bead charge
# -------------------------------------------------------------------
q_DG_target = -6.0

charge_interp = interp1d(
    pH_array,
    charge_array,
    kind='linear',
    bounds_error=False,
    fill_value="extrapolate"
)

fvals = charge_array - q_DG_target
crossing_indices = np.where(fvals[:-1] * fvals[1:] <= 0)[0]

if len(crossing_indices) == 0:
    i_target = np.argmin(np.abs(charge_array - q_DG_target))
    pH_target = pH_array[i_target]
    q_DG_at_target = charge_array[i_target]
else:
    i0 = crossing_indices[0]
    pH_target = brentq(
        lambda x: float(charge_interp(x) - q_DG_target),
        pH_array[i0],
        pH_array[i0 + 1]
    )
    q_DG_at_target = q_DG_target

q_bead_target = right_to_left(q_DG_target)

q_net_7 = float(interp1d(pH_array, charge_array, kind='linear')(7))
q_bead_7 = right_to_left(q_net_7)

# Dashed guides
axes[1].axvline(pH_target, linestyle='--', color='black', linewidth=2)
axes[1].axhline(q_bead_target, linestyle='--', color='black', linewidth=2)

# axes[1].axvline(7, linestyle='--', color='black', linewidth=2)
# axes[1].axhline(q_bead_7, linestyle='--', color='black', linewidth=2)

plt.tight_layout()
plt.show()

pp.savefig(fig, bbox_inches='tight')
pp.close()

print(f"Dielectric-included 1e reduced charge for DG: {s_linear:.3f}")
print(f"pH where q_DG = {q_DG_target:.1f}e: {pH_target:.3f}")
print(f"Reduced DG bead charge at q_DG = {q_DG_target:.1f}e: {q_bead_target:.3f}")
print(f"Total reduced DG charge at q_DG = {q_DG_target:.1f}e: {4*q_bead_target:.3f}")
print(f"Micelle reduced charge at zeta = {zeta_value:.1f} mV: {qi:.3f}")