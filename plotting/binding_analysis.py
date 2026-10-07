from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator


rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{bm}')
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

rc('axes', labelsize=28)
rc('xtick', labelsize=24)
rc('ytick', labelsize=24)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

REPO = Path(__file__).resolve().parents[1]
summary_txt = REPO / "reduced_data" / "binding" / "dg_bound_unbound_analysis_4mer_-6_fixed_regions_3replicas.txt"
output_pdf = REPO / "reproduced_figures" / "figure7_binding_analysis.pdf"
output_pdf.parent.mkdir(parents=True, exist_ok=True)

arr = np.loadtxt(summary_txt, comments="#")
if arr.ndim == 1:
    arr = arr[None, :]

# Column order matches original np.savetxt header
charges = arr[:, 0]
bound_dense = arr[:, 2]
bound_dense_std = arr[:, 3]
unbound_dense = arr[:, 4]
unbound_dense_std = arr[:, 5]
bound_dilute = arr[:, 6]
bound_dilute_std = arr[:, 7]
unbound_dilute = arr[:, 8]
unbound_dilute_std = arr[:, 9]

x = np.arange(len(charges))
w = 0.18

# Wide one-panel figure, but compatible with other manuscript plots
fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.5))

ax.bar(
    x - 1.5 * w,
    bound_dense,
    width=w,
    yerr=bound_dense_std,
    capsize=3,
    label='Bound, central'
)

ax.bar(
    x - 0.5 * w,
    unbound_dense,
    width=w,
    yerr=unbound_dense_std,
    capsize=3,
    label='Unbound, central'
)

ax.bar(
    x + 0.5 * w,
    bound_dilute,
    width=w,
    yerr=bound_dilute_std,
    capsize=3,
    label='Bound, outer'
)

ax.bar(
    x + 1.5 * w,
    unbound_dilute,
    width=w,
    yerr=unbound_dilute_std,
    capsize=3,
    label='Unbound, outer'
)

ax.set_ylim(None, 700)
ax.set_xticks(x)
ax.set_xticklabels([rf'${int(q)}$' for q in charges], fontsize=24)

ax.set_xlabel(r'$\mathbf{\mathit{q}^{*}_{M}}$', fontsize=40, labelpad=5)
ax.set_ylabel(r'{DG count}', fontsize=40, labelpad=8)

ax.legend(
    frameon=False,
    borderpad=0.1,
    labelspacing=0.2,
    columnspacing=0.2,
    borderaxespad=0.4,
    handletextpad=0.4,
    fontsize='20',
    loc='upper right',
    handlelength=0.8
)

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.tick_params(which='major', length=5, pad=4)
ax.tick_params(which='minor', length=3)

plt.tight_layout()
plt.savefig(output_pdf, bbox_inches='tight', dpi=300)

print("Saved figure to:", output_pdf)
plt.show()
