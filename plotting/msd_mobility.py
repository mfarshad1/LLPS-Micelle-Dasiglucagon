#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator

# ============================================================
# Plot settings: compatible with phase/RDF style
# ============================================================
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

# ============================================================
# Constants
# ============================================================
number = '4mer_-6'

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.join(repo_dir, "reduced_data", "msd")
output_dir = os.path.join(repo_dir, "reproduced_figures")
os.makedirs(output_dir, exist_ok=True)

replicas = [1, 2, 3]

q_start = 8
num_q = 5
q_step = 4

dt = 0.0001

colors = ['green', 'blue', 'purple', 'orange', 'red', 'black']
d = 3  # dimensionality

# Reduced thermal energy used in the Einstein relation:
#     D* = mu* kBT*
# Therefore:
#     mu* = D* / kBT*
#
# For the present LJ reduced-unit simulations, kBT* = 1.0.
# Change this value only if the simulation temperature is different.
kBT_star = 1.0

charge = []
mobility = []
mobility_std = []

# Define averaging range
fit_t_min = 0.0
fit_t_max = 10000000.0

# ============================================================
# REGION SETTINGS
#
# These are exactly the same fixed regions used in the
# direct-region phase-diagram code:
#
# outer:       s < 1.5/6 or s >= 4.5/6
# central:     2.0/6 <= s < 4.0/6
# transition:  ignored
#
# Here s = (z - zlo)/Lz, wrapped into 0 <= s < 1.
# ============================================================
region_to_analyze = "central"  # "central" or "outer"

outer_left_end = 1.5 / 6.0
central_start = 2.0 / 6.0
central_end = 4.0 / 6.0
outer_right_start = 4.5 / 6.0

# Rebuild msd_region_com.dat directly from eq_T.dump so that
# the region selection is guaranteed to match the phase diagram.
rebuild_msd_from_dump = False
dump_filename = "eq_T.dump"
msd_filename = "msd_region_com.dat"

# Atom types included in the regional MSD group.
# [2] includes only DG beads, consistent with the MSD caption
# and the intended DG mobility calculation.
atom_types_to_include = [2]

# Type masses used only for center-of-mass drift removal.
# Keep these consistent with the masses in the LAMMPS model.
atom_type_masses = {
    1: 1.0,
    2: 1.0,
}

# Equivalent to subtracting the drift of the selected group's COM.
subtract_group_com_drift = True

# Write the rebuilt MSD to the same filename used by the original code.
write_rebuilt_msd_file = False


# ============================================================
# Region utilities
# ============================================================
def shift_to_0_L(value, lo, hi):
    L = hi - lo
    return (value - lo) % L


def position_in_selected_region(z, zlo, zhi):
    """
    Apply exactly the same fixed spatial masks as the phase-diagram code.
    Transition-region positions are excluded.
    """
    Lz = zhi - zlo
    z_shift = shift_to_0_L(z, zlo, zhi)
    s = z_shift / Lz

    in_central = (s >= central_start) and (s < central_end)
    in_outer = (s < outer_left_end) or (s >= outer_right_start)

    if region_to_analyze == "central":
        return in_central

    if region_to_analyze == "outer":
        return in_outer

    raise ValueError(
        f"Unknown region_to_analyze={region_to_analyze!r}. "
        "Use 'central' or 'outer'."
    )


def minimum_image_displacement(displacement, box):
    return displacement - box * np.round(displacement / box)


# ============================================================
# LAMMPS dump reader
# ============================================================
def read_lammps_dump_frames(dump_file):
    """
    Read a LAMMPS custom dump.

    Supported coordinate columns:
        x y z
        xu yu zu
        xs ys zs

    Both wrapped coordinates and, when available, unwrapped coordinates
    are returned. If the dump contains only wrapped coordinates, the
    selected particles are unwrapped later by minimum-image continuity.
    """
    if not os.path.exists(dump_file):
        raise FileNotFoundError(f"Dump file not found: {dump_file}")

    frames = []

    with open(dump_file, "r") as f:
        while True:
            line = f.readline()

            if line == "":
                break

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(float(f.readline().strip()))

            line = f.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(
                    f"Unexpected NUMBER OF ATOMS section in {dump_file}"
                )

            n_atoms = int(float(f.readline().strip()))

            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(
                    f"Unexpected BOX BOUNDS section in {dump_file}"
                )

            xlo, xhi = map(float, f.readline().split()[:2])
            ylo, yhi = map(float, f.readline().split()[:2])
            zlo, zhi = map(float, f.readline().split()[:2])

            atom_header = f.readline().strip()
            if not atom_header.startswith("ITEM: ATOMS"):
                raise ValueError(
                    f"Unexpected ATOMS section in {dump_file}"
                )

            columns = atom_header.split()[2:]
            col = {name: i for i, name in enumerate(columns)}

            if "id" not in col or "type" not in col:
                raise ValueError(
                    f"id/type not found in ATOMS header: {atom_header}"
                )

            has_wrapped = all(name in col for name in ("x", "y", "z"))
            has_unwrapped = all(name in col for name in ("xu", "yu", "zu"))
            has_scaled = all(name in col for name in ("xs", "ys", "zs"))

            if not (has_wrapped or has_unwrapped or has_scaled):
                raise ValueError(
                    "Could not find x/y/z, xu/yu/zu, or xs/ys/zs "
                    f"in ATOMS header: {atom_header}"
                )

            box = np.array([xhi - xlo, yhi - ylo, zhi - zlo], dtype=float)
            lower = np.array([xlo, ylo, zlo], dtype=float)

            atoms = {}

            for _ in range(n_atoms):
                values = f.readline().split()

                atom_id = int(float(values[col["id"]]))
                atom_type = int(float(values[col["type"]]))

                unwrapped = None

                if has_wrapped:
                    wrapped = np.array(
                        [
                            float(values[col["x"]]),
                            float(values[col["y"]]),
                            float(values[col["z"]]),
                        ],
                        dtype=float
                    )

                    # Ensure a 0...L-equivalent wrapped representation.
                    wrapped = lower + ((wrapped - lower) % box)

                    if has_unwrapped:
                        unwrapped = np.array(
                            [
                                float(values[col["xu"]]),
                                float(values[col["yu"]]),
                                float(values[col["zu"]]),
                            ],
                            dtype=float
                        )

                elif has_scaled:
                    scaled = np.array(
                        [
                            float(values[col["xs"]]),
                            float(values[col["ys"]]),
                            float(values[col["zs"]]),
                        ],
                        dtype=float
                    )

                    wrapped = lower + (scaled % 1.0) * box

                else:
                    unwrapped = np.array(
                        [
                            float(values[col["xu"]]),
                            float(values[col["yu"]]),
                            float(values[col["zu"]]),
                        ],
                        dtype=float
                    )

                    wrapped = lower + ((unwrapped - lower) % box)

                atoms[atom_id] = {
                    "type": atom_type,
                    "wrapped": wrapped,
                    "unwrapped": unwrapped
                }

            frames.append({
                "timestep": timestep,
                "bounds": ((xlo, xhi), (ylo, yhi), (zlo, zhi)),
                "box": box,
                "atoms": atoms,
                "has_unwrapped": has_unwrapped
            })

    if len(frames) < 3:
        raise ValueError(f"Too few frames found in {dump_file}")

    return frames


# ============================================================
# Rebuild regional MSD using the phase-diagram region
# ============================================================
def build_region_msd_from_dump(dump_file):
    """
    Select a fixed group from the requested region in the first frame,
    then calculate its MSD over time.

    The fixed membership is necessary for a meaningful MSD: the same
    particle IDs are tracked in every frame. Region membership is
    determined from the first frame using exactly the same central/outer
    spatial masks as the phase-diagram code.

    If subtract_group_com_drift is True, the displacement of the selected
    group's center of mass is removed before the MSD is evaluated.
    """
    frames = read_lammps_dump_frames(dump_file)

    first = frames[0]
    (_, _), (_, _), (zlo0, zhi0) = first["bounds"]

    selected_ids = []

    for atom_id, atom in first["atoms"].items():
        atom_type = atom["type"]

        if atom_types_to_include is not None:
            if atom_type not in atom_types_to_include:
                continue

        z = atom["wrapped"][2]

        if position_in_selected_region(z, zlo0, zhi0):
            selected_ids.append(atom_id)

    selected_ids = sorted(selected_ids)

    if len(selected_ids) == 0:
        raise ValueError(
            f"No atoms were found in the {region_to_analyze} region "
            f"of the first frame in {dump_file}"
        )

    selected_types = np.array(
        [first["atoms"][atom_id]["type"] for atom_id in selected_ids],
        dtype=int
    )

    masses = np.array(
        [atom_type_masses.get(atom_type, 1.0) for atom_type in selected_types],
        dtype=float
    )

    initial_positions = None
    previous_wrapped = None
    previous_unwrapped = None

    time_values = []
    msd_values = []

    first_timestep = first["timestep"]

    for frame_index, frame in enumerate(frames):
        atoms = frame["atoms"]
        box = frame["box"]

        missing_ids = [atom_id for atom_id in selected_ids if atom_id not in atoms]

        if missing_ids:
            raise ValueError(
                f"{len(missing_ids)} selected atoms are missing at "
                f"timestep {frame['timestep']} in {dump_file}"
            )

        wrapped = np.array(
            [atoms[atom_id]["wrapped"] for atom_id in selected_ids],
            dtype=float
        )

        frame_has_unwrapped = all(
            atoms[atom_id]["unwrapped"] is not None
            for atom_id in selected_ids
        )

        if frame_has_unwrapped:
            unwrapped = np.array(
                [atoms[atom_id]["unwrapped"] for atom_id in selected_ids],
                dtype=float
            )

        elif frame_index == 0:
            unwrapped = wrapped.copy()

        else:
            displacement = wrapped - previous_wrapped
            displacement = minimum_image_displacement(displacement, box)
            unwrapped = previous_unwrapped + displacement

        if frame_index == 0:
            initial_positions = unwrapped.copy()

        displacement_from_start = unwrapped - initial_positions

        if subtract_group_com_drift:
            com_displacement = np.average(
                displacement_from_start,
                axis=0,
                weights=masses
            )
            displacement_for_msd = (
                displacement_from_start - com_displacement[None, :]
            )
        else:
            displacement_for_msd = displacement_from_start

        squared_displacement = np.sum(
            displacement_for_msd**2,
            axis=1
        )

        # Arithmetic mean over the fixed selected group.
        msd = np.mean(squared_displacement)

        time_values.append((frame["timestep"] - first_timestep) * dt)
        msd_values.append(msd)

        previous_wrapped = wrapped.copy()
        previous_unwrapped = unwrapped.copy()

    return {
        "time": np.asarray(time_values, dtype=float),
        "msd": np.asarray(msd_values, dtype=float),
        "n_selected": len(selected_ids),
        "selected_ids": selected_ids,
        "region": region_to_analyze,
        "dump_file": dump_file
    }


def save_region_msd_file(msd_file, result):
    header = (
        "time_reduced msd_reduced\n"
        f"region={result['region']}\n"
        f"central={central_start:.12f}:{central_end:.12f}\n"
        f"outer=0:{outer_left_end:.12f},"
        f"{outer_right_start:.12f}:1\n"
        f"transition={outer_left_end:.12f}:{central_start:.12f},"
        f"{central_end:.12f}:{outer_right_start:.12f}\n"
        f"atom_types={atom_types_to_include}\n"
        f"n_selected={result['n_selected']}\n"
        f"subtract_group_com_drift={subtract_group_com_drift}"
    )

    np.savetxt(
        msd_file,
        np.column_stack([result["time"], result["msd"]]),
        header=header
    )


# ============================================================
# Helper functions
# ============================================================
def read_one_replica_msd(q_formatted, rep):
    qint = int(float(q_formatted))
    msd_file = os.path.join(base_dir, f"q{qint}_rep{rep}_msd_region_com.dat")
    dump_file = ""

    if rebuild_msd_from_dump:
        try:
            rebuilt = build_region_msd_from_dump(dump_file)
        except Exception as exc:
            print(
                f"Skipping q = {q_formatted}, rep = {rep}: "
                f"could not rebuild regional MSD: {exc}"
            )
            return None

        time = rebuilt["time"]
        msd = rebuilt["msd"]

        if write_rebuilt_msd_file:
            save_region_msd_file(msd_file, rebuilt)

        print(
            f"q = {q_formatted}, rep = {rep}: "
            f"selected {rebuilt['n_selected']} atoms from the "
            f"{rebuilt['region']} region"
        )

    else:
        if not os.path.exists(msd_file):
            print(
                f"Skipping q = {q_formatted}, rep = {rep}: "
                "MSD file not found."
            )
            return None

        data = np.loadtxt(msd_file, comments="#")

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[0] < 3:
            print(
                f"Too few data points for q = {q_formatted}, rep = {rep}"
            )
            return None

        time = data[:, 0]
        msd = data[:, 1]

    if len(time) < 3:
        print(
            f"Too few MSD data points for q = {q_formatted}, rep = {rep}"
        )
        return None

    # Compute the instantaneous diffusion coefficient from the MSD:
    #     D* = (1 / 2d) d<Delta r^2>/dt
    #
    # Convert diffusion to mobility using the Einstein relation:
    #     mu* = D* / kBT*
    dmsd_dt = np.gradient(msd, time)
    diffusion_t = dmsd_dt / (2 * d)
    mu_t = diffusion_t / kBT_star

    # Select averaging region.
    fit_mask = (time >= fit_t_min) & (time <= fit_t_max)

    if np.sum(fit_mask) < 3:
        print(
            f"Not enough points in averaging window for "
            f"q = {q_formatted}, rep = {rep}"
        )
        return None

    mu_avg = np.mean(mu_t[fit_mask])

    return {
        "time": time,
        "msd": msd,
        "mu_avg": mu_avg,
        "file": msd_file
    }


def process_three_replicas(q_formatted):
    rep_results = []

    for rep in replicas:
        result = read_one_replica_msd(q_formatted, rep)

        if result is not None:
            print(
                f"q = {q_formatted}, rep = {rep}: "
                f"mu = {result['mu_avg']:.6e}, "
                f"file = {result['file']}"
            )
            rep_results.append(result)

    if len(rep_results) == 0:
        return None

    time_ref = rep_results[0]["time"]

    msd_reps = []
    mu_reps = []

    for result in rep_results:
        time = result["time"]
        msd = result["msd"]

        if len(time) != len(time_ref) or not np.allclose(time, time_ref):
            msd_interp = np.interp(time_ref, time, msd)
            msd_reps.append(msd_interp)
        else:
            msd_reps.append(msd)

        mu_reps.append(result["mu_avg"])

    msd_reps = np.array(msd_reps)
    mu_reps = np.array(mu_reps)

    return {
        "time": time_ref,
        "msd_mean": np.mean(msd_reps, axis=0),
        "msd_std": np.std(msd_reps, axis=0),
        "mu_mean": np.mean(mu_reps),
        "mu_std": np.std(mu_reps),
        "n_replicas": len(rep_results)
    }


# ============================================================
# Create figure
# ============================================================
output_pdf = os.path.join(
    output_dir,
    'figure9_msd_mobility.pdf'
)

pp = PdfPages(output_pdf)

# Match the phase plot size/aspect
fig, (ax_msd, ax_mobility) = plt.subplots(1, 2, figsize=(11, 5))

for n in range(num_q):
    q = q_start + n * q_step
    q_formatted = "{:.2f}".format(q)

    print("\n" + "=" * 70)
    print(f"Processing q = {q_formatted}")
    print("=" * 70)

    result = process_three_replicas(q_formatted)

    if result is None:
        print(
            f"Skipping q = {q_formatted}: "
            "no usable replica MSD data."
        )
        continue

    time = result["time"]
    msd_mean = result["msd_mean"]
    msd_std = result["msd_std"]

    charge.append(q)
    mobility.append(result["mu_mean"])
    mobility_std.append(result["mu_std"])

    color = colors[n % len(colors)]

    # Plot replica-averaged MSD
    ax_msd.plot(
        time,
        msd_mean,
        lw=2.0,
        color=color,
        label=fr'$\mathit{{q}}_{{\mathrm{{M}}}}^* = {q_formatted}$'
    )

    # Standard deviation across replicas
    if result["n_replicas"] > 1:
        ax_msd.fill_between(
            time,
            msd_mean - msd_std,
            msd_mean + msd_std,
            color=color,
            alpha=0.18,
            linewidth=0
        )


# ============================================================
# Finalize MSD panel
# ============================================================
ax_msd.set_xlabel(
    r'$\mathbf{\mathit{t}}^{*}\;(\mathbf{\tau}^{*})$',
    fontsize=40,
    labelpad=5
)

ax_msd.set_ylabel(
    r'$\left\langle \Delta \mathbf{\mathit{r^*}}^{\,2}  \right\rangle'
    r'\;(\mathbf{\sigma}^{*2})$',
    fontsize=40,
    labelpad=8
)

# Automatically choose readable ticks for the actual MSD range.
# ax_msd.set_ylim(bottom=0)
ax_msd.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
ax_msd.ticklabel_format(
    axis='y',
    style='sci',
    scilimits=(0, 0)
)

ax_msd.legend(
    frameon=False,
    borderpad=0.1,
    labelspacing=0.2,
    columnspacing=0.2,
    borderaxespad=0.4,
    handletextpad=0.4,
    fontsize='22',
    loc='upper left',
    handlelength=0.7
)

ax_msd.xaxis.set_minor_locator(AutoMinorLocator())
ax_msd.yaxis.set_minor_locator(AutoMinorLocator())
ax_msd.tick_params(which='major', length=5, pad=4)
ax_msd.tick_params(which='minor', length=3)


# ============================================================
# Mobility panel
# ============================================================
charge = np.array(charge)
mobility = np.array(mobility)
mobility_std = np.array(mobility_std)

ax_mobility.errorbar(
    charge,
    mobility,
    yerr=mobility_std,
    fmt='-o',
    color='blue',
    lw=2.0,
    markersize=9,
    capsize=4,
    mfc='none'
)

ax_mobility.set_xlabel(
    r'$\mathbf{\mathit{q}}_{\mathrm{M}}^{*}$',
    fontsize=40,
    labelpad=5
)

ax_mobility.set_ylabel(
    r'$\mathbf{\mu}^{*}\;'
    r'(\mathbf{\sigma}^{*2}/(\mathbf{\epsilon}^{*}\mathbf{\tau}^{*}))$',
    fontsize=40,
    labelpad=8
)

ax_mobility.set_xticks(np.arange(8, 25, 4))

ax_mobility.xaxis.set_minor_locator(AutoMinorLocator())
ax_mobility.yaxis.set_minor_locator(AutoMinorLocator())
ax_mobility.tick_params(which='major', length=5, pad=4)
ax_mobility.tick_params(which='minor', length=3)


# ============================================================
# Save mobility data
# ============================================================
output_txt = os.path.join(
    output_dir,
    f"mobility_{number}_3replicas.txt"
)

if len(charge) > 0:
    np.savetxt(
        output_txt,
        np.column_stack([charge, mobility, mobility_std]),
        header="q mobility_mean mobility_std"
    )
    print("Saved mobility data to:", output_txt)
print(f"Mobility calculated using kBT* = {kBT_star:.6g}")


# ============================================================
# Layout and save
# ============================================================
plt.tight_layout()

pp.savefig(fig, bbox_inches='tight')
pp.close()

print("Saved figure to:", output_pdf)

plt.show()