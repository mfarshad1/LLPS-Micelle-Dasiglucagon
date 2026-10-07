#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib import rc
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator

# =========================================================
# PLOT SETTINGS: compatible with other manuscript figures
# =========================================================
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

# =========================================================
# USER SETTINGS
# =========================================================
number = "4mer_-6"

base_dir = "/groups/jwhitme1/Data-MF-01/LLPS/llps-data/biphasic/dg-1-7_2/4mer_-6"
output_dir = "/groups/jwhitme1/Data-MF-01/LLPS/"

replicas = [1, 2, 3]

# Same replicated charge window used in the phase-style analysis.
# Change this list if needed.
charges_to_analyze = [0, 4, 8, 12, 16, 20, 24]

number_bins = 1000
dg_beads_per_mol = 4

# analyze only the last N frames from eq_T.dump
n_last_frames = 50

# association criterion:
# a DG molecule is "bound" if at least this many beads are within r_cut of any micelle
min_beads_for_bound = 1

# if True, use RDF-derived cutoff; otherwise use manual_r_cut
use_rdf_cutoff = False
manual_r_cut = 6.0

# Fixed spatial regions, identical to the phase-diagram code:
# outer region:   s < 1.5/6 or s >= 4.5/6
# central region: 2.0/6 <= s < 4.0/6
# transition regions between them are excluded from the four plotted bars.
outer_left_end = 1.5 / 6.0
central_start = 2.0 / 6.0
central_end = 4.0 / 6.0
outer_right_start = 4.5 / 6.0

# output filenames
out_pdf = os.path.join(
    output_dir,
    f"dg_bound_unbound_analysis_{number}_fixed_regions_3replicas.pdf"
)
out_txt = os.path.join(
    output_dir,
    f"dg_bound_unbound_analysis_{number}_fixed_regions_3replicas.txt"
)


# =========================================================
# BASIC UTILITIES
# =========================================================
def minimum_image_displacement(vec, box):
    return vec - box * np.round(vec / box)


def wrap_molecule_relative(coords, box):
    ref = coords[0].copy()
    wrapped = [ref]

    for i in range(1, len(coords)):
        disp = coords[i] - ref
        disp = minimum_image_displacement(disp, box)
        wrapped.append(ref + disp)

    return np.array(wrapped, dtype=float)


def shift_z_to_0_L(z, zlo, zhi):
    Lz = zhi - zlo
    z_shift = z - zlo
    z_shift = z_shift % Lz
    return z_shift


def replica_std(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return np.std(vals, ddof=1)


# =========================================================
# READ RDF AND CHOOSE M-DG CUTOFF
# =========================================================
def get_rdf_cutoff(q_formatted, rep):
    repdir = os.path.join(base_dir, f"q-{q_formatted}", f"rep{rep}")

    rdf_file = os.path.join(
        repdir,
        f"out.q-{q_formatted}.rdf"
    )

    if not os.path.exists(rdf_file):
        raise FileNotFoundError(f"RDF file not found: {rdf_file}")

    all_rdf = []

    with open(rdf_file, "r") as f:
        for line in f:
            s = line.strip()

            if not s or s.startswith("#"):
                continue

            try:
                vals = [float(v) for v in s.split()]
            except ValueError:
                continue

            if len(vals) < 9:
                continue

            if vals[0] > number_bins:
                continue

            all_rdf.append(vals)

    block_size = number_bins
    num_blocks = len(all_rdf) // block_size

    blocks = []

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block = all_rdf[start:end]

        if len(block) == number_bins:
            blocks.append(block)

    if not blocks:
        raise ValueError(f"No RDF blocks found for q={q_formatted}, rep={rep}")

    ave_block = np.zeros((number_bins, 10), dtype=float)

    for block in blocks:
        ave_block += np.array(block, dtype=float)

    ave_block /= len(blocks)

    r = np.array([row[1] for row in ave_block], dtype=float)
    g_mdg = np.array([row[4] for row in ave_block], dtype=float)

    peak_mask = (r > 2.0) & (r < 12.0)
    r_peak = r[peak_mask]
    g_peak = g_mdg[peak_mask]

    peaks, _ = find_peaks(g_peak)

    if len(peaks) == 0:
        raise ValueError(f"Could not find M-DG RDF peak for q={q_formatted}, rep={rep}")

    first_peak_idx_local = peaks[0]
    first_peak_r = r_peak[first_peak_idx_local]

    min_mask = (r > first_peak_r) & (r < 15.0)
    r_min = r[min_mask]
    g_min = g_mdg[min_mask]

    minima, _ = find_peaks(-g_min)

    if len(minima) == 0:
        r_cut = manual_r_cut
    else:
        first_min_idx_local = minima[0]
        r_cut = r_min[first_min_idx_local]

    return {
        "rdf_file": rdf_file,
        "r": r,
        "g_mdg": g_mdg,
        "first_peak_r": first_peak_r,
        "r_cut": r_cut
    }


# =========================================================
# READ ALL FRAMES FROM DUMP
# =========================================================
def read_all_frames_lammps_dump(dump_file):
    if not os.path.exists(dump_file):
        raise FileNotFoundError(f"Dump file not found: {dump_file}")

    frames = []

    with open(dump_file, "r") as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            timestep = int(float(lines[i + 1].strip()))

            if not lines[i + 2].startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Unexpected NUMBER OF ATOMS section in {dump_file}")

            n_atoms = int(float(lines[i + 3].strip()))

            if not lines[i + 4].startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Unexpected BOX BOUNDS section in {dump_file}")

            xlo, xhi = map(float, lines[i + 5].split()[:2])
            ylo, yhi = map(float, lines[i + 6].split()[:2])
            zlo, zhi = map(float, lines[i + 7].split()[:2])

            if not lines[i + 8].startswith("ITEM: ATOMS"):
                raise ValueError(f"Unexpected ATOMS section in {dump_file}")

            cols = lines[i + 8].split()[2:]
            col_index = {c: j for j, c in enumerate(cols)}

            if "id" not in col_index or "type" not in col_index:
                raise ValueError(f"id/type not found in ATOMS header: {lines[i + 8]}")

            xcol = "x" if "x" in col_index else "xu"
            ycol = "y" if "y" in col_index else "yu"
            zcol = "z" if "z" in col_index else "zu"

            if xcol not in col_index or ycol not in col_index or zcol not in col_index:
                raise ValueError(f"x/y/z or xu/yu/zu not found in ATOMS header: {lines[i + 8]}")

            atom_lines = lines[i + 9:i + 9 + n_atoms]

            atoms = []

            for line in atom_lines:
                s = line.split()

                atom_id = int(float(s[col_index["id"]]))
                atom_type = int(float(s[col_index["type"]]))

                x = float(s[col_index[xcol]])
                y = float(s[col_index[ycol]])
                z = float(s[col_index[zcol]])

                atoms.append([atom_id, atom_type, x, y, z])

            frames.append({
                "timestep": timestep,
                "bounds": ((xlo, xhi), (ylo, yhi), (zlo, zhi)),
                "atoms": atoms
            })

            i = i + 9 + n_atoms

        else:
            i += 1

    if not frames:
        raise ValueError(f"No frames found in {dump_file}")

    return frames


# =========================================================
# GROUP DG BEADS INTO MOLECULES
# =========================================================
def group_dg_molecules_by_sorted_ids(dg_atoms, beads_per_mol=4):
    dg_atoms_sorted = sorted(dg_atoms, key=lambda a: a[0])

    if len(dg_atoms_sorted) % beads_per_mol != 0:
        raise ValueError("Number of DG beads is not divisible by 4.")

    dg_mols = []

    for i in range(0, len(dg_atoms_sorted), beads_per_mol):
        mol_atoms = dg_atoms_sorted[i:i + beads_per_mol]
        coords = np.array([[a[2], a[3], a[4]] for a in mol_atoms], dtype=float)
        ids = [a[0] for a in mol_atoms]
        dg_mols.append({"ids": ids, "coords": coords})

    return dg_mols


# =========================================================
# ANALYZE ONE FRAME
# =========================================================
def analyze_frame(frame, r_cut, min_beads_for_bound=1):
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = frame["bounds"]
    box = np.array([xhi - xlo, yhi - ylo, zhi - zlo], dtype=float)

    atoms = frame["atoms"]

    micelles = [a for a in atoms if a[1] == 1]
    dg_atoms = [a for a in atoms if a[1] == 2]

    if len(micelles) == 0:
        raise ValueError("No micelle atoms of type 1 found in frame.")

    if len(dg_atoms) == 0:
        raise ValueError("No DG atoms of type 2 found in frame.")

    mic_coords = np.array([[a[2], a[3], a[4]] for a in micelles], dtype=float)

    dg_mols = group_dg_molecules_by_sorted_ids(
        dg_atoms,
        beads_per_mol=dg_beads_per_mol
    )

    counts = {
        "bound_dense": 0,
        "unbound_dense": 0,
        "bound_dilute": 0,
        "unbound_dilute": 0,
        "bound_transition": 0,
        "unbound_transition": 0,
        "total_dg": 0
    }

    for mol in dg_mols:
        coords = wrap_molecule_relative(mol["coords"], box)
        com = coords.mean(axis=0)

        z_shift = shift_z_to_0_L(com[2], zlo, zhi)

        n_close_beads = 0

        for bead in coords:
            disp = mic_coords - bead
            disp = minimum_image_displacement(disp, box)
            dists = np.linalg.norm(disp, axis=1)

            if np.any(dists < r_cut):
                n_close_beads += 1

        bound = (n_close_beads >= min_beads_for_bound)

        # Same fixed regions used in the phase-diagram code.
        s = z_shift / box[2]

        in_dilute = (s < outer_left_end) or (s >= outer_right_start)
        in_dense = (s >= central_start) and (s < central_end)

        if in_dense:
            if bound:
                counts["bound_dense"] += 1
            else:
                counts["unbound_dense"] += 1

        elif in_dilute:
            if bound:
                counts["bound_dilute"] += 1
            else:
                counts["unbound_dilute"] += 1

        else:
            # Same transition regions that are ignored in the phase diagram.
            if bound:
                counts["bound_transition"] += 1
            else:
                counts["unbound_transition"] += 1

        counts["total_dg"] += 1

    return counts


# =========================================================
# ANALYZE ONE CHARGE AND ONE REPLICA
# =========================================================
def analyze_charge_replica(q, rep):
    q_formatted = f"{q:.2f}"
    repdir = os.path.join(base_dir, f"q-{q_formatted}", f"rep{rep}")

    if use_rdf_cutoff:
        rdf_info = get_rdf_cutoff(q_formatted, rep)
        r_cut = rdf_info["r_cut"]
    else:
        rdf_info = None
        r_cut = manual_r_cut

    dump_file = os.path.join(repdir, "eq_T.dump")

    frames = read_all_frames_lammps_dump(dump_file)

    frames_to_use = frames[-n_last_frames:] if len(frames) >= n_last_frames else frames

    sum_counts = {
        "bound_dense": 0,
        "unbound_dense": 0,
        "bound_dilute": 0,
        "unbound_dilute": 0,
        "bound_transition": 0,
        "unbound_transition": 0,
        "total_dg": 0
    }

    per_frame_counts = []

    for frame in frames_to_use:
        counts = analyze_frame(
            frame=frame,
            r_cut=r_cut,
            min_beads_for_bound=min_beads_for_bound
        )

        per_frame_counts.append((frame["timestep"], counts))

        for key in sum_counts:
            sum_counts[key] += counts[key]

    n_frames_used = len(frames_to_use)
    avg_counts = {k: sum_counts[k] / n_frames_used for k in sum_counts}

    (_, _), (_, _), (zlo, zhi) = frames_to_use[0]["bounds"]
    Lz = zhi - zlo

    region_info = {
        "Lz": Lz,
        "dense_lo": central_start * Lz,
        "dense_hi": central_end * Lz,
        "dilute_left_hi": outer_left_end * Lz,
        "dilute_right_lo": outer_right_start * Lz
    }

    return {
        "q": q,
        "rep": rep,
        "q_formatted": q_formatted,
        "region_info": region_info,
        "rdf_info": rdf_info,
        "r_cut": r_cut,
        "n_frames_used": n_frames_used,
        "avg_counts": avg_counts,
        "per_frame_counts": per_frame_counts,
        "dump_file": dump_file
    }


# =========================================================
# ANALYZE ONE CHARGE OVER THREE REPLICAS
# =========================================================
def analyze_charge(q):
    q_formatted = f"{q:.2f}"

    rep_results = []

    for rep in replicas:
        try:
            res = analyze_charge_replica(q, rep)

        except Exception as e:
            print(f"Skipping q={q_formatted}, rep={rep}: {e}")
            continue

        rep_results.append(res)

        region = res["region_info"]

        print(f"\nq*_M = {q_formatted}, rep = {rep}")
        print(f"  dump file:    {res['dump_file']}")
        print(f"  Lz: {region['Lz']:.2f}")
        print(
            f"  central region shifted z: "
            f"{region['dense_lo']:.2f} to {region['dense_hi']:.2f}"
        )
        print(
            f"  outer regions shifted z: 0.00 to "
            f"{region['dilute_left_hi']:.2f}, and "
            f"{region['dilute_right_lo']:.2f} to {region['Lz']:.2f}"
        )
        print(f"  r_cut (M-DG): {res['r_cut']:.3f}")
        print(f"  frames used: {res['n_frames_used']}")
        print(f"  avg bound central      = {res['avg_counts']['bound_dense']:.2f}")
        print(f"  avg unbound central    = {res['avg_counts']['unbound_dense']:.2f}")
        print(f"  avg bound outer        = {res['avg_counts']['bound_dilute']:.2f}")
        print(f"  avg unbound outer      = {res['avg_counts']['unbound_dilute']:.2f}")
        print(f"  avg bound transition   = {res['avg_counts']['bound_transition']:.2f}")
        print(f"  avg unbound transition = {res['avg_counts']['unbound_transition']:.2f}")
        print(f"  avg total DG           = {res['avg_counts']['total_dg']:.2f}")

    if len(rep_results) == 0:
        return None

    keys = [
        "bound_dense",
        "unbound_dense",
        "bound_dilute",
        "unbound_dilute",
        "bound_transition",
        "unbound_transition",
        "total_dg"
    ]

    avg_counts = {}
    std_counts = {}

    for key in keys:
        vals = np.array([r["avg_counts"][key] for r in rep_results], dtype=float)
        avg_counts[key] = np.mean(vals)
        std_counts[key] = replica_std(vals)

    r_cut_vals = np.array([r["r_cut"] for r in rep_results], dtype=float)

    return {
        "q": q,
        "q_formatted": q_formatted,
        "rep_results": rep_results,
        "n_replicas": len(rep_results),
        "avg_counts": avg_counts,
        "std_counts": std_counts,
        "r_cut_mean": np.mean(r_cut_vals),
        "r_cut_std": replica_std(r_cut_vals)
    }


# =========================================================
# MAIN
# =========================================================
results = []

for q in charges_to_analyze:
    print("\n" + "=" * 72)
    print(f"Processing q*_M = {q:.2f}")
    print("=" * 72)

    res = analyze_charge(q)

    if res is None:
        print(f"Skipping q*_M = {q:.2f}: no usable replicas")
        continue

    results.append(res)

    print(f"\nq*_M = {res['q_formatted']} replica average")
    print(f"  replicas used: {res['n_replicas']}")
    print(f"  r_cut mean/std: {res['r_cut_mean']:.3f} / {res['r_cut_std']:.3f}")
    print(f"  mean bound central      = {res['avg_counts']['bound_dense']:.2f} +/- {res['std_counts']['bound_dense']:.2f}")
    print(f"  mean unbound central    = {res['avg_counts']['unbound_dense']:.2f} +/- {res['std_counts']['unbound_dense']:.2f}")
    print(f"  mean bound outer        = {res['avg_counts']['bound_dilute']:.2f} +/- {res['std_counts']['bound_dilute']:.2f}")
    print(f"  mean unbound outer      = {res['avg_counts']['unbound_dilute']:.2f} +/- {res['std_counts']['unbound_dilute']:.2f}")
    print(f"  mean bound transition   = {res['avg_counts']['bound_transition']:.2f} +/- {res['std_counts']['bound_transition']:.2f}")
    print(f"  mean unbound transition = {res['avg_counts']['unbound_transition']:.2f} +/- {res['std_counts']['unbound_transition']:.2f}")
    print(f"  mean total DG           = {res['avg_counts']['total_dg']:.2f} +/- {res['std_counts']['total_dg']:.2f}")


if len(results) == 0:
    raise RuntimeError("No data were processed. Check base_dir, charges_to_analyze, and replica folders.")


# =========================================================
# SAVE TXT SUMMARY
# =========================================================
summary_rows = []

for res in results:
    summary_rows.append([
        res["q"],
        res["n_replicas"],

        res["avg_counts"]["bound_dense"],
        res["std_counts"]["bound_dense"],

        res["avg_counts"]["unbound_dense"],
        res["std_counts"]["unbound_dense"],

        res["avg_counts"]["bound_dilute"],
        res["std_counts"]["bound_dilute"],

        res["avg_counts"]["unbound_dilute"],
        res["std_counts"]["unbound_dilute"],

        res["avg_counts"]["bound_transition"],
        res["std_counts"]["bound_transition"],

        res["avg_counts"]["unbound_transition"],
        res["std_counts"]["unbound_transition"],

        res["avg_counts"]["total_dg"],
        res["std_counts"]["total_dg"],

        res["r_cut_mean"],
        res["r_cut_std"],
    ])

np.savetxt(
    out_txt,
    np.array(summary_rows, dtype=float),
    header=(
        "q n_replicas "
        "bound_central_mean bound_central_std "
        "unbound_central_mean unbound_central_std "
        "bound_outer_mean bound_outer_std "
        "unbound_outer_mean unbound_outer_std "
        "bound_transition_mean bound_transition_std "
        "unbound_transition_mean unbound_transition_std "
        "total_dg_mean total_dg_std "
        "r_cut_mean r_cut_std"
    )
)

print("\nSaved summary to:", out_txt)


# =========================================================
# PLOT
# =========================================================
charges = [res["q"] for res in results]

bound_dense = [res["avg_counts"]["bound_dense"] for res in results]
unbound_dense = [res["avg_counts"]["unbound_dense"] for res in results]
bound_dilute = [res["avg_counts"]["bound_dilute"] for res in results]
unbound_dilute = [res["avg_counts"]["unbound_dilute"] for res in results]

bound_dense_std = [res["std_counts"]["bound_dense"] for res in results]
unbound_dense_std = [res["std_counts"]["unbound_dense"] for res in results]
bound_dilute_std = [res["std_counts"]["bound_dilute"] for res in results]
unbound_dilute_std = [res["std_counts"]["unbound_dilute"] for res in results]

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

ax.set_ylim(None,700)
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
plt.savefig(out_pdf, bbox_inches='tight', dpi=300)

print("Saved figure to:", out_pdf)

plt.show()