from pathlib import Path
import argparse
import numpy as np
import pandas as pd


NUMBER_BINS = 1000
REPLICAS = [1, 2, 3]
CHARGES = [8, 16, 24, 32]


def read_numeric_rows(path, min_cols):
    rows = []

    with open(path, "r") as f:
        for line in f:
            s = line.strip()

            if not s or s.startswith("#"):
                continue

            try:
                values = [float(v) for v in s.split()]
            except ValueError:
                continue

            if len(values) < min_cols:
                continue

            # Same condition as manuscript analysis code
            if values[0] > NUMBER_BINS:
                continue

            rows.append(values)

    return rows


# ============================================================
# DENSITY
# Same logic as density+rdf-reps-v03.py
# ============================================================

def process_density_replica(path):
    rows = read_numeric_rows(path, 4)

    nblocks = len(rows) // NUMBER_BINS

    blocks = []

    for i in range(nblocks):

        # Exact manuscript rule:
        # keep only blocks i > 50
        if i > 50:
            start = i * NUMBER_BINS
            end = start + NUMBER_BINS

            block = rows[start:end]

            if len(block) == NUMBER_BINS:
                blocks.append(np.asarray(block, dtype=float))

    if not blocks:
        raise RuntimeError(f"No usable density blocks in {path}")

    ave_block = np.zeros_like(blocks[0], dtype=float)

    for block in blocks:
        ave_block += block

    ave_block /= len(blocks)

    x = ave_block[:, 1]
    density = ave_block[:, 3]

    # Exact shift used by manuscript script
    x = x - np.min(x)

    return x, density, len(blocks)


def process_density_charge(source_root, q):
    qf = f"{q:.2f}"

    results = []

    for rep in REPLICAS:
        path = (
            source_root /
            f"q-{qf}" /
            f"rep{rep}" /
            f"density_polymer.q-{qf}.data"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        x, density, nblocks = process_density_replica(path)

        print(
            f"Density q={qf} rep={rep}: "
            f"{nblocks} blocks used"
        )

        results.append((rep, x, density, nblocks))

    x_ref = results[0][1]

    profiles = []

    for rep, x, density, nblocks in results:

        if len(x) != len(x_ref) or not np.allclose(x, x_ref):
            density = np.interp(x_ref, x, density)

        profiles.append(density)

    profiles = np.asarray(profiles)

    mean = np.mean(profiles, axis=0)

    # Exact np.std behavior of manuscript script: ddof=0
    std = np.std(profiles, axis=0)

    out = pd.DataFrame({
        "Charge": q,
        "x": x_ref,
        "mean_density": mean,
        "std_density": std,
        "n_replicas": len(results),
    })

    metadata = []

    for rep, x, density, nblocks in results:
        metadata.append({
            "Charge": q,
            "Replica": rep,
            "Blocks_used": nblocks,
        })

    return out, metadata


# ============================================================
# RDF
# Same logic as density+rdf-reps-v03.py
# ============================================================

def process_rdf_replica(path):
    rows = read_numeric_rows(path, 9)

    nblocks = len(rows) // NUMBER_BINS

    blocks = []

    for i in range(nblocks):
        start = i * NUMBER_BINS
        end = start + NUMBER_BINS

        block = rows[start:end]

        if len(block) == NUMBER_BINS:
            blocks.append(np.asarray(block, dtype=float))

    if not blocks:
        raise RuntimeError(f"No usable RDF blocks in {path}")

    ave_block = np.zeros_like(blocks[0], dtype=float)

    for block in blocks:
        ave_block += block

    ave_block /= len(blocks)

    x = ave_block[:, 1]

    # Exact columns used by manuscript script
    rdf_1_1 = ave_block[:, 2]
    rdf_1_2 = ave_block[:, 4]
    rdf_2_2 = ave_block[:, 8]

    return x, rdf_1_1, rdf_1_2, rdf_2_2, len(blocks)


def process_rdf_charge(source_root, q):
    qf = f"{q:.2f}"

    results = []

    for rep in REPLICAS:
        path = (
            source_root /
            f"q-{qf}" /
            f"rep{rep}" /
            f"out.q-{qf}.rdf"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        result = process_rdf_replica(path)

        x, r11, r12, r22, nblocks = result

        print(
            f"RDF q={qf} rep={rep}: "
            f"{nblocks} blocks used"
        )

        results.append(
            (rep, x, r11, r12, r22, nblocks)
        )

    x_ref = results[0][1]

    r11_reps = []
    r12_reps = []
    r22_reps = []

    metadata = []

    for rep, x, r11, r12, r22, nblocks in results:

        if len(x) != len(x_ref) or not np.allclose(x, x_ref):
            r11 = np.interp(x_ref, x, r11)
            r12 = np.interp(x_ref, x, r12)
            r22 = np.interp(x_ref, x, r22)

        r11_reps.append(r11)
        r12_reps.append(r12)
        r22_reps.append(r22)

        metadata.append({
            "Charge": q,
            "Replica": rep,
            "Blocks_used": nblocks,
        })

    r11_reps = np.asarray(r11_reps)
    r12_reps = np.asarray(r12_reps)
    r22_reps = np.asarray(r22_reps)

    out = pd.DataFrame({
        "Charge": q,
        "r": x_ref,

        "rdf_1_1_mean": np.mean(r11_reps, axis=0),
        "rdf_1_1_std": np.std(r11_reps, axis=0),

        "rdf_1_2_mean": np.mean(r12_reps, axis=0),
        "rdf_1_2_std": np.std(r12_reps, axis=0),

        "rdf_2_2_mean": np.mean(r22_reps, axis=0),
        "rdf_2_2_std": np.std(r22_reps, axis=0),

        "n_replicas": len(results),
    })

    return out, metadata


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-root",
        required=True,
        help="Original 4mer_-6 simulation directory"
    )

    parser.add_argument(
        "--repo",
        default=".",
        help="Repository root"
    )

    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    repo = Path(args.repo).resolve()

    density_frames = []
    density_metadata = []

    rdf_frames = []
    rdf_metadata = []

    for q in CHARGES:

        print()
        print("=" * 60)
        print(f"Processing q = {q}")
        print("=" * 60)

        d, dm = process_density_charge(source_root, q)

        density_frames.append(d)
        density_metadata.extend(dm)

        r, rm = process_rdf_charge(source_root, q)

        rdf_frames.append(r)
        rdf_metadata.extend(rm)

    density = pd.concat(
        density_frames,
        ignore_index=True
    )

    rdf = pd.concat(
        rdf_frames,
        ignore_index=True
    )

    density_dir = (
        repo /
        "reduced_data" /
        "density_profiles"
    )

    rdf_dir = (
        repo /
        "reduced_data" /
        "rdf"
    )

    density_dir.mkdir(parents=True, exist_ok=True)
    rdf_dir.mkdir(parents=True, exist_ok=True)

    density_file = (
        density_dir /
        "figure6_dg_density_q8_16_24_32.csv"
    )

    density_meta_file = (
        density_dir /
        "figure6_dg_density_q8_16_24_32_metadata.csv"
    )

    rdf_file = (
        rdf_dir /
        "figure8_rdf_q8_16_24_32.csv"
    )

    rdf_meta_file = (
        rdf_dir /
        "figure8_rdf_q8_16_24_32_metadata.csv"
    )

    density.to_csv(density_file, index=False)

    pd.DataFrame(
        density_metadata
    ).to_csv(
        density_meta_file,
        index=False
    )

    rdf.to_csv(rdf_file, index=False)

    pd.DataFrame(
        rdf_metadata
    ).to_csv(
        rdf_meta_file,
        index=False
    )

    print()
    print("Saved:")
    print(density_file)
    print(density_meta_file)
    print(rdf_file)
    print(rdf_meta_file)


if __name__ == "__main__":
    main()
