# Charge Density Effect of Micelles on Liquid–Liquid Phase Separation

This repository contains simulation input files, analysis scripts, and reduced data supporting the manuscript:

**Charge Density Effect of Micelles on Liquid–Liquid Phase Separation**

## Repository structure

- `inputs/` – Representative LAMMPS input files and the three-replica simulation workflow.
- `analysis/charge_mapping/` – Charge and zeta-potential mapping analysis.
- `analysis/phase_behavior/` – Phase-behavior analysis used for Figure 5.
- `analysis/density_rdf/` – Density-profile and radial-distribution-function analysis used for Figures 6 and 8.
- `analysis/binding/` – Bound/unbound DG analysis used for Figure 7.
- `analysis/dynamics/` – Mean-squared-displacement and mobility analysis used for Figure 9.
- `reduced_data/` – Processed data supporting the manuscript figures.

## Full simulation data

The trajectory-level analysis scripts operate on the original LAMMPS simulation outputs, including density files, RDF files, and trajectory dumps. These full simulation datasets are not included in this repository because of their size.

Full data generated in preparing this article are available from the corresponding author upon reasonable request.

## Notes

The analysis scripts are preserved in the form used for the manuscript and therefore contain paths corresponding to the original CRC computing environment. Representative simulation inputs and reduced manuscript data are included in this repository.

## Reproducing manuscript figures

The Python-generated quantitative figures can be reproduced directly from the reduced data included in this repository.

Install the Python dependencies with `pip install -r requirements.txt`.

Then run `./run_all_plots.sh`.

The script reproduces the computational plots corresponding to Figures 3, 5, 6b, 7, 8, and 9 using only files contained in this repository. Generated PDFs are written to `reproduced_figures/`.

The script `tools/prepare_figure6_8_reduced_data.py` documents the one-time reduction procedure used to construct the compact Figure 6 and Figure 8 datasets from the original trajectory-level simulation outputs. The original large simulation outputs are not required to run `run_all_plots.sh`.
