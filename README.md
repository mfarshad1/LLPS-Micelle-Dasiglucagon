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
