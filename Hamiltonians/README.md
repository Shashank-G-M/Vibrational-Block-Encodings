# Hamiltonian Data

This directory contains the vibrational Hamiltonians used with the code and examples in this repository.

## Included Hamiltonians

The following smaller datasets are included directly in the GitHub repository:

- `Trotter_Hamiltonians/`: Hamiltonians for H<sub>2</sub>S, CO<sub>2</sub>, CH<sub>2</sub>O, and CH<sub>4</sub>.
- `Energy_vs_tensor_error_Hamiltonians/`: Hamiltonians for the CH<sub>4</sub> and C<sub>2</sub>H<sub>4</sub> tensor-error analysis.

## PAH Hamiltonians

The larger polycyclic aromatic hydrocarbon (PAH) Hamiltonians are distributed separately through Zenodo:

- **Archive:** `PAH_Hamiltonians.zip`
- **DOI:** [10.5281/zenodo.23305214](https://doi.org/10.5281/zenodo.23305214)

The archive contains the seven two-mode-coupling PAH Hamiltonians and the PAH1 and PAH2 three-mode-coupling Hamiltonians used in the paper. It also includes precomputed CP3-initialized tensor hypercontraction (THC) parameters used as a precursor to the Triangular LCU decomposition of the three-mode-coupling Hamiltonians.

Download and extract the archive inside this `Hamiltonians` directory while preserving its internal structure. The resulting layout should begin with:

```text
Hamiltonians/
├── Energy_vs_tensor_error_Hamiltonians/
├── PAH_Hamiltonians/
└── Trotter_Hamiltonians/
```

In the filenames, `M` denotes the number of VSCF modals per mode and `MC` denotes the maximum mode-coupling order.
