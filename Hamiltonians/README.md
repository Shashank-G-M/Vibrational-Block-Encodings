# Hamiltonian Data

This directory contains the vibrational Hamiltonians used with the code and examples in this repository.

## Included Hamiltonians

The following smaller datasets are included directly in the GitHub repository:

- `Trotter_Hamiltonians/`: Hamiltonians for H$_2$S, CO$_2$, CH$_2$O, and CH$_4$.
- `Energy_vs_tensor_error_Hamiltonians/`: Hamiltonians for the CH$_4$ and C$_2$H$_4$ tensor-error analysis.

## PAH Hamiltonians

The larger polycyclic aromatic hydrocarbon (PAH) Hamiltonians are distributed separately through Zenodo:

- **Archive:** `PAH_Hamiltonians.zip`
- **DOI:** [10.5281/zenodo.23305214](https://doi.org/10.5281/zenodo.23305214)

The archive contains the seven two-mode-coupling PAH Hamiltonians and the PAH1 and PAH2 three-mode-coupling Hamiltonians used in the paper.

Download and extract the archive inside this `Hamiltonians` directory while preserving its internal structure. The resulting layout should begin with:

```text
Hamiltonians/
├── Energy_vs_tensor_error_Hamiltonians/
├── PAH_Hamiltonians/
└── Trotter_Hamiltonians/
```

In the filenames, `M` denotes the number of VSCF modals per mode and `MC` denotes the maximum mode-coupling order.
