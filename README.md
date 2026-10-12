# Vibrational Block Encodings

This repository contains the code and numerical results accompanying the paper:

> **Efficient Block Encodings of Vibrational Hamiltonians**

The project implements and analyzes several linear-combination-of-unitaries representations of vibrational Hamiltonians:

- Triangular LCU ([previous work](https://doi.org/10.1021/acs.jctc.5c01419))
- Pauli and BLISS-Pauli LCUs (this work)
- Tensor hypercontraction (THC) and BLISS-THC LCUs (this work)

It also includes tensor-factorization routines, quantum phase estimation resource estimates, eigensolvers, example notebooks, and scripts used to generate the figures in the paper.

## Repository structure

```text
Vibrational-Block-Encodings/
├── BLISS-Pauli_notebook.ipynb
├── BLISS-THC_notebook.ipynb
├── Triangular_notebook.ipynb
├── Hamiltonians/
├── Results/
└── vib_frags/
```

### Example notebooks

- `BLISS-Pauli_notebook.ipynb` demonstrates Pauli and BLISS-Pauli LCU construction and resource estimation.
- `BLISS-THC_notebook.ipynb` demonstrates CP3 initialization, THC factorization, BLISS-THC optimization, and resource estimation.
- `Triangular_notebook.ipynb` demonstrates construction and resource estimation for the Triangular LCU.

### Source modules

The `vib_frags` package provides utilities for:

- Vibrational tensor manipulation
- Pauli and THC LCU construction
- BLISS transformations
- CP3 and symmetric CP3 tensor decomposition
- Toffoli and logical-qubit resource estimation
- Sparse and JAX-based eigenvalue calculations

### Hamiltonians

Small Hamiltonian files are included in this repository so that the basic examples can be run directly. The larger PAH Hamiltonians are distributed separately. Their contents, download instructions, and expected directory structure are documented in `Hamiltonians/README.md`.

### Results

The `Results` directory contains the numerical resource estimates and spectral-analysis data used in the paper. The scripts under `Results/Plots` reproduce the corresponding figures.
