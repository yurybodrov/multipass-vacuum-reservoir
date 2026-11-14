# Multipass Neuromorphic Plasma-Optical Cavity Simulator (3D SSF-BPM & Finite-Difference PDEs)

Official Python simulation suite for the manuscript:  
**"Spatiotemporal State Reconstruction and Regularized Inverse Problem Solving in a Multipass Neuromorphic Plasma-Optical Cavity via Off-Axis Digital Holography"**  
*Author:* Yury Bodrov (Chemcreator Limited, London, United Kingdom)

---

## Overview
This repository contains the first-principles numerical solvers combining **3D Split-Step Fourier Beam Propagation (SSF-BPM)** across a 400-pass degenerate self-imaging Herriott optical cavity (*L*<sub>eff</sub> = 40 m) with explicit **finite-difference partial differential equation (PDE)** solvers (2nd-order Heun/RK2 and Courant-stable Upwind advection-diffusion-reaction schemes) and a **12-bit monochrome CMOS off-axis digital holography** readout model with Poisson photon shot noise.

## Repository Structure
* `physics_config.py` — Physical constants (SI) and hardware parameters of the magnetically confined plasma cavity (*P* = 10<sup>−3</sup> Torr, *N*<sub>passes</sub> = 400, 12-bit CMOS sensor).
* `chamber_pde_solver.py` — 2D/3D finite-difference Laplacian and RK2 relaxation-diffusion PDE integrator with CFL sub-stepping.
* `optical_bpm_holo.py` — Paraxial Helmholtz 3D Split-Step Fourier BPM propagator and off-axis Mach–Zehnder CMOS hologram quantizer.
* `ridge_readout.py` — Tikhonov-regularized Ridge regression readout and fidelity metrics.
* `exp1_wavefront_conjugation.py` — **Experiment 1:** Optical phase conjugation of 8 × 8 complex wavefronts (**99.65%** test fidelity vs. **1.88%** without reference wave).
* `exp2_particle_wake_pde.py` — **Experiment 2:** Step-by-step PDE ionizing particle wake recovery (**86.71%** spatial fidelity, **75.18%** kinematic track *R*²).
* `exp3_chemical_kinetics_pde.py` — **Experiment 3:** Single-shot inversion of 5 coupled nonlinear bimolecular radical reaction-diffusion-enthalpy PDEs (**98.56%** for A<sup>•</sup> + B<sup>•</sup>, **99.56%** for [AB]<sup>‡</sup>).
* `exp4_qubit_lindblad_pde.py` — **Experiment 4:** Open-system Lindblad quantum qubit decoherence inversion under quantum Fluctuation-Dissipation bath noise (> **90%** fidelity up to Δ*t* / τ = 1.07).
* `exp5_continuous_xor_memory.py` — **Experiment 5:** Courant-stable Upwind traveling plasma wave simulation for emergent delayed XOR logic (**100.00%** accuracy) and multi-step fading memory (MC = **5.04 bits**).
* `run_all_physical_experiments.py` — Master script that sequentially executes all 5 physical benchmarks and generates all manuscript figures (`real_exp1`–`real_exp5`).

## Quick Start
Requires Python 3.9+ with `numpy` and `matplotlib`:

```bash
pip install numpy matplotlib
python run_all_physical_experiments.py
