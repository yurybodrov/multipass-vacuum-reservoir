"""
exp1_wavefront_conjugation.py
BPM wave propagation phase-conjugation under high-Q multipass optical setup.
"""
import numpy as np
import matplotlib.pyplot as plt
from physics_config import ChamberConfig
from chamber_pde_solver import ChamberPDESolver
from optical_bpm_holo import OpticalBPMHolography
from ridge_readout import RidgeReadout, calc_wave_fidelity

def upsample_modes_to_grid(modes_8x8: np.ndarray, Nx: int, Ny: int) -> np.ndarray:
    grid_8x8 = modes_8x8.reshape(8, 8)
    return np.kron(grid_8x8, np.ones((Nx // 8, Ny // 8)))

def run_experiment_1():
    print("--- Running Experiment 1: Wave-Optical BPM Wavefront Phase Conjugation ---")
    cfg = ChamberConfig()
    pde = ChamberPDESolver(cfg)
    optics = OpticalBPMHolography(cfg)
    rng = np.random.RandomState(42)

    N_samples = 1500
    N_modes = 64
    split = int(0.8 * N_samples)

    X_cmos_with_ref = np.zeros((cfg.Nx * cfg.Ny, N_samples))
    X_cmos_no_ref = np.zeros((cfg.Nx * cfg.Ny, N_samples))
    Y_target = np.zeros((N_modes, N_samples), dtype=complex)
    dn_steady = np.zeros((cfg.Nx, cfg.Ny))

    for i in range(N_samples):
        modes_in = (rng.randn(N_modes) + 1j * rng.randn(N_modes)) / np.sqrt(2.0)
        Y_target[:, i] = np.conj(modes_in)
        E_in_2d = upsample_modes_to_grid(modes_in, cfg.Nx, cfg.Ny)

        dn_current = pde.step_reservoir_pde(
            dn_steady, np.abs(E_in_2d)**2 * 0.05, np.zeros((cfg.Nx, cfg.Ny)), cfg.exposure_time, rng
        )
        dn_3d = pde.build_3d_index_volume(dn_current)
        E_out = optics.propagate_bpm(E_in_2d, dn_3d)

        X_cmos_with_ref[:, i] = optics.record_cmos_hologram(E_out, use_reference=True, rng=rng)
        X_cmos_no_ref[:, i] = optics.record_cmos_hologram(E_out, use_reference=False, rng=rng)

    model_ref = RidgeReadout(lambd=1e-2).fit(X_cmos_with_ref[:, :split], Y_target[:, :split])
    Y_pred_ref = model_ref.predict(X_cmos_with_ref[:, split:])
    fid_ref = np.mean([calc_wave_fidelity(Y_target[:, split+k], Y_pred_ref[:, k]) for k in range(N_samples - split)]) * 100

    model_noref = RidgeReadout(lambd=1e-1).fit(X_cmos_no_ref[:, :split], Y_target[:, :split])
    Y_pred_noref = model_noref.predict(X_cmos_no_ref[:, split:])
    fid_noref = np.mean([calc_wave_fidelity(Y_target[:, split+k], Y_pred_noref[:, k]) for k in range(N_samples - split)]) * 100

    print(f"No-Reference Intensity-Only Fidelity: {fid_noref:.2f}%")
    print(f"Off-Axis Holographic Cavity Fidelity:  {fid_ref:.2f}%\n")

    idx = 5
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    ax[0].imshow(np.abs(Y_target[:, split+idx]).reshape(8, 8), cmap='inferno')
    ax[0].set_title("Target: Conjugated Wavefront", fontweight='bold')
    ax[0].axis('off')
    ax[1].imshow(X_cmos_with_ref[:, split+idx].reshape(cfg.Nx, cfg.Ny), cmap='viridis')
    ax[1].set_title("12-bit CMOS Cavity Hologram", fontweight='bold')
    ax[1].axis('off')
    ax[2].imshow(np.abs(Y_pred_ref[:, idx]).reshape(8, 8), cmap='inferno')
    ax[2].set_title(f"Reconstructed ({fid_ref:.1f}%)", fontweight='bold')
    ax[2].axis('off')
    plt.tight_layout()
    plt.savefig("real_exp1_wavefront.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    run_experiment_1()