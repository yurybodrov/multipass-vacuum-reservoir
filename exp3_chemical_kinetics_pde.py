"""
exp3_chemical_kinetics_pde.py
Non-linear multi-component chemical reaction kinetics solver + off-axis holography.
"""
import numpy as np
import matplotlib.pyplot as plt
from physics_config import ChamberConfig
from chamber_pde_solver import ChamberPDESolver
from optical_bpm_holo import OpticalBPMHolography
from ridge_readout import RidgeReadout, calc_wave_fidelity

def run_experiment_3():
    print("--- Running Experiment 3: Authentic Multi-Kinetics Reaction-Diffusion PDEs ---")
    cfg = ChamberConfig()
    pde = ChamberPDESolver(cfg)
    optics = OpticalBPMHolography(cfg)
    rng = np.random.RandomState(42)

    N_samples = 1000
    split = int(0.8 * N_samples)
    N_pix = cfg.Nx * cfg.Ny

    X_cmos = np.zeros((N_pix, N_samples))
    Y_init_radicals = np.zeros((N_pix, N_samples))
    Y_mid_complex = np.zeros((N_pix, N_samples))

    k1_rate = 1.8e5
    k2_rate = 4.5e4
    n_steps = 16
    dt = 3.0e-6

    E_probe = np.ones((cfg.Nx, cfg.Ny), dtype=complex)
    rad_sigma = 0.50e-3

    for i in range(N_samples):
        xc = rng.uniform(-0.45e-3, 0.45e-3)
        yc = rng.uniform(-0.45e-3, 0.45e-3)
        theta = rng.uniform(0, np.pi)
        sep = rng.uniform(0.65e-3, 0.95e-3)

        xA, yA = xc - sep * np.cos(theta), yc - sep * np.sin(theta)
        xB, yB = xc + sep * np.cos(theta), yc + sep * np.sin(theta)

        C_A = np.exp(-((pde.XX - xA)**2 + (pde.YY - yA)**2) / (2.0 * rad_sigma**2))
        C_B = np.exp(-((pde.XX - xB)**2 + (pde.YY - yB)**2) / (2.0 * rad_sigma**2))
        C_AB = np.zeros((cfg.Nx, cfg.Ny))
        C_C = np.zeros((cfg.Nx, cfg.Ny))
        T_enthalpy = np.zeros((cfg.Nx, cfg.Ny))

        Y_init_radicals[:, i] = (C_A + C_B).flatten()
        v_drift = sep / (6.0 * dt)
        vx_A, vy_A = v_drift * np.cos(theta), v_drift * np.sin(theta)

        for step in range(n_steps):
            if step < 6:
                grad_Ax, grad_Ay = np.gradient(C_A, cfg.dx, cfg.dy)
                grad_Bx, grad_By = np.gradient(C_B, cfg.dx, cfg.dy)
                C_A -= dt * (vx_A * grad_Ax + vy_A * grad_Ay)
                C_B -= dt * (-vx_A * grad_Bx - vy_A * grad_By)

            R1 = k1_rate * np.maximum(C_A, 0) * np.maximum(C_B, 0)
            R2 = k2_rate * np.maximum(C_AB, 0)

            C_A += dt * (0.8 * cfg.D_diff * pde.laplacian_2d(C_A) - R1)
            C_B += dt * (0.8 * cfg.D_diff * pde.laplacian_2d(C_B) - R1)
            C_AB += dt * (0.5 * cfg.D_diff * pde.laplacian_2d(C_AB) + R1 - R2)
            C_C += dt * (0.4 * cfg.D_diff * pde.laplacian_2d(C_C) + R2)
            T_enthalpy += dt * (0.6 * cfg.D_diff * pde.laplacian_2d(T_enthalpy) + 0.7 * R1 + 0.5 * R2 - T_enthalpy / cfg.tau_relax)

            C_A = np.maximum(C_A, 0.0)
            C_B = np.maximum(C_B, 0.0)

            if step == 6:
                Y_mid_complex[:, i] = C_AB.flatten()

        dn_final = cfg.dn_scale * (1.2 * C_C + 1.8 * C_AB + 1.5 * (C_A + C_B) + 0.9 * T_enthalpy)
        dn_3d = pde.build_3d_index_volume(dn_final)
        E_out = optics.propagate_bpm(E_probe, dn_3d)
        X_cmos[:, i] = optics.record_cmos_hologram(E_out, use_reference=True, rng=rng)

    Y_all = np.vstack([Y_init_radicals, Y_mid_complex])
    model = RidgeReadout(lambd=8e-2).fit(X_cmos[:, :split], Y_all[:, :split])
    Y_pred = model.predict(X_cmos[:, split:])

    Y_rad_pred = np.maximum(Y_pred[:N_pix, :], 0.0)
    Y_ts_pred = np.maximum(Y_pred[N_pix:, :], 0.0)

    fid_rad = np.mean([calc_wave_fidelity(Y_init_radicals[:, split+k], Y_rad_pred[:, k]) for k in range(N_samples - split)]) * 100
    fid_ts = np.mean([calc_wave_fidelity(Y_mid_complex[:, split+k], Y_ts_pred[:, k]) for k in range(N_samples - split)]) * 100

    print(f"Vanished Radicals (A* + B*) PDE Fidelity:          {fid_rad:.2f}%")
    print(f"Intermediate Transition State [AB]# PDE Fidelity:  {fid_ts:.2f}%\n")

    idx = 4
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    ax[0].imshow(Y_init_radicals[:, split+idx].reshape(cfg.Nx, cfg.Ny).T, origin='lower', cmap='inferno')
    ax[0].set_title("1. Initial $A^\\bullet + B^\\bullet$ ($t=0$)", fontweight='bold')
    ax[1].imshow(Y_mid_complex[:, split+idx].reshape(cfg.Nx, cfg.Ny).T, origin='lower', cmap='magma')
    ax[1].set_title("2. Mid Transition $[AB]^\\ddagger$", fontweight='bold')
    ax[2].imshow(X_cmos[:, split+idx].reshape(cfg.Nx, cfg.Ny).T, cmap='viridis')
    ax[2].set_title("3. 12-bit Final CMOS Speckle", fontweight='bold'); ax[2].axis('off')
    ax[3].imshow(Y_rad_pred[:, idx].reshape(cfg.Nx, cfg.Ny).T, origin='lower', cmap='inferno')
    ax[3].set_title(f"4. Inverted $A^\\bullet+B^\\bullet$ ({fid_rad:.1f}%)", fontweight='bold')

    plt.tight_layout()
    plt.savefig("real_exp3_chemical_pde.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    run_experiment_3()