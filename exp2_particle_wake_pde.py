"""
exp2_particle_wake_pde.py
Experiment 2: Step-by-step PDE simulation of an ionizing particle wake undergoing
ambipolar diffusion and relaxation, with proper ionization rate normalization (tau / dt_step).
"""
import numpy as np
import matplotlib.pyplot as plt
from physics_config import ChamberConfig
from chamber_pde_solver import ChamberPDESolver
from optical_bpm_holo import OpticalBPMHolography
from ridge_readout import RidgeReadout, calc_wave_fidelity, calc_r2_score

def run_experiment_2():
    print("--- Running Experiment 2: Step-by-Step PDE Particle Wake Recovery ---")
    cfg = ChamberConfig()
    pde = ChamberPDESolver(cfg)
    optics = OpticalBPMHolography(cfg)
    rng = np.random.RandomState(42)

    N_samples = 1400
    split = int(0.8 * N_samples)
    n_time_steps = 10
    total_flight_time = 35.0e-6  # 35 us lookback interval
    dt_step = total_flight_time / n_time_steps

    X_cmos = np.zeros((cfg.Nx * cfg.Ny, N_samples))
    Y_past_density = np.zeros((cfg.Nx * cfg.Ny, N_samples))
    Y_present_density = np.zeros((cfg.Nx * cfg.Ny, N_samples))
    Y_kinematics = np.zeros((4, N_samples))

    E_probe = np.ones((cfg.Nx, cfg.Ny), dtype=complex)
    beam_Radius = 0.48e-3  # Initial ionization core radius (0.48 mm)

    for i in range(N_samples):
        x0 = rng.uniform(-0.85e-3, 0.85e-3)
        y0 = rng.uniform(-0.85e-3, 0.85e-3)
        dx_flight = rng.uniform(-0.95e-3, 0.95e-3)
        dy_flight = rng.uniform(-0.95e-3, 0.95e-3)
        vx = dx_flight / total_flight_time
        vy = dy_flight / total_flight_time

        Y_kinematics[:, i] = [x0 * 1e3, y0 * 1e3, dx_flight * 1e3, dy_flight * 1e3]

        # Target spatial density profile at past entry time t - Delta_t
        rho_initial = np.exp(-((pde.XX - x0)**2 + (pde.YY - y0)**2) / (2.0 * beam_Radius**2))
        Y_past_density[:, i] = rho_initial.flatten()

        # Present particle density profile at time t (end of track)
        x1, y1 = x0 + dx_flight, y0 + dy_flight
        rho_present = np.exp(-((pde.XX - x1)**2 + (pde.YY - y1)**2) / (2.0 * (0.75 * beam_Radius)**2))
        Y_present_density[:, i] = rho_present.flatten()

        # Integrate the 2D relaxation-diffusion PDE step-by-step along the trajectory
        dn_state = np.zeros((cfg.Nx, cfg.Ny))
        rate_norm = cfg.tau_relax / dt_step  # Compensates for division by tau inside step_reservoir_pde

        for step in range(n_time_steps):
            t_curr = step * dt_step
            x_t = x0 + vx * t_curr
            y_t = y0 + vy * t_curr
            frac = step / float(n_time_steps - 1)  # 0.0 at entry (t - dt) -> 1.0 at present (t)

            # Physical Bragg/wake profile: older wake segments were deposited by a broader shock wave,
            # while the present particle head creates a sharper, high-density plasma core
            w_step = beam_Radius * (1.15 - 0.35 * frac)
            r_sq = (pde.XX - x_t)**2 + (pde.YY - y_t)**2

            ion_track = (cfg.dn_scale * 0.22 * rate_norm) * (0.7 + 0.6 * frac) * np.exp(-r_sq / (2.0 * w_step**2))
            dn_state = pde.step_reservoir_pde(
                dn_state, np.ones((cfg.Nx, cfg.Ny)) * 0.01, ion_track, dt_step, rng
            )

        dn_3d = pde.build_3d_index_volume(dn_state)
        E_out = optics.propagate_bpm(E_probe, dn_3d)
        X_cmos[:, i] = optics.record_cmos_hologram(E_out, use_reference=True, rng=rng)

    # Train readout layer to recover past density, present density, and kinematic vector
    Y_combined = np.vstack([Y_past_density, Y_present_density, Y_kinematics])
    readout = RidgeReadout(lambd=2e-2).fit(X_cmos[:, :split], Y_combined[:, :split])
    Y_pred = readout.predict(X_cmos[:, split:])

    N_pix = cfg.Nx * cfg.Ny
    Y_dens_test = Y_past_density[:, split:]
    Y_dens_pred = np.maximum(Y_pred[:N_pix, :], 0.0)
    Y_pres_pred = np.maximum(Y_pred[N_pix:2*N_pix, :], 0.0)
    Y_kin_test = Y_kinematics[:, split:]

    # Extract entry (t - dt) and present (t) centroids from the reconstructed wavepackets
    x_mm = pde.XX.flatten() * 1e3
    y_mm = pde.YY.flatten() * 1e3
    w_past = Y_dens_pred**4
    w_pres = Y_pres_pred**4
    x0_c = (x_mm @ w_past) / (np.sum(w_past, axis=0) + 1e-12)
    y0_c = (y_mm @ w_past) / (np.sum(w_past, axis=0) + 1e-12)
    x1_c = (x_mm @ w_pres) / (np.sum(w_pres, axis=0) + 1e-12)
    y1_c = (y_mm @ w_pres) / (np.sum(w_pres, axis=0) + 1e-12)

    Y_kin_direct = Y_pred[2*N_pix:, :]
    Y_kin_centroid = np.vstack([x0_c, y0_c, x1_c - x0_c, y1_c - y0_c])
    Y_kin_pred = 0.65 * Y_kin_direct + 0.35 * Y_kin_centroid

    fid_spatial = np.mean([calc_wave_fidelity(Y_dens_test[:, k], Y_dens_pred[:, k]) for k in range(N_samples - split)]) * 100
    r2_kin = np.mean([calc_r2_score(Y_kin_test[j, :], Y_kin_pred[j, :]) for j in range(4)]) * 100

    print(f"Past Spatial Density Fidelity:         {fid_spatial:.2f}%")
    print(f"Initial Kinematic Track Accuracy (R2): {r2_kin:.2f}%\n")

    idx = 12
    ext = [-cfg.Lx*500, cfg.Lx*500, -cfg.Ly*500, cfg.Ly*500]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x0_t, y0_t, dxt, dyt = Y_kin_test[:, idx]
    x0_p, y0_p, dxp, dyp = Y_kin_pred[:, idx]

    ax[0].imshow(Y_dens_test[:, idx].reshape(cfg.Nx, cfg.Ny).T, extent=ext, origin='lower', cmap='inferno')
    ax[0].quiver(x0_t, y0_t, dxt, dyt, angles='xy', scale_units='xy', scale=1, color='cyan', width=0.022, label='True Track')
    ax[0].scatter([x0_t], [y0_t], color='white', edgecolors='k', s=60, zorder=5)
    ax[0].set_title("Target Particle Track ($t - \\Delta t$)", fontweight='bold')
    ax[0].set_xlabel("X (mm)"); ax[0].set_ylabel("Y (mm)")
    ax[0].legend(loc='upper right')

    ax[1].imshow(X_cmos[:, split+idx].reshape(cfg.Nx, cfg.Ny).T, cmap='viridis')
    ax[1].set_title("Monochrome CMOS Wake Speckle", fontweight='bold')
    ax[1].axis('off')

    ax[2].imshow(Y_dens_pred[:, idx].reshape(cfg.Nx, cfg.Ny).T, extent=ext, origin='lower', cmap='inferno')
    ax[2].quiver(x0_p, y0_p, dxp, dyp, angles='xy', scale_units='xy', scale=1, color='lime', width=0.022, label='Recovered Track')
    ax[2].scatter([x0_p], [y0_p], color='white', edgecolors='k', s=60, zorder=5)
    ax[2].set_title(f"Reconstructed Past ({fid_spatial:.1f}%, $R^2$: {r2_kin:.1f}%)", fontweight='bold')
    ax[2].set_xlabel("X (mm)"); ax[2].set_ylabel("Y (mm)")
    ax[2].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("real_exp2_particle_wake.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    run_experiment_2()