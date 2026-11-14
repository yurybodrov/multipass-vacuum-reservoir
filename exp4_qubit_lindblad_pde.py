"""
exp4_qubit_lindblad_pde.py
Experiment 4: Quantum Qubit Decoherence Inversion via Lindblad Master Equation
with Quantum Fluctuation-Dissipation reservoir noise and Split-Step Fourier BPM.
"""
import numpy as np
import matplotlib.pyplot as plt
from physics_config import ChamberConfig
from chamber_pde_solver import ChamberPDESolver
from optical_bpm_holo import OpticalBPMHolography
from ridge_readout import RidgeReadout

def run_experiment_4():
    print("--- Running Experiment 4: Lindblad Master Equation Qubit Decoherence Inversion ---")
    cfg = ChamberConfig()
    pde = ChamberPDESolver(cfg)
    optics = OpticalBPMHolography(cfg)
    rng = np.random.RandomState(42)

    N_samples = 1000
    split = int(0.8 * N_samples)
    
    theta_q = rng.uniform(0, np.pi, N_samples)
    phi_q = rng.uniform(0, 2.0 * np.pi, N_samples)
    Bloch_init = np.vstack([
        np.sin(theta_q) * np.cos(phi_q),
        np.sin(theta_q) * np.sin(phi_q),
        np.cos(theta_q)
    ])

    r_zone = 0.65e-3
    zones = [
        np.exp(-((pde.XX + 0.7e-3)**2 + (pde.YY + 0.7e-3)**2) / (2 * r_zone**2)),
        np.exp(-((pde.XX - 0.7e-3)**2 + (pde.YY - 0.7e-3)**2) / (2 * r_zone**2)),
        np.exp(-((pde.XX + 0.7e-3)**2 + (pde.YY - 0.7e-3)**2) / (2 * r_zone**2)),
        np.exp(-((pde.XX - 0.7e-3)**2 + (pde.YY + 0.7e-3)**2) / (2 * r_zone**2)),
    ]
    E_probe = np.ones((cfg.Nx, cfg.Ny), dtype=complex)

    dt_tau_ratios = np.linspace(0.2, 2.8, 10)
    qubit_fidelities = []
    uncorrected_fidelities = []

    for idx_r, ratio in enumerate(dt_tau_ratios):
        total_t = ratio * cfg.tau_relax
        n_steps = 6
        dt_step = total_t / n_steps

        # Quantum Fluctuation-Dissipation Theorem: as coherent amplitude decays by exp(-ratio),
        # vacuum zero-point / thermal reservoir fluctuations enter with std proportional to sqrt(1 - exp(-2*ratio))
        q_fdt_std = cfg.dn_scale * 0.42 * np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * ratio))) * np.exp(0.45 * ratio)

        X_cmos = np.zeros((cfg.Nx * cfg.Ny, N_samples))

        for i in range(N_samples):
            rx0, ry0, rz0 = Bloch_init[:, i]
            p0 = 0.5 * (1.0 + rz0)
            p1 = 0.5 * (1.0 - rz0)
            coh_re = 0.5 * rx0
            coh_im = 0.5 * ry0

            dn_state = cfg.dn_scale * 2.5 * (
                p0 * zones[0] + p1 * zones[1] + coh_re * zones[2] + coh_im * zones[3]
            )

            for _ in range(n_steps):
                dn_state = pde.step_reservoir_pde(
                    dn_state, np.ones((cfg.Nx, cfg.Ny)) * 0.05, np.zeros((cfg.Nx, cfg.Ny)), dt_step, rng
                )

            # Inject Lindblad quantum bath fluctuations accumulated over total_t
            dn_state += rng.randn(cfg.Nx, cfg.Ny) * q_fdt_std

            dn_3d = pde.build_3d_index_volume(dn_state)
            E_out = optics.propagate_bpm(E_probe, dn_3d)
            X_cmos[:, i] = optics.record_cmos_hologram(E_out, use_reference=True, rng=rng)

        model = RidgeReadout(lambd=8e-2).fit(X_cmos[:, :split], Bloch_init[:, :split])
        B_pred = model.predict(X_cmos[:, split:])
        B_pred_norm = B_pred / (np.linalg.norm(B_pred, axis=0, keepdims=True) + 1e-12)

        dot_prod = np.sum(Bloch_init[:, split:] * B_pred_norm, axis=0)
        fid_q = np.mean(0.5 * (1.0 + np.clip(dot_prod, -1.0, 1.0))) * 100
        qubit_fidelities.append(fid_q)

        uncorr = 0.5 * (1.0 + np.exp(-ratio)) * 100
        uncorrected_fidelities.append(uncorr)

        if idx_r == 1:
            sample_true = Bloch_init[:, split:split+40]
            sample_pred = B_pred_norm[:, :40]
            fid_sample = fid_q
            ratio_sample = ratio

        print(f"  dt/tau = {ratio:.2f} -> Recovered Qubit Fidelity: {fid_q:.2f}% (Uncorrected: {uncorr:.1f}%)")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].scatter(sample_true[0, :], sample_true[1, :], color='#1b4f72', s=55, label='Initial $|\\psi(0)\\rangle$')
    ax[0].scatter(sample_pred[0, :], sample_pred[1, :], color='#e74c3c', marker='x', s=55, lw=2, label='PDE+BPM Recovered')
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', lw=1.3)
    ax[0].add_artist(circle)
    ax[0].set_xlim(-1.2, 1.2); ax[0].set_ylim(-1.35, 1.15); ax[0].set_aspect('equal')
    ax[0].set_title(f"Bloch Recovery at $\\Delta t/\\tau={ratio_sample:.2f}$ ({fid_sample:.1f}%)", fontweight='bold')
    ax[0].set_xlabel("$\\langle \\sigma_x \\rangle$"); ax[0].set_ylabel("$\\langle \\sigma_y \\rangle$")
    ax[0].legend(loc='lower center', ncol=2, fontsize=8.5)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(dt_tau_ratios, qubit_fidelities, 'o-', color='#27ae60', lw=2.4, label='PDE+BPM Coherence Recovery')
    ax[1].plot(dt_tau_ratios, uncorrected_fidelities, '--', color='#7f8c8d', lw=2.0, label='Bare Lindblad $T_2$ Decay')
    ax[1].axhline(90.0, color='#c0392b', linestyle=':', lw=1.5, label='90% Threshold')
    ax[1].set_xlabel("Normalized Lookback Time $\\Delta t / \\tau$")
    ax[1].set_ylabel("Quantum State Fidelity (%)")
    ax[1].set_title("Lindblad Decoherence Inversion Horizon", fontweight='bold')
    ax[1].set_ylim(48, 103)
    ax[1].legend(loc='lower left')
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("real_exp4_qubit_lindblad.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print()

if __name__ == "__main__":
    run_experiment_4()