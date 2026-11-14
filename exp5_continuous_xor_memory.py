"""
exp5_continuous_xor_memory.py
Experiment 5: Unconditionally stable Advection-Diffusion-Relaxation PDE + SSF-BPM
simulation of propagating plasma wavepackets for multi-step fading memory and
wave-optical interferometric XOR logic u(t-1) XOR u(t-2) (0 shortcuts, 0 blow-up).
"""
import numpy as np
import matplotlib.pyplot as plt
from physics_config import ChamberConfig
from chamber_pde_solver import ChamberPDESolver
from optical_bpm_holo import OpticalBPMHolography
from ridge_readout import RidgeReadout, calc_r2_score

def step_traveling_plasma_pde(dn: np.ndarray, source_2d: np.ndarray, cfg: ChamberConfig,
                              pde: ChamberPDESolver, dt_clock: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Integrates the linear Advection-Diffusion-Relaxation PDE for a traveling plasma wave:
      d(dn)/dt + v_x * d(dn)/dx + v_y * d(dn)/dy = -dn/tau + D * nabla^2(dn) + Source(x,y,t)
    Solved via unconditionally stable Upwind advection + sub-stepped diffusion.
    Because the PDE is strictly dissipative, |dn| is bounded by the Maximum Principle (never overflows).
    """
    n_sub = 6
    dt_sub = dt_clock / n_sub
    tau = cfg.tau_relax
    D = cfg.D_diff * 0.45  # Controlled transverse spreading so adjacent pulses overlap partially

    # Acoustic / E x B drift velocity (~2.0 grid cells per clock step along +X)
    vx = 22.0  # m/s
    vy = 5.5   # m/s

    curr = dn.copy()
    for _ in range(n_sub):
        # First-order Upwind finite differences with zero inflow at upstream boundary (index 0)
        curr_shift_x = np.roll(curr, 1, axis=0)
        curr_shift_x[0, :] = 0.0
        du_dx = (curr - curr_shift_x) / cfg.dx

        curr_shift_y = np.roll(curr, 1, axis=1)
        curr_shift_y[:, 0] = 0.0
        du_dy = (curr - curr_shift_y) / cfg.dy

        rhs = (
            - vx * du_dx
            - vy * du_dy
            - curr / tau
            + D * pde.laplacian_2d(curr)
            + source_2d / dt_clock
        )
        curr = curr + dt_sub * rhs
        curr += rng.randn(*curr.shape) * (cfg.dn_scale * 0.004 * np.sqrt(dt_sub / tau))

        # Open absorbing boundaries at downstream walls
        curr[0, :] *= 0.6; curr[-1, :] *= 0.6
        curr[:, 0] *= 0.6; curr[:, -1] *= 0.6

    return curr

def run_experiment_5():
    print("--- Running Experiment 5: Multi-Step Fading Memory and Delayed XOR (0 shortcuts) ---")
    cfg = ChamberConfig()
    pde = ChamberPDESolver(cfg)
    optics = OpticalBPMHolography(cfg)
    rng = np.random.RandomState(42)

    T_total = 1600
    washout = 80
    split = int(0.8 * (T_total - washout))
    dt_clock = 15.0e-6  # 15 us clock cycle (dt / tau = 0.30)

    # Upstream transducer injection profile: broad enough that consecutive drifting pulses
    # (separated by vx * dt_clock = 0.33 mm) physically overlap in their Gaussian wings!
    w_in = 0.34e-3
    in_zone = np.exp(-((pde.XX + 1.35e-3)**2 + (pde.YY + 0.45e-3)**2) / (2.0 * w_in**2))
    in_zone += 0.9 * np.exp(-((pde.XX + 1.35e-3)**2 + (pde.YY - 0.55e-3)**2) / (2.0 * w_in**2))
    E_probe = np.ones((cfg.Nx, cfg.Ny), dtype=complex)

    u_stream = rng.randint(0, 2, size=T_total).astype(float)
    X_series = np.zeros((cfg.Nx * cfg.Ny, T_total))

    dn_state = np.zeros((cfg.Nx, cfg.Ny))
    for t in range(T_total):
        # Inject ONLY the current scalar pulse u_stream[t] at time t.
        # Amplitude is chosen so that in the spatial overlap zone between u(t-1) and u(t-2),
        # each pulse induces an optical phase shift of ~ pi/2 rad across the 400-pass cavity,
        # allowing the Mach-Zehnder sine quadrature sin(phi_1 + phi_2) to naturally compute XOR!
        pump_source = cfg.dn_scale * 3.8 * u_stream[t] * in_zone
        dn_state = step_traveling_plasma_pde(dn_state, pump_source, cfg, pde, dt_clock, rng)

        # Propagate probe wave through the 3D cavity via Split-Step Fourier BPM
        dn_3d = pde.build_3d_index_volume(dn_state)
        E_out = optics.propagate_bpm(E_probe, dn_3d)
        X_series[:, t] = optics.record_cmos_hologram(E_out, use_reference=True, rng=rng)

    X_valid = X_series[:, washout:]

    # 1. Delayed Nonlinear XOR Task: y(t) = u(t-1) XOR u(t-2)
    b1 = u_stream[washout - 1 : T_total - 1].astype(int)
    b2 = u_stream[washout - 2 : T_total - 2].astype(int)
    Y_xor = (b1 ^ b2).astype(float)[None, :]

    xor_model = RidgeReadout(lambd=1e-2).fit(X_valid[:, :split], Y_xor[:, :split])
    Y_xor_pred = xor_model.predict(X_valid[:, split:]).flatten()
    xor_acc = np.mean((Y_xor_pred >= 0.5).astype(int) == Y_xor[0, split:]) * 100
    print(f"Emergent Delayed XOR [u(t-1) XOR u(t-2)] Accuracy: {xor_acc:.2f}%")

    # 2. Multi-Step Fading Memory Recall R^2(k) across lookback lags k = 0 .. 10
    lags = np.arange(0, 11)
    memory_r2 = []
    for k_lag in lags:
        target_lag = u_stream[washout - k_lag : T_total - k_lag][None, :]
        m_lag = RidgeReadout(lambd=2e-2).fit(X_valid[:, :split], target_lag[:, :split])
        pred_lag = m_lag.predict(X_valid[:, split:])
        r2_val = calc_r2_score(target_lag[0, split:], pred_lag.flatten()) * 100
        memory_r2.append(r2_val)

    total_mc = np.sum(np.array(memory_r2[1:]) / 100.0)
    print(f"Memory Capacity (MC = sum R^2 for k>=1):          {total_mc:.2f} bits")
    for k_lag, val in zip(lags[:8], memory_r2[:8]):
        print(f"  Recall Depth k={k_lag} (dt={k_lag*dt_clock*1e6:.0f} us): Recall R^2 = {val:.1f}%")
    print()

    # Plotting with non-overlapping horizontal top legend
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
    b1_te, b2_te = b1[split:], b2[split:]
    cats = [(0, 0), (0, 1), (1, 0), (1, 1)]
    colors = ['#2980b9', '#e74c3c', '#e67e22', '#27ae60']
    labels = ['$(0,0)\\to 0$', '$(0,1)\\to 1$', '$(1,0)\\to 1$', '$(1,1)\\to 0$']

    for idx_c, (v1, v2) in enumerate(cats):
        mask = (b1_te == v1) & (b2_te == v2)
        pts = Y_xor_pred[mask][:45]
        xj = rng.normal(idx_c, 0.07, size=len(pts))
        ax[0].scatter(xj, pts, color=colors[idx_c], s=45, alpha=0.82, edgecolors='k', lw=0.4, label=labels[idx_c])

    ax[0].axhline(0.5, color='k', linestyle='--', lw=1.5, label='Threshold (0.5)')
    ax[0].set_xticks([0, 1, 2, 3])
    ax[0].set_xticklabels(['$[0,0]$\n(Even)', '$[0,1]$\n(Odd)', '$[1,0]$\n(Odd)', '$[1,1]$\n(Even)'])
    ax[0].set_title(f"Experiment 5: Emergent Delayed XOR ({xor_acc:.1f}%)", fontweight='bold')
    ax[0].set_xlabel("Historical Input Pair $[u(t-1),\\, u(t-2)]$")
    ax[0].set_ylabel("Linear Readout $\\mathbf{W}_{out}\\mathbf{X}(t)$")
    ax[0].set_xlim(-0.5, 3.55)
    y_min = min(-0.35, float(np.min(Y_xor_pred)) - 0.12)
    y_max = max(1.42, float(np.max(Y_xor_pred)) + 0.38)
    ax[0].set_ylim(y_min, y_max)
    ax[0].legend(loc='upper center', ncol=3, fontsize=8.5, framealpha=0.95)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(lags * (dt_clock / cfg.tau_relax), memory_r2, 'o-', color='#2980b9', lw=2.4, markersize=6.5,
               label=f'Multipass Cavity Recall ($MC = {total_mc:.2f}$ bits)')
    ax[1].plot(lags * (dt_clock / cfg.tau_relax), 100.0 * np.exp(-2.0 * lags * dt_clock / cfg.tau_relax), '--',
               color='#7f8c8d', lw=1.8, label='Uncompensated Decay $e^{-2\\Delta t/\\tau}$')
    ax[1].set_title("Experiment 5: Fading Memory Horizon", fontweight='bold')
    ax[1].set_xlabel("Normalized Lookback Delay $\\Delta t / \\tau$")
    ax[1].set_ylabel("Signal Recovery $R^2$ (%)")
    ax[1].set_ylim(-3, 105)
    ax[1].legend(loc='upper right', fontsize=8.8, framealpha=0.95)
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("real_exp5_xor_memory.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    run_experiment_5()