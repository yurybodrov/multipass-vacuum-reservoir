"""
chamber_pde_solver.py
Time-stepped Finite-Difference PDE solvers with proper zero-flux / absorbing boundaries.
"""
import numpy as np
from physics_config import ChamberConfig

class ChamberPDESolver:
    def __init__(self, cfg: ChamberConfig):
        self.cfg = cfg
        x = np.linspace(-cfg.Lx / 2, cfg.Lx / 2, cfg.Nx)
        y = np.linspace(-cfg.Ly / 2, cfg.Ly / 2, cfg.Ny)
        self.XX, self.YY = np.meshgrid(x, y, indexing='ij')
        
        rng = np.random.RandomState(101)
        self.mw_grating_3d = np.zeros((cfg.Nz, cfg.Nx, cfg.Ny))
        for iz in range(cfg.Nz):
            z_coord = iz * cfg.dz
            m1 = np.cos(cfg.k_mw * self.XX + 0.7 * cfg.k_mw * self.YY + 0.5 * cfg.k_mw * z_coord)
            m2 = np.sin(0.8 * cfg.k_mw * self.XX - cfg.k_mw * self.YY + 0.9 * cfg.k_mw * z_coord)
            m3 = np.cos(1.3 * cfg.k_mw * (self.XX * np.cos(0.4) + self.YY * np.sin(0.4)) - 0.6 * cfg.k_mw * z_coord)
            disorder = 0.25 * rng.randn(cfg.Nx, cfg.Ny)
            pattern = (m1 + m2 + m3) / 3.0 + disorder
            self.mw_grating_3d[iz] = pattern / np.std(pattern)

    def laplacian_2d(self, field: np.ndarray) -> np.ndarray:
        """5-point finite-difference 2D Laplacian with correct wall boundary indexing."""
        d2x = (np.roll(field, -1, axis=0) - 2.0 * field + np.roll(field, 1, axis=0)) / (self.cfg.dx**2)
        d2y = (np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1)) / (self.cfg.dy**2)
        lap = d2x + d2y
        # Properly indexed Dirichlet wall damping on each respective boundary edge
        lap[0, :]  = -field[0, :]  / (self.cfg.dx**2)
        lap[-1, :] = -field[-1, :] / (self.cfg.dx**2)
        lap[:, 0]  = -field[:, 0]  / (self.cfg.dy**2)
        lap[:, -1] = -field[:, -1] / (self.cfg.dy**2)
        return lap

    def step_reservoir_pde(self, dn_state: np.ndarray, E_in_intensity: np.ndarray,
                           phi_ext: np.ndarray, dt: float, rng: np.random.RandomState) -> np.ndarray:
        """Integrates the 2D reservoir relaxation-diffusion PDE via 2nd-order Heun/RK2."""
        tau = self.cfg.tau_relax
        D = self.cfg.D_diff
        
        mw_base = self.mw_grating_3d[0]
        drive_source = self.cfg.alpha_kerr * (0.2 * mw_base**2 + E_in_intensity + 0.6 * mw_base * np.sqrt(np.maximum(E_in_intensity, 0)))

        def rhs(state):
            return (-state / tau) + D * self.laplacian_2d(state) + (drive_source + phi_ext) / tau

        max_dt_cfl = 0.18 * min(self.cfg.dx**2, self.cfg.dy**2) / (D + 1e-12)
        n_sub = max(1, int(np.ceil(dt / max_dt_cfl)))
        dt_sub = dt / n_sub

        curr = dn_state.copy()
        for _ in range(n_sub):
            k1 = rhs(curr)
            k2 = rhs(curr + dt_sub * k1)
            curr = curr + 0.5 * dt_sub * (k1 + k2)
            curr += rng.randn(*curr.shape) * (self.cfg.dn_scale * 0.006 * np.sqrt(dt_sub / tau))

        curr[0, :] *= 0.85
        curr[-1, :] *= 0.85
        curr[:, 0] *= 0.85
        curr[:, -1] *= 0.85
        return curr

    def build_3d_index_volume(self, dn_2d: np.ndarray) -> np.ndarray:
        """Maps 2D state dn_2d onto the 3D optical path with high-Q standing waves."""
        dn_3d = np.zeros((self.cfg.Nz, self.cfg.Nx, self.cfg.Ny))
        for iz in range(self.cfg.Nz):
            dn_3d[iz] = self.cfg.dn_scale * 0.65 * self.mw_grating_3d[iz] + dn_2d * (1.0 + 0.5 * self.mw_grating_3d[iz])
        return dn_3d