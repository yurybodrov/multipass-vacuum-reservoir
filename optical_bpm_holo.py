"""
optical_bpm_holo.py
Split-Step Fourier BPM solver for the Paraxial Wave Equation and high-Q cavity.
"""
import numpy as np
from physics_config import ChamberConfig

class OpticalBPMHolography:
    def __init__(self, cfg: ChamberConfig):
        self.cfg = cfg
        kx = 2.0 * np.pi * np.fft.fftfreq(cfg.Nx, d=cfg.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(cfg.Ny, d=cfg.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.K_perp_sq = self.KX**2 + self.KY**2
        
        # Multi-pass step SSF propagation back-and-forth
        # Multiplies effective optical step dz by N_passes to account for optical cavity length!
        self.effective_dz = cfg.dz * cfg.N_passes
        self.diff_half_step = np.exp(-1j * self.K_perp_sq * (self.effective_dz / 2.0) / (2.0 * cfg.k0))
        
        L_free = 0.015
        self.free_space_prop = np.exp(-1j * self.K_perp_sq * L_free / (2.0 * cfg.k0))

        # Carrier frequency of off-axis reference wave
        x = np.linspace(-cfg.Lx / 2, cfg.Lx / 2, cfg.Nx)
        y = np.linspace(-cfg.Ly / 2, cfg.Ly / 2, cfg.Ny)
        XX, YY = np.meshgrid(x, y, indexing='ij')
        
        k_tilt_x = 2.0 * np.pi / (3.2 * cfg.dx)
        k_tilt_y = 2.0 * np.pi / (3.5 * cfg.dy)
        self.A_ref = 6.0
        self.E_ref_field = self.A_ref * np.exp(1j * (k_tilt_x * XX + k_tilt_y * YY))

    def propagate_bpm(self, E_in: np.ndarray, dn_3d: np.ndarray) -> np.ndarray:
        """SSF-BPM propagation with effective multi-pass optical path inside the cavity."""
        E = E_in.astype(complex).copy()
        for iz in range(self.cfg.Nz):
            E_k = np.fft.fft2(E) * self.diff_half_step
            E = np.fft.ifft2(E_k)
            
            # The refractive index phase shift is magnified by the multipass factor (N_passes)
            phase_screen = np.exp(1j * self.cfg.k0 * dn_3d[iz] * self.effective_dz)
            E = E * phase_screen
            
            E_k = np.fft.fft2(E) * self.diff_half_step
            E = np.fft.ifft2(E_k)

        E_cmos_plane = np.fft.ifft2(np.fft.fft2(E) * self.free_space_prop)
        return E_cmos_plane

    def record_cmos_hologram(self, E_out: np.ndarray, use_reference: bool = True,
                             rng: np.random.RandomState = None) -> np.ndarray:
        """Monochrome CMOS: interference, Quantum Efficiency, Poisson and dark Gaussian noise."""
        if rng is None:
            rng = np.random.RandomState()

        if use_reference:
            total_field = E_out + self.E_ref_field
        else:
            total_field = E_out

        intensity = np.abs(total_field)**2
        scale_to_electrons = (0.60 * self.cfg.full_well_e) / (np.max(np.abs(self.A_ref + 1.5)**2) + 1e-12)
        mean_electrons = intensity * scale_to_electrons * self.cfg.quantum_efficiency
        
        noisy_electrons = rng.poisson(np.maximum(mean_electrons, 0.0)).astype(float)
        noisy_electrons += rng.randn(*intensity.shape) * self.cfg.readout_noise_e
        
        # 12-bit ADC digitization
        max_adc = (2**self.cfg.adc_bits) - 1
        adc_counts = np.clip(np.round(noisy_electrons / self.cfg.full_well_e * max_adc), 0, max_adc)
        return adc_counts.flatten()