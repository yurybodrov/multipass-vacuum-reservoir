"""
physics_config.py
Physical constants (SI) and hardware parameters of the high-Q optical resonator
with magnetically confined ambipolar diffusion (tau_diff > tau_relax).
"""
import numpy as np

C_LIGHT = 2.99792458e8
EPSILON_0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19
M_ELECTRON = 9.1093837015e-31
H_PLANCK = 6.62607015e-34
K_BOLTZMANN = 1.380649e-23

class ChamberConfig:
    def __init__(self):
        # 1. Chamber Geometry & Magnetically Confined Plasma Parameters
        self.Lx = 4.0e-3
        self.Ly = 4.0e-3
        self.Lz = 0.10                   # Physical chamber length: 10 cm
        self.Nx = 24
        self.Ny = 24
        self.Nz = 12                     # Axial slices for Split-Step Fourier BPM
        
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz

        self.pressure_torr = 0.5
        self.pressure_pa = self.pressure_torr * 133.322
        self.temp_kelvin = 320.0
        
        # Transverse ambipolar diffusion D_perp under axial magnetic confinement (B_z ~ 0.08 T):
        # Confinement prevents the 0.45 mm spot from washing out into the walls in 4 us,
        # extending the spatial diffusion lifetime tau_diff = w^2 / (4*D_perp) to ~63 us.
        self.D_diff = 8.0e-4             # 8.0 cm^2/s = 8.0e-4 m^2/s
        self.tau_relax = 50.0e-6         # 50 microseconds bulk relaxation lifetime
        
        # 2. High-Q Multipass Herriott Optical Cavity
        # 400 passes across the 10-cm chamber -> L_eff = 40.0 meters
        self.N_passes = 400
        self.L_eff = self.L_z_eff = self.Lz * self.N_passes
        
        # 3. Optical Probe Laser (He-Ne 632.8 nm)
        self.wavelength = 632.8e-9
        self.k0 = 2.0 * np.pi / self.wavelength
        self.freq_opt = C_LIGHT / self.wavelength
        self.photon_energy = H_PLANCK * self.freq_opt
        
        # Physical refractive index modulation depth in rarefied plasma / metastable gas
        self.dn_scale = 1.2e-8
        self.alpha_kerr = 1.5e-8
        
        # 4. Microwave Standing-Wave Perturbation Cavity (60 GHz)
        self.freq_mw = 60.0e9
        self.wavelength_mw = C_LIGHT / self.freq_mw
        self.k_mw = 2.0 * np.pi / self.wavelength_mw
        
        # 5. Monochrome 12-bit CMOS Sensor Array
        self.N_cmos_side = 24
        self.exposure_time = 2.0e-6
        self.quantum_efficiency = 0.72
        self.readout_noise_e = 2.5
        self.adc_bits = 12
        self.full_well_e = 25000.0