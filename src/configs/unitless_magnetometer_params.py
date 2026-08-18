# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
import numpy as np

@dataclass
class UnitlessSimulationParams:
    x_0: np.ndarray
    dt: float
    T2: float
    w01: float
    dc: float
    tau: float
    type: str
    sig_v: float

@dataclass
class FilterNoiseParams:
    Q: np.ndarray

@dataclass
class FilterMeasurementParams:
    dim_z: int
    H: np.ndarray
    measurement_strength: float
    R: np.ndarray

@dataclass
class UnitlessFilterParams:
    x_0: np.ndarray
    t_0: float
    dt: float
    inference_method: str
    type: str
    tau: float
    w01: float
    P0: np.ndarray
    noise: FilterNoiseParams
    measurement: FilterMeasurementParams
    max_step: float = None

@dataclass
class UnitlessSimpleMagnetometerConfigurator:
    # --- Baseline Physics & Simulation Parameters ---
    N: float = 0.44 * 1e12                  # Number of atoms
    q: float = 0.198                        # Parameter q
    T2: float = 0.87 * 1e-3                 # Relaxation time in s
    g_D: float = 0.00177                    # [pA]
    Sph: float = 96.0                       # Photon noise PSD [pA^2/Hz]
    w0: float = 2 * np.pi * 1e4            # Mean Larmor frequency
    h: float = 50 * 1e-9                    # Solver time step
    tf: float = 2.0                         # Final time in ms
    dc: float = 0.01                        # Diffusion constant of the OU process
    tau: float = 0.001                      # Time constant of the OU process
    measure_every_nth: int = 100            # Probing rate decimation factor
    sim_type: str = None                    # None, "OU", "jump", "sine"

    # --- Derived / Transformed Parameters (computed post-init) ---
    Nq: float = field(init=False)
    xc: float = field(init=False)
    yc: float = field(init=False)
    meas_probing_rate: float = field(init=False)
    sig_v: float = field(init=False)
    meas_probing_rate_unitless: float = field(init=False)
    w01: float = field(init=False)
    h1: float = field(init=False)
    x_0_sim: np.ndarray = field(init=False)

    def __post_init__(self):
        # --- Computations / Derivations with Physical Meanings ---
        
        # Nq represents the effective number of atoms (taking parameter q into account)
        self.Nq = self.N * self.q
        
        # xc is the scaling factor to convert physical angular momentum J to unitless J_unitless (J_unitless = J * xc)
        # It is related to the quantum projection noise scaling of the spin state
        self.xc = np.sqrt(2 / self.Nq)
        
        # yc is the scaling factor to convert physical sensor current (in pA) to unitless signal (z_unitless = z * yc)
        # yc is equal to xc / g_D
        self.yc = np.sqrt(2 / (self.g_D**2 * self.Nq))
        
        # Probing interval (measurement time step) in seconds
        self.meas_probing_rate = self.h * self.measure_every_nth
        
        # sig_v is the standard deviation of unitless measurement noise.
        # It converts physical photon noise PSD (Sph) over the probing interval into unitless noise strength.
        self.sig_v = np.sqrt(2 * self.Sph / (self.g_D**2 * self.Nq * self.meas_probing_rate))
        
        # Unitless probing interval (normalized by relaxation time T2)
        self.meas_probing_rate_unitless = self.meas_probing_rate / self.T2
        
        # Unitless Larmor frequency (normalized by relaxation time T2)
        self.w01 = self.w0 * self.T2
        
        # Unitless solver time step (normalized by relaxation time T2)
        self.h1 = self.h / self.T2
        
        # x_0_sim is the unitless initial state vector [Jy_unitless, Jz_unitless, omega_unitless]
        # Jz_unitless = Jz_physical * xc = (0.5 * N) * xc
        # omega_unitless = w0 * T2
        self.x_0_sim = np.array([0.0, np.sqrt(2 / (self.q * self.N)) * 0.5 * self.N, self.w0 * self.T2])

    def get_sim_params(self) -> UnitlessSimulationParams:
        return UnitlessSimulationParams(
            x_0=self.x_0_sim.copy(),
            dt=self.h1,
            T2=self.T2,
            w01=self.w01,
            dc=self.dc,
            tau=self.tau,
            type=self.sim_type,
            sig_v=self.sig_v
        )

    def get_filter_params(self, dw0: float = 0.01, dJ: float = 0.01, filter_dc: float = None) -> UnitlessFilterParams:
        if filter_dc is None:
            if self.sim_type == "jump":
                filter_dc = 1.0
            elif self.sim_type == "sine":
                filter_dc = 0.01
            else:
                filter_dc = self.dc
            
        if self.sim_type == "sine":
            dw0_val = 1.1
            P0_freq = (self.w0 * self.T2 / 3) ** 2
        else:
            dw0_val = dw0
            P0_freq = (self.w0 * self.T2 * dw0_val / 3) ** 2

        x0_filter = np.array([0.0, 0.5 * self.xc * self.N * (1 + dJ), self.w0 * (1 + dw0_val) * self.T2])
        P0_filter = np.diag([
            (0.5 * self.xc * self.N * dJ / 3) ** 2,
            (0.5 * self.xc * self.N * dJ / 3) ** 2,
            P0_freq
        ])
        Q_filter = np.diag([2.0, 2.0, filter_dc])

        if self.sim_type == "OU":
            factor = 200.0
        elif self.sim_type in ["sine", "jump"]:
            factor = 20.0
        else:
            factor = 1.0
            
        max_step = self.meas_probing_rate_unitless / (self.measure_every_nth * factor)

        return UnitlessFilterParams(
            x_0=x0_filter,
            t_0=0.0,
            dt=self.meas_probing_rate_unitless,
            inference_method='discrete',
            type=self.sim_type,
            tau=self.tau,
            w01=self.w01,
            P0=P0_filter,
            noise=FilterNoiseParams(Q=Q_filter),
            measurement=FilterMeasurementParams(
                dim_z=1,
                H=np.array([[0.0, 1.0, 0.0]]),
                measurement_strength=1.0,
                R=np.array([[self.sig_v ** 2]])
            ),
            max_step=max_step
        )
