# -*- coding: utf-8 -*-
from dataclasses import dataclass
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

class UnitlessSimpleMagnetometerConfigurator:
    def __init__(self,
                 N: float = 0.44 * 1e12,
                 q: float = 0.198,
                 T2: float = 0.87 * 1e-3,
                 g_D: float = 0.00177,
                 Sph: float = 96.0,
                 w0: float = 2 * np.pi * 1e4,
                 h: float = 50 * 1e-9,
                 tf: float = 2.0,
                 dc: float = 0.01,
                 tau: float = 0.001,
                 measure_every_nth: int = 100,
                 sim_type: str = None):
        self.N = N
        self.q = q
        self.T2 = T2
        self.g_D = g_D
        self.Sph = Sph
        self.w0 = w0
        self.h = h
        self.tf = tf
        self.dc = dc
        self.tau = tau
        self.measure_every_nth = measure_every_nth
        self.sim_type = sim_type

        # --- Computations / Derivations with Physical Meanings ---
        
        # Nq represents the effective number of atoms (taking parameter q into account)
        self.Nq = N * q
        
        # xc is the scaling factor to convert physical angular momentum J to unitless J_unitless (J_unitless = J * xc)
        # It is related to the quantum projection noise scaling of the spin state
        self.xc = np.sqrt(2 / self.Nq)
        
        # yc is the scaling factor to convert physical sensor current (in pA) to unitless signal (z_unitless = z * yc)
        # yc is equal to xc / g_D
        self.yc = np.sqrt(2 / (g_D**2 * self.Nq))
        
        # Probing interval (measurement time step) in seconds
        self.meas_probing_rate = h * measure_every_nth
        
        # sig_v is the standard deviation of unitless measurement noise.
        # It converts physical photon noise PSD (Sph) over the probing interval into unitless noise strength.
        self.sig_v = np.sqrt(2 * Sph / (g_D**2 * self.Nq * self.meas_probing_rate))
        
        # Unitless probing interval (normalized by relaxation time T2)
        self.meas_probing_rate_unitless = self.meas_probing_rate / T2
        
        # Unitless Larmor frequency (normalized by relaxation time T2)
        self.w01 = w0 * T2
        
        # Unitless solver time step (normalized by relaxation time T2)
        self.h1 = h / T2
        
        # x_0_sim is the unitless initial state vector [Jy_unitless, Jz_unitless, omega_unitless]
        # Jz_unitless = Jz_physical * xc = (0.5 * N) * xc
        # omega_unitless = w0 * T2
        self.x_0_sim = np.array([0.0, np.sqrt(2 / (q * N)) * 0.5 * N, w0 * T2])

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
            filter_dc = 0.01 if self.sim_type in ["sine", "jump"] else self.dc
            
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

        return UnitlessFilterParams(
            x_0=x0_filter,
            t_0=0.0,
            dt=self.meas_probing_rate_unitless,
            inference_method='RK45',
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
            )
        )
