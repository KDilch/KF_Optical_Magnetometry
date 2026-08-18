# -*- coding: utf-8 -*-
import numpy as np

from kalman_filter.cd_ckf import CD_CKF


class CD_CKF_unitless_magnetometer(CD_CKF):
    """
    Continuous-Discrete Cubature Kalman Filter for the unitless magnetometer model.
    Inherits from CD_CKF and implements discrete precession state transition dynamics.
    """
    def __init__(self, model_params):
        super().__init__(model_params)

    def propagate_state(self, x):
        """
        Analytical discrete-time state transition equivalent to MATLAB's dt_rhs.m.
        Calculates precession of spins J_y and J_z, and frequency w.
        """
        dt = self._dt
        omega = x[2]
        td = omega * dt
        c = np.cos(td)
        s = np.sin(td)
        e = np.exp(-dt)

        x_next = np.zeros(3)
        # Precession of spins
        x_next[0] = x[0] * e * c + x[1] * e * s
        x_next[1] = -x[0] * e * s + x[1] * e * c

        # Frequency dynamics (OU process vs constant)
        if self.model_params.type == "OU":
            w01 = self.model_params.w01
            tau = self.model_params.tau
            e_tau = np.exp(-dt / tau)
            x_next[2] = w01 + (omega - w01) * e_tau
        else:
            x_next[2] = omega

        return x_next
