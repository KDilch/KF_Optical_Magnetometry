# -*- coding: utf-8 -*-
import numpy as np

from kalman_filter.cd_ekf import CD_EKF


class CD_EKF_unitless_magnetometer(CD_EKF):
    """
    Continuous-Discrete Extended Kalman Filter for the unitless magnetometer model.
    Inherits from kalman_filter.cd_ekf.CD_EKF.
    """
    def __init__(self, model_params):
        CD_EKF.__init__(self, model_params)

    @staticmethod
    def F(x, t, model_params):
        """
        Jacobian of the drift function f(x, t).
        """
        sim_type = model_params.type
        if sim_type == "OU":
            tau = model_params.tau
            return np.array([
                [-1.0, x[2], x[1]],
                [-x[2], -1.0, -x[0]],
                [0.0, 0.0, -1.0 / tau]
            ])
        else:
            return np.array([
                [-1.0, x[2], x[1]],
                [-x[2], -1.0, -x[0]],
                [0.0, 0.0, 0.0]
            ])

    @staticmethod
    def fx(x_0, t, model_params):
        """
        Drift function f(x_0, t).
        """
        sim_type = model_params.type
        x = np.zeros(3)
        x[0] = -x_0[0] + x_0[2] * x_0[1]
        x[1] = -x_0[2] * x_0[0] - x_0[1]
        if sim_type == "OU":
            w01 = model_params.w01
            tau = model_params.tau
            x[2] = (w01 - x_0[2]) / tau
        else:
            x[2] = 0.0
        return x

    def predict(self):
        method = self.model_params.inference_method
        if method == 'discrete':
            dt = self._dt
            x = self._x
            omega = x[2]
            tau = self.model_params.tau
            
            # State propagation using RK4
            self._x = self.rk4_step(self.fx, self._x, self._t, dt, self.model_params, self.__class__)
            
            # Covariance propagation using exact analytical Jacobian Ad for the precessing system
            c = np.cos(omega * dt)
            s = np.sin(omega * dt)
            e = np.exp(-dt)
            e_tau = np.exp(-dt / tau) if self.model_params.type == "OU" else 1.0
            
            # Note: the derivatives with respect to omega correctly account for precession dynamics
            Ad = np.array([
                [c * e, s * e, (-x[0] * s + x[1] * c) * dt * e],
                [-s * e, c * e, (-x[0] * c - x[1] * s) * dt * e],
                [0.0, 0.0, e_tau]
            ])
            
            Qd = self._Q * dt
            
            self._P = Ad @ self._P @ Ad.T + Qd
            self._P = 0.5 * (self._P + self._P.T)
            self._t += dt
            return
            
        # Fall back to parent class CD_EKF predict for continuous solvers
        super().predict()
