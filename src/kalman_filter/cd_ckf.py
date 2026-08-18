#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class CD_CKF(object):
    """
    Continuous-Discrete Cubature Kalman Filter (CD-CKF) Base Class.
    Implements a 3rd-degree spherical-radial Cubature Kalman Filter.
    """
    def __init__(self, model_params):
        self.model_params = model_params
        self._x = model_params.x_0.copy()
        self._t = model_params.t_0
        self._dt = model_params.dt
        self._dim_x = len(self._x)
        self._dim_z = model_params.measurement.dim_z
        self._H = model_params.measurement.H
        self._R = model_params.measurement.R
        self._Q = model_params.noise.Q
        self._P = model_params.P0.copy()
        
        # Spherical-radial cubature points and weights setup
        self.n = self._dim_x
        self.wghts = np.ones(2 * self.n) / (2 * self.n)
        # pts has shape (n, 2n), consisting of positive and negative coordinate axes scaled by sqrt(n)
        self.pts = np.sqrt(self.n) * np.block([np.eye(self.n), -np.eye(self.n)])

    @property
    def x_est(self):
        return self._x

    @property
    def P_est(self):
        return self._P

    @property
    def t(self):
        return self._t

    def propagate_state(self, x):
        """
        Discrete-time state transition function x_{k} = f(x_{k-1}).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Implement propagate_state in a subclass.")

    def predict(self):
        """
        Performs the CKF prediction step by propagating cubature points.
        """
        # 1. Compute Cholesky lower factor S_05 of current covariance P: P = S_05 @ S_05.T
        try:
            S_05 = np.linalg.cholesky(self._P)
        except np.linalg.LinAlgError:
            # Fall back to spectral decomposition if P is slightly non-positive definite
            vals, vecs = np.linalg.eigh(self._P)
            vals = np.maximum(vals, 1e-12)
            S_05 = vecs @ np.diag(np.sqrt(vals))

        # 2. Generate cubature points
        # chi has shape (n, 2n)
        chi = self._x[:, np.newaxis] + S_05 @ self.pts

        # 3. Propagate cubature points through discrete transition function
        chi_star = np.zeros_like(chi)
        for k in range(2 * self.n):
            chi_star[:, k] = self.propagate_state(chi[:, k])

        # 4. Compute predicted mean
        # self.wghts has shape (2n,), self.wghts[np.newaxis, :] has shape (1, 2n)
        x_pred = np.sum(self.wghts[np.newaxis, :] * chi_star, axis=1)

        # 5. Compute predicted covariance
        P_pred = np.zeros_like(self._P)
        for k in range(2 * self.n):
            diff = (chi_star[:, k] - x_pred)[:, np.newaxis]
            P_pred += self.wghts[k] * (diff @ diff.T)

        # Add discrete process noise Qd = Q * dt
        Qd = self._Q * self._dt
        P_pred += Qd

        # Save predictions and enforce symmetry
        self._x = x_pred
        self._P = 0.5 * (P_pred + P_pred.T)
        self._t += self._dt

    def update(self, z):
        """
        Performs the CKF correction/update step using the measurement z.
        """
        H = self._H
        R = self._R
        P = self._P

        # Innovation covariance S = H @ P @ H.T + R
        S = H @ P @ H.T + R
        S_val = S[0, 0]

        # Kalman gain K = P @ H.T / S
        K = (P @ H.T) / S_val

        # State update
        innovation = np.array([z]) - H @ self._x
        x_new = self._x + (K @ innovation).flatten()

        # Enforce frequency boundaries on omega matching Matlab limits
        # omega is x_new[2]
        x_new[2] = np.clip(x_new[2], 0.0, 110.0)

        # Covariance update: P_new = (I - K @ H) @ P
        P_new = (np.eye(self.n) - K @ H) @ P
        P_new = 0.5 * (P_new + P_new.T)

        self._x = x_new
        self._P = P_new
