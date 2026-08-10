#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are


class CD_EKF(object):
    def __init__(self,
                 model_params):
        self.model_params = model_params
        self._x = model_params.x_0
        self._t = model_params.t_0
        self._dt = model_params.dt
        self._dim_x = len(self._x)
        self._dim_z = model_params.measurement.dim_z
        self._F = np.eye(self._dim_x)
        self._H = model_params.measurement.H
        self._measurement_strength = model_params.measurement.measurement_strength
        self._R = model_params.measurement.R  # Remember this R should be R_delta!!!
        self._Q = model_params.noise.Q
        self._P = model_params.P0
        self._y = np.zeros(self._dim_z)

        self._z = None  # the most recent measurement outcome

        self._K = np.zeros(shape=(self._x.shape[0], 1))  # kalman gain
        self._R_inv = np.linalg.inv(self._R)
        self._steady_cov = None

    @staticmethod
    def F(x, t, model_params):
        raise NotImplementedError('Implement F function (jacobian).')

    @staticmethod
    def fx(x_0, t, model_params):
        raise NotImplementedError('Implement fx function.')

    @staticmethod
    def dx_dt(t, x, dim_x, model_params, cls=None):
        cls = cls or CD_EKF
        return cls.fx(x, t, model_params)

    @staticmethod
    def dP_dt(t, P, x, Q, dim_x, model_params, cls=None):
        cls = cls or CD_EKF
        P_matrix = np.reshape(P, (dim_x, dim_x))
        return np.reshape(cls.F(x, t, model_params) @ P_matrix + P_matrix @ cls.F(x, t, model_params).T + Q,
                          dim_x ** 2)

    def predict(self):
        P_sol = solve_ivp(CD_EKF.dP_dt,
                          [self._t, self._t + self._dt],
                          np.reshape(self._P, self._dim_x ** 2),
                          method=self.model_params.inference_method,
                          dense_output=True,
                          args=(self._x,
                                self._Q,
                                self._dim_x,
                                self.model_params,
                                self.__class__))
        P = P_sol.sol(self._t + self._dt)
        x_sol = solve_ivp(CD_EKF.dx_dt,
                          [self._t, self._t + self._dt],
                          self._x,
                          method=self.model_params.inference_method,
                          dense_output=True,
                          args=(self._dim_x, self.model_params, self.__class__))
        x = x_sol.sol(self._t + self._dt)
        self._x = x
        self._P = np.reshape(P, (self._dim_x, self._dim_x))
        self._t += self._dt

    def update(self, z):
        self._z = z
        self._y = self._z - np.dot(self._H, self._x)  # innovation
        PHT = np.dot(self._P, self._H.T)
        S = np.dot(self._H, PHT) + self._R
        S_INV = np.linalg.inv(S)
        # K = PH'inv(S)
        self._K = np.dot(PHT, S_INV)
        # x = x + Ky
        self._x = self._x + np.dot(self._K, self._y)
        I_KH = np.identity(self._dim_x) - np.dot(self._K, self._H)
        self._P = np.dot(np.dot(I_KH, self._P), I_KH.T) + np.dot(np.dot(self._K, self._R), self._K.T)

    @property
    def x_est(self):
        return self._x

    @property
    def P_est(self):
        return self._P

    @property
    def steady_cov(self):
        return self._steady_cov

    @steady_cov.setter
    def steady_cov(self, value):
        self._steady_cov = value

    def steady_state(self):
        self.steady_cov = solve_continuous_are(a=np.transpose(self._F),
                                               b=np.transpose(self._H),
                                               q=self._Q,
                                               r=self._R * self._dt)
        return self.steady_cov

    def predict_update(self, z, calculate_ss=False):
        """ In continuous-discrete filter the equations for x and P in prediction step are solved numerically
        and then the appropriate correction is applied."""
        self.predict()
        self.update(z)
        if calculate_ss:
            self.steady_state()
