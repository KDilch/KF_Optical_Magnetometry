#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import copy
from scipy.linalg import solve_continuous_are
from scipy.integrate import odeint


class EKF(object):
    def __init__(self,
                 model_params):
        self._x = model_params.x_0
        self._t = model_params.t_0
        self._dt = model_params.dt
        self._measurement_strength = model_params.measurement.measurement_strength
        self._dim_x = len(self._x)
        self._dim_z = model_params.measurement.dim_z
        self._F = np.eye(self._dim_x)  # linearized dynamics
        self._H = model_params.measurement.H
        self._R = model_params.measurement.R
        self._Q = model_params.noise.Q
        self._P = model_params.P0
        self._y = np.zeros((self._dim_z, 1))  # residual

        self._dz = np.array([model_params.x_0[1]] * self._dim_z)

        self._K = np.zeros(self._x.shape)  # kalman gain
        self._R_inv = np.linalg.inv(self._R)

    def F(self):
        pass

    def fx(self):
        pass

    def predict_update(self, dz):
        self._dz = copy.deepcopy(dz)

        self._K = np.dot(np.dot(self._P, self._H.T), self._R_inv)
        self._y = dz - self._measurement_strength*np.dot(self._H, self._x)*self._dt

        # self._x = odeint(dx_dt, self._x, np.linspace(self._t, self._t+self._dt, 20), args=(self,))[-1, :]
        # print(np.dot(self._K, self._y)/self._dt)
        dx = self.fx() + np.dot(self._K, self._y)
        dP = np.dot(self.F(), self._P)*self._dt+np.dot(self._P, np.transpose(self.F()))*self._dt-np.dot(np.dot(self._K, self._H), self._P)*self._dt + self._Q*self._dt
        self._x += dx
        self._P += dP
        self._t += self._dt
        return

    @property
    def x_est(self):
        return self._x

    @property
    def P_est(self):
        return self._P


def dx_dt(x, t, self):
    return np.dot(self._F, x) + np.dot(self._K, self._y)/self._dt