#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import copy
from scipy.integrate import odeint


class Corr_CD_EKF(object):
    def __init__(self,
                 model_params):
        self._x = model_params.x_0
        self._t = model_params.t_0
        self._dt = model_params.dt
        self._measurement_strength = model_params.measurement.measurement_strength
        self._dim_x = len(self._x)
        self._dim_z = model_params.measurement.dim_z
        self._F = np.eye(self._dim_x)  # linearized space_state_model
        self._H = model_params.measurement.H
        self._R = model_params.measurement.R
        self._Q = model_params.noise.Q
        self._P = model_params.P0
        self._y = np.zeros((self._dim_z, 1))  # residual
        self._dz = np.array([model_params.x_0[1]] * self._dim_z)
        self._K = np.zeros(self._x.shape)  # kalman gain
        self._R_inv = np.linalg.inv(self._R)

    @staticmethod
    def F(x, model_params):
        raise NotImplementedError('Implement F function.')

    @staticmethod
    def fx(x_0, model_params):
        raise NotImplementedError('Implement fx function.')

    def predict_update(self, dz):
        self._dz = copy.deepcopy(dz)
        self._K = np.dot(np.dot(self._P, self._H.T) + np.dot(self._B, self._S.T), self._R_inv)
        self._y = dz - self._measurement_strength * np.dot(self._H, self._x) * self._dt
        t = np.linspace(self._t, self._t + self._dt, num=20)
        P = odeint(Corr_CD_EKF.dP_dt,
                   np.reshape(self._P, self._dim_x ** 2),
                   t,
                   args=(self._x,
                         self.F,
                         self._K,
                         self._B,
                         self._H,
                         self._Q,
                         self._dim_x,
                         self.model_params))[-1, :]
        # dP = np.dot(self.F(self._x, self.model_params), self._P)*self._dt +\
        #      np.dot(self._P, np.transpose(self.F(self._x, self.model_params)))*self._dt -\
        #      np.dot(np.dot(self._K, self._H), self._P)*self._dt + np.dot(self._B, self._Q).dot(self._B.T)*self._dt
        # dx = self.fx(self._x, self.model_params)*self._dt + np.dot(self._K, self._y)
        x = odeint(dx_dt, self._x, t, args=(self._K, self._y, self._dt, self.model_params))[-1, :]
        self._x = x
        self._P = np.reshape(P, (self._dim_x, self._dim_x))
        # self._x += dx
        # dP = CorrSimpleModelEKF.dP_dt(np.reshape(self._P, self._dim_x ** 2), self._t+self._dt,
        #                               self._x,
        #                               self.F,
        #                  self._K,
        #                  self._B,
        #                  self._H,
        #                  self._Q,
        #                  self._dim_x,
        #                  self.model_params)
        # self._P += np.reshape(dP, (self._dim_x, self._dim_x))*self._dt

        self._t += self._dt

    @property
    def x_est(self):
        return self._x

    @property
    def P_est(self):
        return self._P
