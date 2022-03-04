#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import copy


class EKF(object):
    def __init__(self,
                 model_params):
        self._x = model_params.x_0
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

        self._z = np.array([None] * self._dim_z)

        self._K = np.zeros(self._x.shape)  # kalman gain
        self._R_inv = np.zeros((self._dim_z, self._dim_z))

        self._I = np.eye(self._dim_x)

        self._x_prior = self._x.copy()
        self._P_prior = self._P.copy()

        self._x_post = self._x.copy()
        self._P_post = self._P.copy()

    def predict_update(self, z):
        PHT = np.dot(self._P, self._H.T)
        self._R_inv = np.linalg.inv(self._R)
        self._K = np.dot(PHT, self._R_inv)
        self._y = z - self._measurement_strength*np.dot(self._H, self._x)

        self._x += np.dot(self._F, self._x)*self._dt+np.dot(self._K, self._y)*self._dt

        self._P += np.dot(self._F, self._P).dot(self._F.T)*self._dt + self._Q*self._dt-np.dot(np.dot(self._K, self._H), self._P)*self._dt

        self._z = copy.deepcopy(z)

    @property
    def x_est(self):
        return self._x

    @property
    def P_est(self):
        return self._P
