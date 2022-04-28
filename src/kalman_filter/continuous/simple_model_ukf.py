import numpy as np
import copy
from filterpy.kalman.UKF import UnscentedKalmanFilter
from filterpy.kalman import JulierSigmaPoints


class SimpleMagnetometerUKF(object):
    def __init__(self, model_params):
        self._dim_x = len(model_params.x_0)
        self._dim_z = 1
        self.H = model_params.measurement.H
        self._x = model_params.x_0
        self.sigmas = JulierSigmaPoints(n=self._dim_x)
        self.ukf = UnscentedKalmanFilter(self._dim_x, self._dim_z, model_params.dt, self.hx, self.fx, self.sigmas)
        self.model_params = model_params

    @staticmethod
    def fx(x, dt, model_params):
        x = np.zeros(3)
        x[0] += - model_params.decoherence_x * x[0] + x[1] * x[2]
        x[1] += - model_params.decoherence_y * x[1] - x[0] * x[2]
        x[2] += 0
        return x

    @staticmethod
    def hx(x, H):
        return H.dot(x)
