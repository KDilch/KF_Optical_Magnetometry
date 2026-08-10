import numpy as np
import copy
from filterpy.kalman.UKF import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints


class SimpleMagnetometerUKF(object):
    def __init__(self, model_params):
        self._dim_x = len(model_params.x_0)
        self._dim_z = 1
        self.H = model_params.measurement.H
        self._x = model_params.x_0
        self._P = model_params.P0
        self.sigma_points_generator = MerweScaledSigmaPoints(n=self._dim_x, alpha=1e-3, beta=2., kappa=3-self._dim_x)
        self.ukf = UnscentedKalmanFilter(self._dim_x, self._dim_z, model_params.dt, self.hx, self.fx, self.sigma_points_generator)
        self.ukf.x = self._x.copy()
        self.ukf.P = self._P.copy()
        self.ukf.Q = model_params.noise.Q.copy()
        self.ukf.R = model_params.measurement.R.copy()
        self.model_params = model_params

    @property
    def x_est(self):
        return self._x

    @property
    def P_est(self):
        return self._P

    def predict_update(self, z, compute_ss=False):
        self.ukf.predict(model_params=self.model_params)
        self.ukf.update(z, H=self.H)
        self._x = self.ukf.x
        self._P = self.ukf.P

    @staticmethod
    def fx(x, dt, model_params):
        x_next = np.zeros(3)
        x_next[0] = x[0] + (- (1/model_params.T2) * x[0] + x[1] * x[2]) * dt
        x_next[1] = x[1] + (- (1/model_params.T2) * x[1] - x[0] * x[2]) * dt
        x_next[2] = x[2]
        return x_next

    @staticmethod
    def hx(x, H):
        return H.dot(x)

