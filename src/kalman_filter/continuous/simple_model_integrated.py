import numpy as np
import copy
from kalman_filter.continuous.ekf import EKF


class MagnetometerEKFIntegrated(EKF):
    def __init__(self, model_params):
        EKF.__init__(self, model_params=model_params)
        self.model_params = model_params
        self._F = self.F(self._x, self._t, model_params)

    @staticmethod
    def F(x, t, model_params):
        return np.array([[0, 0, -np.exp(-model_params.decoherence_x*t)*(model_params.x_0[0])*np.sin(t*x[2])],
                         [0, 0, -np.exp(-model_params.decoherence_x*t)*(model_params.x_0[0])*np.cos(t*x[2])],
                         [0.0, 0.0, 1.0]])

    @staticmethod
    def fx(x, t, model_params):
        x_est = np.zeros(3)
        x_est[0] = np.exp(-model_params.decoherence_x*t)*(model_params.x_0[0])*np.cos(t*x[2])
        x_est[1] = -np.exp(-model_params.decoherence_x*t)*(model_params.x_0[0])*np.sin(t*x[2])
        x_est[2] = x[2]
        return x

    def dx_dt(self):
        dx_dt = np.zeros(3)
        dx_dt[0] = -self.model_params.decoherence_x*np.exp(-self.model_params.decoherence_x * self._t) * (self.model_params.x_0[0]) * np.cos(self._t * self._x[2])
        dx_dt[1] = self.model_params.decoherence_x*np.exp(-self.model_params.decoherence_x * self._t) * (self.model_params.x_0[0]) * np.sin(self._t * self._x[2])
        dx_dt[2] = 0
        return dx_dt

    def predict_update(self, dz):
        self._dz = copy.deepcopy(dz)
        self._K = np.dot(np.dot(self._P, self._H.T), np.linalg.inv(self._H.dot(self._P).dot(self._H.T)+self._R))
        P = self.F(self._x, self._t, self.model_params).dot((self._P-self._K.dot(self._H).dot(self._P))).dot(self._F.T)+self._Q
        self._x = MagnetometerEKFIntegrated.fx(self._x, self._t + self._dt, self.model_params)
        self._y = dz - self._measurement_strength * np.dot(self._H, self.dx_dt())*self._dt
        self._x = self._x + self._K.dot(self._y)
        self._P = P
        self._t += self._dt
        return
