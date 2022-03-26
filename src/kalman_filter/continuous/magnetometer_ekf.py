import numpy as np

from kalman_filter.continuous.ekf import EKF


class MagnetometerEKF(EKF):
    def __init__(self, model_params):
        EKF.__init__(self, model_params=model_params)
        self.model_params = model_params
        self._F = self.F()

    def F(self):
        return np.array([[-0., self._x[2], self._x[1]],
                        [-self._x[2], -0., -self._x[0]],
                        [0.0, 0.0, 0.0]])

    def fx(self):
        x = np.zeros(3)
        x[0] += - 0.0 * self._x[0] * self._dt + self._x[1] * self._x[2] * self._dt
        x[1] += -0.0 * self._x[1] * self._dt - self._x[0] * self._x[2] * self._dt
        x[2] += 0
        return x

