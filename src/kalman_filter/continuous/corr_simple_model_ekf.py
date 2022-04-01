import numpy as np
from copy import copy

from kalman_filter.continuous.ekf import EKF


class CorrSimpleModelEKF(EKF):
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
