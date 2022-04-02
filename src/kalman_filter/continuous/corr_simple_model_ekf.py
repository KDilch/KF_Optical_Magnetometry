import numpy as np
import copy
from scipy.integrate import odeint

from kalman_filter.continuous.ekf import EKF


class CorrSimpleModelEKF(EKF):
    def __init__(self, model_params):
        EKF.__init__(self, model_params=model_params)
        self.model_params = model_params
        self._F = self.F(self._x)
        self._B = model_params.noise.B
        self._S = model_params.noise.S

    @staticmethod
    def F(x):
        return np.array([[-0., x[2], x[1]],
                         [-x[2], -0., -x[0]],
                         [0.0, 0.0, 0.0]])

    @staticmethod
    def fx(x_0):
        x = np.zeros(3)
        x[0] += - 0.0 * x_0[0] + x_0[1] * x_0[2]
        x[1] += -0.0 * x_0[1] - x_0[0] * x_0[2]
        x[2] += 0
        return x

    @staticmethod
    def dx_dt(x, t, K, y, dt):
        return CorrSimpleModelEKF.fx(x) + np.dot(K, y)/dt

    @staticmethod
    def dP_dt(P, t, x, K, H, Q, dim_x):
        return np.reshape(np.dot(CorrSimpleModelEKF.F(x),
                                 np.reshape(P, (dim_x, dim_x))) + np.dot(np.reshape(P, (dim_x, dim_x)),
                                                                         np.transpose(
                                                                             CorrSimpleModelEKF.F(x))) - np.dot(
            np.dot(K, H), np.reshape(P, (dim_x, dim_x))) + Q, dim_x ** 2)

    def predict_update(self, dz):
        self._dz = copy.deepcopy(dz)
        self._K = np.dot(np.dot(self._P, self._H.T) + np.dot(self._B, self._S.T), self._R_inv)
        self._y = dz - self._measurement_strength * np.dot(self._H, self._x) * self._dt
        # P = odeint(MagnetometerEKF.dP_dt,
        #            np.reshape(self._P, self._dim_x**2),
        #            np.linspace(self._t, self._t+self._dt, 40),
        #            args=(self._x, self._K, self._H, self._Q, self._dim_x))[1, :]
        t = np.linspace(self._t, self._t + self._dt, num=20)  # times to report solution
        x = odeint(CorrSimpleModelEKF.dx_dt, self._x, t, args=(self._K, self._y, self._dt))[-1, :]
        temp = self.F(self._x) - np.dot(self._B, self._S).dot(self._R_inv).dot(self._H)
        dP = np.dot(temp, self._P) * self._dt + np.dot(self._P, temp) * self._dt - \
             np.dot(self._P, self._H.T).dot(self._R_inv).dot(self._H).dot(self._P) * self._dt + \
             self._B.dot(self._Q - self._S.dot(self._R_inv).dot(self._S.T)).dot(self._B.T) * self._dt
        self._x = x
        self._P += dP
        self._t += self._dt
        return
