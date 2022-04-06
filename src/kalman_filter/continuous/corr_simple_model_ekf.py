import numpy as np
import copy
from scipy.integrate import odeint

from kalman_filter.continuous.ekf import EKF


class CorrSimpleModelEKF(EKF):
    def __init__(self, model_params):
        EKF.__init__(self, model_params=model_params)
        self.model_params = model_params
        self._F = self.F(self._x, model_params)
        self._B = model_params.noise.B
        self._S = np.array([[], [], []])

    @staticmethod
    def F(x, model_params):
        return np.array([[-model_params.decoherence_x, x[2], x[1]],
                         [-x[2], -model_params.decoherence_y, -x[0]],
                         [0.0, 0.0, 0.0]])

    @staticmethod
    def fx(x_0, model_params):
        x = np.zeros(3)
        x[0] += - model_params.decoherence_x * x_0[0] + x_0[1] * x_0[2]
        x[1] += - model_params.decoherence_y * x_0[1] - x_0[0] * x_0[2]
        x[2] += 0
        return x

    @staticmethod
    def dx_dt(x, t, K, y, dt, model_params):
        return CorrSimpleModelEKF.fx(x, model_params) + np.dot(K, y) / dt

    @staticmethod
    def dP_dt(P, t, x, B, S, R_inv, H, Q, dim_x, model_params):
        P = np.reshape(P, (dim_x, dim_x))
        temp = CorrSimpleModelEKF.F(x, model_params) - np.dot(B, S).dot(R_inv).dot(H)
        P = np.dot(temp, P) + np.dot(P, temp.T) - np.dot(P, H.T).dot(R_inv).dot(H).dot(P) + B.dot(Q - S.dot(R_inv).dot(S.T)).dot(B.T)
        return np.reshape(P, dim_x ** 2)

    def predict_update(self, dz):
        self._dz = copy.deepcopy(dz)
        self._K = np.dot(np.dot(self._P, self._H.T) + np.dot(self._B, self._S), self._R_inv)
        self._y = dz - self._measurement_strength * np.dot(self._H, self._x) * self._dt
        t = np.linspace(self._t, self._t + self._dt, num=20)
        P = odeint(CorrSimpleModelEKF.dP_dt,
                   np.reshape(self._P, self._dim_x ** 2),
                   t,
                   args=(self._x,
                         self._B,
                         self._S,
                         self._R_inv,
                         self._H,
                         self._Q,
                         self._dim_x,
                         self.model_params))[1, :]
        x = odeint(CorrSimpleModelEKF.dx_dt, self._x, t, args=(self._K, self._y, self._dt, self.model_params))[-1, :]
        self._x = x
        # dP = self.dx_dt()
        self._P = np.reshape(P, (self._dim_x, self._dim_x))
        self._t += self._dt
        return
