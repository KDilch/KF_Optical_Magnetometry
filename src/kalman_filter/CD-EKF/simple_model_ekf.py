import numpy as np
import copy
from scipy.integrate import odeint
from kalman_filter.continuous.ekf import EKF


class SimpleModelEKF(EKF):
    def __init__(self, model_params):
        EKF.__init__(self, model_params=model_params)
        self.model_params = model_params
        self._F = self.F(self._x, model_params)

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
        return SimpleModelEKF.fx(x, model_params) + np.dot(K, y) / dt

    @staticmethod
    def dP_dt(P, t, x, K, H, Q, dim_x, model_params):
        return np.reshape(np.dot(SimpleModelEKF.F(x, model_params),
                                 np.reshape(P, (dim_x, dim_x))) + np.dot(np.reshape(P, (dim_x, dim_x)),
                                                                         np.transpose(SimpleModelEKF.F(x,
                                                                                                        model_params))) + Q, dim_x ** 2)

    def predict_update(self, dz):
        self._dz = copy.deepcopy(dz)
        self._K = np.dot(np.dot(self._P, self._H.T), self._R_inv)
        self._y = dz - self._measurement_strength * np.dot(self._H, self._x) * self._dt
        P = odeint(SimpleModelEKF.dP_dt,
                   np.reshape(self._P, self._dim_x ** 2),
                   np.linspace(self._t, self._t + self._dt, 20),
                   args=(self._x,
                         self._K,
                         self._H,
                         self._Q,
                         self._dim_x,
                         self.model_params))[-1, :]
        t = np.linspace(self._t, self._t + self._dt, num=20)  # times to report solution
        x = odeint(SimpleModelEKF.dx_dt, self._x, t, args=(self._K,
                                                            self._y,
                                                            self._dt,
                                                            self.model_params))[-1, :]
        self._x = x
        self._P = np.reshape(P, (self._dim_x, self._dim_x))
        R_k = self._R + self._H.dot(self._P).dot(self._H.T)
        K_k = self._P.dot(self._H.T).dot(self._R_inv)
        self._t += self._dt
        return
