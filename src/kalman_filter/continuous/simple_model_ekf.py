import numpy as np
import copy
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from kalman_filter.continuous.ekf import EKF


class MagnetometerEKF(EKF):
    def __init__(self, model_params):
        EKF.__init__(self, model_params=model_params)
        self.model_params = model_params
        self._F = self.F(self._x, self._t, model_params)
        self._steady_cov = None

    @property
    def steady_cov(self):
        return self._steady_cov
    @staticmethod
    def F(x, t, model_params):
        return np.array([[-1./(model_params.T2), x[2], x[1]],
                         [-x[2], -1./model_params.T2, -x[0]],
                         [0.0, 0.0, 0.0]])

    def steady_state(self):
        steady_cov = solve_continuous_are(a=np.transpose(self._F),
                                          b=np.transpose(self._H),
                                          q=self._Q,
                                          r=self._R)

        return steady_cov

    @staticmethod
    def fx(x_0, t, model_params):
        dx_dt = np.zeros(3)
        dx_dt[0] = - (1/model_params.T2) * x_0[0] + x_0[1] * x_0[2]
        dx_dt[1] = - (1/model_params.T2) * x_0[1] - x_0[0] * x_0[2]
        dx_dt[2] = 0
        return dx_dt

    @staticmethod
    def dx_dt(t, x, K, y, dt, model_params):
        return MagnetometerEKF.fx(x, t, model_params) + np.dot(K, y) / dt

    @staticmethod
    def dP_dt(t, P, x, K, H, Q, dim_x, model_params):
        return np.reshape(np.dot(MagnetometerEKF.F(x, t, model_params),
                                 np.reshape(P, (dim_x, dim_x))) + np.dot(np.reshape(P, (dim_x, dim_x)),
                                                                         np.transpose(MagnetometerEKF.F(x,
                                                                                                        t,
                                                                                                        model_params))) - np.dot(
            np.dot(K, H*model_params.measurement.measurement_strength), np.reshape(P, (dim_x, dim_x))) + Q, dim_x ** 2)

    def predict_update(self, dz, compute_ss=False):
        self._dz = copy.deepcopy(dz)
        self._K = np.dot(np.dot(self._P, self._measurement_strength*self._H.T), self._R_inv)
        self._y = dz - self._measurement_strength * np.dot(self._H, self._x) * self._dt
        P_sol = solve_ivp(MagnetometerEKF.dP_dt,
                      [self._t, self._t + self._dt],
                      np.reshape(self._P, self._dim_x**2),
                      method=self.model_params.inference_method,
                      dense_output=True,
                       args=(self._x,
                             self._K,
                                  self._H,
                                  self._Q,
                                  self._dim_x,
                                  self.model_params))
        P = P_sol.sol(self._t+self._dt)
        x_sol = solve_ivp(MagnetometerEKF.dx_dt,
                          [self._t, self._t + self._dt],
                          self._x,
                          method=self.model_params.inference_method,
                          dense_output=True,
                          args=(self._K,
                                self._y,
                                self._dt,
                                self.model_params))
        x = x_sol.sol(self._t + self._dt)
        self._x = x
        self._P = np.reshape(P, (self._dim_x, self._dim_x))
        if compute_ss:
            self._steady_cov = self.steady_state()
        self._t += self._dt
        return
