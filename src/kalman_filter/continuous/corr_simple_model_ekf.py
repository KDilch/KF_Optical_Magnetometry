import numpy as np
import copy
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import solve_discrete_are
from kalman_filter.continuous.ekf import EKF


class CorrSimpleModelEKF(EKF):
    def __init__(self, model_params):
        EKF.__init__(self, model_params=model_params)
        self.model_params = model_params
        self._F = self.F(self._x, model_params)
        self._B = model_params.noise.B
        self._S = model_params.noise.S

    @staticmethod
    def F(x, model_params):
        return np.array([[-1/(model_params.T2), x[2], x[1]],
                         [-x[2], -1/(model_params.T2), -x[0]],
                         [0.0, 0.0, 0.0]])

    @staticmethod
    def fx(x_0, model_params):
        x = np.zeros(3)
        x[0] = - 1/(model_params.T2) * x_0[0] + x_0[1] * x_0[2]
        x[1] = - 1/(model_params.T2) * x_0[1] - x_0[0] * x_0[2]
        x[2] = 0
        return x

    @staticmethod
    def dx_dt(x, t, K, y, dt, model_params):
        return CorrSimpleModelEKF.fx(x, model_params) + np.dot(K, y) / dt

    @staticmethod
    def dP_dt(P, t, x, F, K, B, H, Q, dim_x, model_params):
        P = np.reshape(P, (dim_x, dim_x))
        dP = np.dot(F(x, model_params), P) +\
             np.dot(P, np.transpose(F(x, model_params))) -\
             np.dot(np.dot(K, H), P) + np.dot(B, Q).dot(B.T)
        #
        #
        #      np.reshape(np.dot(CorrSimpleModelEKF.F(x, model_params),
        #                       np.reshape(P, (dim_x, dim_x)))
        #                + np.dot(np.reshape(P, (dim_x, dim_x)),
        #                         np.transpose(CorrSimpleModelEKF.F(x, model_params)))
        #                - np.dot(np.dot(K, H), np.reshape(P, (dim_x, dim_x))) + np.dot(B, Q).dot(B.T), dim_x ** 2)
        return np.reshape(dP, dim_x ** 2)

    def predict_update(self, dz):
        self._dz = copy.deepcopy(dz)
        self._K = np.dot(np.dot(self._P, self._H.T) + np.dot(self._B, self._S.T), self._R_inv)
        self._y = dz - self._measurement_strength * np.dot(self._H, self._x) * self._dt
        t = np.linspace(self._t, self._t + self._dt, num=20)
        P = odeint(CorrSimpleModelEKF.dP_dt,
                   np.reshape(self._P, self._dim_x ** 2),
                   t,
                   args=(self._x,
                         self.F,
                         self._K,
                         self._B,
                         self._H,
                         self._Q,
                         self._dim_x,
                         self.model_params))[-1, :]
        # dP = np.dot(self.F(self._x, self.model_params), self._P)*self._dt +\
        #      np.dot(self._P, np.transpose(self.F(self._x, self.model_params)))*self._dt -\
        #      np.dot(np.dot(self._K, self._H), self._P)*self._dt + np.dot(self._B, self._Q).dot(self._B.T)*self._dt
        # dx = self.fx(self._x, self.model_params)*self._dt + np.dot(self._K, self._y)
        x = odeint(dx_dt, self._x, t, args=(self._K, self._y, self._dt, self.model_params))[-1, :]
        self._x = x
        self._P = np.reshape(P, (self._dim_x, self._dim_x))
        # self._x += dx
        # dP = CorrSimpleModelEKF.dP_dt(np.reshape(self._P, self._dim_x ** 2), self._t+self._dt,
        #                               self._x,
        #                               self.F,
        #                  self._K,
        #                  self._B,
        #                  self._H,
        #                  self._Q,
        #                  self._dim_x,
        #                  self.model_params)
        # self._P += np.reshape(dP, (self._dim_x, self._dim_x))*self._dt

def dx_dt(x, t, K, y, dt, model_params):
    return CorrSimpleModelEKF.fx(x, model_params) + np.dot(K, y) / dt