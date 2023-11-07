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
        return np.array([[x[3]*model_params.N/2*np.exp(-model_params.dt/model_params.T2)*np.cos(x[2]*model_params.dt),
                          -x[3]*model_params.N/2*np.exp(-model_params.dt/model_params.T2)*np.sin(x[2]*model_params.dt),
                          x[3]*model_params.N/2*np.exp(-model_params.dt/model_params.T2)*(-x[0]*np.sin(model_params.dt*x[2])*model_params.dt-x[1]*np.cos(model_params.dt*x[2])*model_params.dt),
                          model_params.N/2*np.exp(-model_params.dt/model_params.T2)*(x[0]*np.cos(x[2]*model_params.dt)-x[1]*np.sin(x[2]*model_params.dt))],
                         [x[3]*model_params.N/2*np.exp(-model_params.dt/model_params.T2)*np.sin(x[2]*model_params.dt),
                          x[3]*model_params.N/2*np.exp(-model_params.dt/model_params.T2)*np.cos(x[2]*model_params.dt),
                          x[3]*model_params.N/2*np.exp(-model_params.dt/model_params.T2)*(-x[1]*np.sin(model_params.dt*x[2])*model_params.dt+x[0]*np.cos(model_params.dt*x[2])*model_params.dt),
                          model_params.N/2*np.exp(-model_params.dt/model_params.T2)*(x[1]*np.cos(model_params.dt *x[2]) + x[0]*np.sin(model_params.dt*x[2]))],
                         [0.0,
                          0.0,
                          1.0,
                          0.0],
                         [0.0, 0.0, 0.0, 1.0]])

    @staticmethod
    def fx(x, t, model_params):
        """
        :param x: [J_x, J_y, omega]
        :param t:
        :param model_params:
        :return:
        """
        x_est = np.zeros(4)
        x_est[0] = model_params.N/2*np.exp(-(1/model_params.T2)*model_params.dt)*(x[0]*np.cos(model_params.dt*x[2])-x[1]*np.sin(model_params.dt*x[2]))
        x_est[1] = model_params.N/2*np.exp(-(1/model_params.T2)*model_params.dt)*(x[1]*np.cos(model_params.dt*x[2])+x[0]*np.sin(model_params.dt*x[2]))
        x_est[2] = x[2]
        x_est[3] = x[3]
        return x

    def predict_update(self, dz_dt):
        self._dz = copy.deepcopy(dz_dt)
        self._F = self.F(self._x, self._t, self.model_params)
        self._K = np.dot(np.dot(self._P, self._H.T), np.linalg.inv(self._H.dot(self._P).dot(self._H.T)+self._R))
        P = self._F.dot((self._P-self._K.dot(self._H).dot(self._P))).dot(self._F.T)+self._Q
        x = MagnetometerEKFIntegrated.fx(self._x, self._t + self._dt, self.model_params)
        self._P = P
        self._x = x
        self._y = dz_dt - self._measurement_strength * np.dot(self._H, self._x)  # innovation
        self._x += self._K.dot(self._y)
        self._t += self._dt
        return
