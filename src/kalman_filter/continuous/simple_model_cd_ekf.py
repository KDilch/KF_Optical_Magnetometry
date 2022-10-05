import numpy as np
from scipy.integrate import odeint, simps
from kalman_filter.continuous.ekf import EKF
from abc import ABC, abstractmethod


class CD_EKF(EKF):
    def __init__(self, model_params):
        EKF.__init__(self, model_params)
        self.model_params = model_params
        self._R = self._R/self._dt
        self._Phi_delta = None
        self._Q_delta = None

    @staticmethod
    def F(x, t, model_params):
        return np.array([[-1/(model_params.T2), x[2], x[1]],
                         [-x[2], -1/(model_params.T2), -x[0]],
                         [0.0, 0.0, 0.0]])

    @staticmethod
    def fx(x_0, t, model_params):
        x = np.zeros(3)
        x[0] += - 1/(model_params.T2) * x_0[0] + x_0[1] * x_0[2]
        x[1] += - 1/(model_params.T2) * x_0[1] - x_0[0] * x_0[2]
        x[2] += 0
        return x

    @staticmethod
    def dx_dt(x, t, dim_x, model_params):
        return CD_EKF.fx(x, t, model_params)

    @staticmethod
    def dP_dt(P, t, x, Q, dim_x, model_params):
        return np.reshape(np.dot(CD_EKF.F(x, t, model_params),
                                 np.reshape(P, (dim_x, dim_x))) + np.dot(np.reshape(P, (dim_x, dim_x)),
                                                                         np.transpose(CD_EKF.F(x,
                                                                                               t,
                                                                                               model_params))) + Q, dim_x ** 2)

    def predict(self, method='default'):
        if method == 'default' or method == 'odeint':
            self.__predict_odeint()
        elif method == 'Q_delta':
            self.__predict_Q_delta()
        return

    def __predict_odeint(self):
        t = np.linspace(self._t, self._t + self._dt, num=20)  # times to report solution
        P = odeint(CD_EKF.dP_dt,
                   np.reshape(self._P, self._dim_x ** 2),
                   t,
                   args=(self._x,
                         self._Q,
                         self._dim_x,
                         self.model_params))[-1, :]
        x = odeint(CD_EKF.dx_dt,
                   self._x,
                   t,
                   args=(self._dim_x, self.model_params))[-1, :]
        self._P = np.reshape(P, (self._dim_x, self._dim_x))
        self._x = x
        return

    def __predict_Q_delta(self):
        self.compute_Phi_delta__Q_delta_odeint(t_0=self._t, num_terms=20)
        self._x = np.dot(self._Phi_delta, self._x)
        self._P = np.dot(np.dot(self._Phi_delta, self._P), self._Phi_delta.T) + self._Q_delta
        self._t += self._dt

    def update(self, z):
        self._z = z
        self._y = self._z - self._measurement_strength*np.dot(self._H, self._x) # innovation
        PHT = np.dot(self._P, self._H.T)
        # S = HPH' + R
        S = np.dot(self._H, PHT) + self._R
        SI = np.linalg.inv(S)
        # K = PH'inv(S)
        self._K = np.dot(PHT, SI)
        # x = x + Ky
        self._x = self._x + np.dot(self._K, self._y)
        I_KH = np.identity(self._dim_x) - np.dot(self._K, self._H)
        self._P = np.dot(np.dot(I_KH, self._P), I_KH.T) + np.dot(np.dot(self._K, self._R), self._K.T)

    def compute_Phi_delta__Q_delta_odeint(self, t_0, num_terms=20):
        Phi_0 = np.reshape(np.identity(self._dim_x), self._dim_x ** 2)  # initial Phi_delta is identity

        def dPhidt(Phi, t):
            return np.reshape(np.dot(self.F(self._x, t, self.model_params), np.reshape(Phi, (self._dim_x, self._dim_x))), self._dim_x**2)

        t = np.linspace(t_0, t_0 + self._dt, num=num_terms)  # times to report solution
        Phi_deltas, _ = odeint(dPhidt, np.reshape(Phi_0, self._dim_x**2), t, full_output=True)
        # Numerical
        Phi_s_matrix_form = [np.reshape(Phi_deltas[i], (self._dim_x, self._dim_x)) for i in range(len(Phi_deltas))]
        Phi_s_transpose_matrix_form = [np.transpose(a) for a in Phi_s_matrix_form]
        integrands = np.array([np.dot(np.dot(a, self._Q), b) for a, b in zip(Phi_s_matrix_form, Phi_s_transpose_matrix_form)])
        integrand_split = list(map(list, zip(*integrands.reshape(*integrands.shape[:1], -1))))
        # calculate integral numerically using simpsons rule
        self._Q_delta = np.reshape(np.array([simps(i, t) for i in integrand_split]), (self._dim_x, self._dim_x))
        self._Phi_delta = np.reshape(Phi_deltas[1], (self._dim_x, self._dim_x))

    def predict_update(self, z, method='default'):
        """ In continuous-discrete filter the equations for x and P in prediction step are solved numerically and then the appropriate
        correction is applied."""
        self.predict(method=method)
        self.update(z)
