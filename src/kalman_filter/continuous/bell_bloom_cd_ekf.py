import numpy as np
from scipy.integrate import solve_ivp, simps
from scipy.linalg import solve_discrete_are, solve_continuous_are

from kalman_filter.continuous.ekf import EKF


class CD_Bell_Bloom_EKF(EKF):
    def __init__(self, model_params):
        EKF.__init__(self, model_params)
        self.model_params = model_params
        self._R_delta = self._R/self._dt
        self._Phi_delta = None
        self._Q_delta = None
        self.steady_cov = None

    @staticmethod
    def F(x, t, model_params):
        # Jacobian
        return np.array([[-1/model_params.T2, x[2], x[1]],
                         [-x[2], -1/model_params.T2-model_params.pump_rate, -x[0]],
                         [0.0, 0.0, 0.0]])

    @staticmethod
    def fx(x_0, t, model_params):
        x = np.zeros(3)
        x[0] += - (1/model_params.T2) * x_0[0] + x_0[1] * x_0[2]
        x[1] += - (1/model_params.T2) * x_0[1] - x_0[0] * x_0[2] + model_params.pump_rate*(model_params.num_atoms/2 - x_0[1])
        x[2] += 0
        return x

    @staticmethod
    def dx_dt(t, x, dim_x, model_params):
        return CD_Bell_Bloom_EKF.fx(x, t, model_params)

    @staticmethod
    def dP_dt(t, P, x, Q, dim_x, model_params):
        return np.reshape(np.dot(CD_Bell_Bloom_EKF.F(x, t, model_params),
                                 np.reshape(P, (dim_x, dim_x))) + np.dot(np.reshape(P, (dim_x, dim_x)),
                                                                         np.transpose(CD_Bell_Bloom_EKF.F(x,
                                                                                               t,
                                                                                               model_params))) + Q,
                          dim_x ** 2)

    def predict(self, Phi_Q_method=False):
        P_sol = solve_ivp(CD_Bell_Bloom_EKF.dP_dt,
                          [self._t, self._t + self._dt],
                          np.reshape(self._P, self._dim_x ** 2),
                          method=self.model_params.inference_method,
                          dense_output=True,
                          args=(self._x,
                                self._Q,
                                self._dim_x,
                                self.model_params))
        P = P_sol.sol(self._t + self._dt)
        x_sol = solve_ivp(CD_Bell_Bloom_EKF.dx_dt,
                          [self._t, self._t + self._dt],
                          self._x,
                          method=self.model_params.inference_method,
                          dense_output=True,
                          args=(self._dim_x, self.model_params))
        x = x_sol.sol(self._t + self._dt)
        self._x = x
        self._P = np.reshape(P, (self._dim_x, self._dim_x))
        self._t += self._dt

    def update(self, z):
        self._z = z
        self._y = self._z - self._measurement_strength*np.dot(self._H, self._x)  # innovation
        PHT = np.dot(self._P, self._H.T)
        S = np.dot(self._H, PHT) + self._R_delta
        S_INV = np.linalg.inv(S)
        # K = PH'inv(S)
        self._K = np.dot(PHT, S_INV)
        # x = x + Ky
        self._x = self._x + np.dot(self._K, self._y)
        I_KH = np.identity(self._dim_x) - np.dot(self._K, self._H)
        self._P = np.dot(np.dot(I_KH, self._P), I_KH.T) + np.dot(np.dot(self._K, self._R_delta), self._K.T)

    def predict_update(self, z, calculate_ss=False, Phi_Q_method=False):
        """ In continuous-discrete filter the equations for x and P in prediction step are solved numerically
        and then the appropriate correction is applied."""
        self.predict(Phi_Q_method=Phi_Q_method)
        self.update(z)
        if calculate_ss:
            self.steady_state()

    def steady_state(self, Phi_Q_method=False):
        if Phi_Q_method:
            self.steady_cov = solve_discrete_are(a=self._Phi_delta.T, b=self._H.T, q=self._Q_delta, r=self._R)
        else:
            self.steady_cov = solve_continuous_are(a=np.transpose(self._F),
                                                   b=np.transpose(self._H),
                                                   q=self._Q,
                                                   r=self._R*self._dt)
        return self.steady_cov
