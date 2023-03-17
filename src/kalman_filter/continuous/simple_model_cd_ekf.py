import numpy as np
from scipy.integrate import solve_ivp, simps, odeint, quad
from scipy.linalg import solve_discrete_are, expm

from kalman_filter.continuous.ekf import EKF


class CD_EKF(EKF):
    def __init__(self, model_params):
        EKF.__init__(self, model_params)
        self.model_params = model_params
        self._R_delta = self._R/self._dt
        self._Phi_delta = None
        self._Q_delta = None

    @staticmethod
    def F(x, t, model_params):
        # Jacobian
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
    def dx_dt(t, x, dim_x, model_params):
        return CD_EKF.fx(x, t, model_params)

    @staticmethod
    def dP_dt(t, P, x, Q, dim_x, model_params):
        return np.reshape(np.dot(CD_EKF.F(x, t, model_params),
                                 np.reshape(P, (dim_x, dim_x))) + np.dot(np.reshape(P, (dim_x, dim_x)),
                                                                         np.transpose(CD_EKF.F(x,
                                                                                               t,
                                                                                               model_params))) + Q,
                          dim_x ** 2)

    def predict(self, Phi_Q_method=False):
        if Phi_Q_method:
            self.compute_Phi_delta__Q_delta_odeint(t_0=self._t)
            self._x = np.dot(self._Phi_delta, self._x)
            self._P = np.dot(np.dot(self._Phi_delta, self._P), self._Phi_delta.T) + self._Q_delta
            self._t += self._dt
        else:
            P_sol = solve_ivp(CD_EKF.dP_dt,
                              [self._t, self._t + self._dt],
                              np.reshape(self._P, self._dim_x ** 2),
                              method=self.model_params.inference_method,
                              dense_output=True,
                              args=(self._x,
                                    self._Q,
                                    self._dim_x,
                                    self.model_params))
            P = P_sol.sol(self._t + self._dt)
            x_sol = solve_ivp(CD_EKF.dx_dt,
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
        self._y = self._z - self._measurement_strength*np.dot(self._H, self._x) # innovation
        PHT = np.dot(self._P, self._H.T)
        # S = HPH' + R
        S = np.dot(self._H, PHT) + self._R_delta
        SI = np.linalg.inv(S)
        # K = PH'inv(S)
        self._K = np.dot(PHT, SI)
        # x = x + Ky
        self._x = self._x + np.dot(self._K, self._y)
        I_KH = np.identity(self._dim_x) - np.dot(self._K, self._H)
        self._P = np.dot(np.dot(I_KH, self._P), I_KH.T) + np.dot(np.dot(self._K, self._R_delta), self._K.T)

    def compute_Phi_delta__Q_delta_odeint(self, t_0, rule="simps"):
        Phi_0 = np.reshape(np.identity(self._dim_x), self._dim_x ** 2)  # initial Phi_delta is identity

        def dPhidt(t, Phi):
            return np.reshape(np.dot(self.F(self._x, t, self.model_params), np.reshape(Phi, (self._dim_x, self._dim_x))),
                              self._dim_x**2)

        Phi_sol = solve_ivp(dPhidt,
                            [t_0, t_0 + self._dt],
                            np.reshape(Phi_0, self._dim_x ** 2),
                            method=self.model_params.inference_method,
                            dense_output=True)
        if rule == "simps":
            t = np.arange(t_0, t_0+self._dt, self._dt/20)

            Phi_s_matrix_form = [np.reshape(Phi_sol.sol(i), (self._dim_x, self._dim_x)) for i in t]
            Phi_s_transpose_matrix_form = [np.transpose(a) for a in Phi_s_matrix_form]
            integrands = np.array(
                [np.dot(np.dot(a, self._Q), b) for a, b in zip(Phi_s_matrix_form, Phi_s_transpose_matrix_form)])
            integrand_split = list(map(list, zip(*integrands.reshape(*integrands.shape[:1], -1))))
            self._Q_delta = np.reshape(np.array([simps(i, t) for i in integrand_split]), (self._dim_x, self._dim_x))
        if rule == "quad":
            import threading
            from scipy import integrate

            def Phi_matrix(time, dim_x):
                return np.reshape(Phi_sol.sol(time), (dim_x, dim_x))
            def integrand(time):
                return np.dot(np.dot(Phi_matrix(time, 3), self._Q), np.transpose(Phi_matrix(time, 3)))

            res = np.zeros_like(self._Q)

            def f(i):
                for j in range(3):
                    integrand_ij= lambda k: integrand(k)[i, j]
                    integral = integrate.quad(integrand_ij, t_0, t_0+self._dt)
                    res[i, j] = integral[0]
            for i in range(3):
                threading.Thread(target=f(i)).start()

            self._Q_delta = res

        self._Phi_delta = np.reshape(Phi_sol.sol(t_0 + self._dt), (self._dim_x, self._dim_x))


    def predict_update(self, z, calculate_ss=False, Phi_Q_method=False):
        """ Idn continuous-discrete filter the equations for x and P in prediction step are solved numerically
        and then the appropriate correction is applied."""
        self.predict(Phi_Q_method=Phi_Q_method)
        self.update(z)
        if calculate_ss:
            self.steady_state()

    def steady_state(self):
        return solve_discrete_are(a=self._Phi_delta.T, b=self._H.T, q=self._Q_delta, r=self._R)
