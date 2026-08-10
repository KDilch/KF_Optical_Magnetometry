import numpy as np

from kalman_filter.cd_ekf import CD_EKF


class CD_EKF_simple_magnetometer(CD_EKF):
    def __init__(self, model_params):
        CD_EKF.__init__(self, model_params)

    @staticmethod
    def F(x, t, model_params):
        # Jacobian
        return np.array([[-1/model_params.T2, x[2], x[1]],
                         [-x[2], -1/model_params.T2, -x[0]],
                         [0.0, 0.0, 0.0]])

    @staticmethod
    def fx(x_0, t, model_params):
        x = np.zeros(3)
        x[0] += - (1/model_params.T2) * x_0[0] + x_0[1] * x_0[2]
        x[1] += - (1/model_params.T2) * x_0[1] - x_0[0] * x_0[2]
        x[2] += 0
        return x
