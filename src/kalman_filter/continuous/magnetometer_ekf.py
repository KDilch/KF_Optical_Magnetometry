import numpy as np

from kalman_filter.continuous.ekf import EKF


class MagnetometerEKF(EKF):
    def __init__(self, model_params):
        EKF.__init__(self, model_params=model_params)
        self._F = np.array([[-model_params.spin_corr_const, self._x[2], self._x[1]],
                            [-self._x[2], -model_params.spin_corr_const, -self._x[0]],
                            [0.0, 0.0, 0.0]])
