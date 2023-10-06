# -*- coding: utf-8 -*-
import logging
import sdeint
import numpy as np

from space_state_model.model import Model


class Simple_Sensor_Model(Model):
    def __init__(self, t, simulation_params, discrete_measurement=False, logger=None):
        self._logger = logger or logging.getLogger(__name__)
        Model.__init__(self, t, simulation_params, logger=logger)
        self._discrete_measurement = discrete_measurement
        self._H = simulation_params.measurement.H  # specifies the measurement model
        self._dim_x = len(self._x)

    def step(self, method="default", num_steps=20):
        self._t += self._dt
        t_span = np.linspace(self._t, self._t + self._dt, num_steps)
        dW = np.array([self.get_intrinsic_noise(self._dt) for _ in t_span[1:]])
        if method == "ito_Euler_Mayurama":
            x = sdeint.itoEuler(f=self.f_x_t,
                                G=self.G,
                                y0=self._x,
                                tspan=t_span,
                                dW=dW)
            self._x = x[-1]
        else:
            x = sdeint.itoSRI2(f=self.f_x_t,
                                G=self.G,
                                y0=self._x,
                                tspan=t_span,
                                dW=dW)
            self._x = x[-1]

        self.read_sensor()
        return self._x, self._z

    def read_sensor(self):
        self._z = self.hx() * self._dt + self.get_measurement_noise()
        return

    def f_x_t(self, x, t):
        return np.array([- (1 / self._params.T2) * x[0] + x[1] * x[2],
                         - (1 / self._params.T2) * x[1] - x[0] * x[2],
                         0.0])
    def G(self, x, t):
        return np.identity(len(x))

    def hx(self):
        return self._params.measurement.measurement_strength * self._H.dot(self._x)

    def get_intrinsic_noise(self, dt=None):
        """Generates dW Wiener increment of the intrinsic noise."""
        dt_noise = dt if dt is None else self._dt
        return np.array([np.sqrt(dt_noise * self._params.noise.Q_jx),
                        np.sqrt(dt_noise * self._params.noise.Q_jy),
                        np.sqrt(dt_noise * self._params.noise.Q_freq)]) * np.random.randn(self._dim_x)

    def get_measurement_noise(self):
        """Generates dW Wiener increment of the measurement noise."""
        if self._discrete_measurement:
            return np.array([np.sqrt(self._params.measurement.noise.R_delta) * np.random.randn()])
        else:
            return np.array([np.sqrt(self._params.dt * self._params.measurement.noise.R) * np.random.randn()])
