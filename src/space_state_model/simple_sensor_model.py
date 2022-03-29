# -*- coding: utf-8 -*-
import logging
from scipy.integrate import odeint
import numpy as np

from space_state_model.model import Model


class Simple_CC_Sensor_Model(Model):
    def __init__(self,
                 t,
                 simulation_params,
                 logger=None
                 ):
        self._logger = logger or logging.getLogger(__name__)
        Model.__init__(self, t, simulation_params, logger=logger)
        self._H = simulation_params.measurement.H
        self._dim_x = len(self._x)

    def step(self, method="default"):
        self._t += self._dt
        self._logger.debug('Performing a step for time %r' % str(self._t))
        if method == 'runge-kutta':
            x = odeint(Simple_CC_Sensor_Model.dx_dt, self._x,
                       np.linspace(self._t, self._t + self._dt, 20),
                       args=(self._params.decoherence_x,
                             self._params.decoherence_y,
                             self.get_intrinsic_noise()))[-1, :]
            self._x = x + self.get_measurement_noise()

        if (method == 'default') or (method == 'naive'):
            dx = np.array([- self._params.decoherence_x * self._x[0] * self._dt + self._x[1] * self._x[2] * self._dt,
                           -self._params.decoherence_y * self._x[1] * self._dt - self._x[0] * self._x[2] * self._dt,
                           0.0])
            self._x += dx + self.get_intrinsic_noise()

        else:
            raise ValueError('Invalid method selected')
        self.read_sensor()
        return self._x, self._z

    def read_sensor(self):
        self._z = self.hx() * self._dt + self.get_measurement_noise()
        return

    @staticmethod
    def dx_dt(x, decoherence_x, decoherence_y, dt, measurement_noise):
        dx_dt = np.array([- decoherence_x * x[0] + x[1] * x[2],
                          - decoherence_y * x[1] - x[0] * x[2],
                          0.0])
        dx_dt += np.sqrt(dt)*measurement_noise
        return dx_dt

    def hx(self):
        return self._params.measurement.measurement_strength * self._H.dot(self._x)

    def get_intrinsic_noise(self):
        return np.array([np.sqrt(self._params.dt * self._params.noise.Q_jx),
                         np.sqrt(self._params.dt * self._params.noise.Q_jx),
                         np.sqrt(self._params.dt * self._params.noise.Q_freq)]) * np.random.randn(self._dim_x)

    def get_measurement_noise(self):
        return np.array([np.sqrt(self._params.dt * self._params.measurement.noise.R) * np.random.randn()])


