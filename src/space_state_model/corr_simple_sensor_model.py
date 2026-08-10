# -*- coding: utf-8 -*-
import logging
from scipy.integrate import odeint
import sdeint
from functools import partial
import numpy as np

from space_state_model.model import Model


class Simple_CC_Correlated_Sensor_Model(Model):
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
        noise = np.random.randn()
        if method == 'sdeint':
            tspan = np.linspace(self._t, self._t + self._dt, 20)
            fx_bound = partial(self.fx, params=self._params)
            B_bound = partial(self.B, params=self._params)

        if method == 'odeint':
            x = odeint(Simple_CC_Correlated_Sensor_Model.dx_dt, self._x,
                       np.linspace(self._t, self._t + self._dt, 20),
                       args=(self._params,))[-1, :]
            self._x = x + self.get_intrinsic_noise(noise)

        if (method == 'default') or (method == 'naive'):
            dx = np.array([- (1/self._params.T2) * self._x[0] * self._dt + self._x[1] * self._x[2] * self._dt,
                           -(1/self._params.T2) * self._x[1] * self._dt - self._x[0] * self._x[2] * self._dt,
                           0.0])
            self._x += dx + self.get_intrinsic_noise(noise)

        self.read_cont_sensor(noise)
        return self._x, self._z

    def read_cont_sensor(self, noise=None):
        if noise is None:
            raise ValueError('In correlated version noise should not be None')
        else:
            self._z = self.hx() * self._dt + self.get_measurement_noise(noise)
        return

    def read_sensor(self):
        noise = np.random.randn()
        self.read_cont_sensor(noise)

    @staticmethod
    def fx(x, t, params):
        dx_dt = np.array([- 1/(params.T2)* x[0] + x[1] * x[2],
                          - 1/(params.T2) * x[1] - x[0] * x[2],
                          0.0])
        return dx_dt

    @staticmethod
    def dx_dt(x, t, params):
        dx_dt = Simple_CC_Correlated_Sensor_Model.fx(x, t, params)
        return dx_dt

    def hx(self):
        return self._params.measurement.measurement_strength * self._H.dot(self._x)

    def get_intrinsic_noise(self, noise):
        return np.array([np.sqrt(self._params.dt * self._params.noise.Q_m),
                         np.sqrt(self._params.dt * self._params.noise.Q_m),
                         np.sqrt(self._params.dt * self._params.noise.Q_freq)]) * noise

    def get_measurement_noise(self, noise):
        return np.array([np.sqrt(self._params.dt * self._params.noise.Q_m) * noise])
