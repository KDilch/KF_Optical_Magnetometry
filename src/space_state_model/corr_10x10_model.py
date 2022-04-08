# -*- coding: utf-8 -*-
import logging
from scipy.integrate import odeint
import sdeint
import numpy as np
from functools import partial

from space_state_model.model import Model


class Corr_10x10_CC_Sensor_Model(Model):
    def __init__(self,
                 t,
                 simulation_params,
                 logger=None
                 ):
        self._logger = logger or logging.getLogger(__name__)
        Model.__init__(self, t, simulation_params, logger=logger)
        self._H = simulation_params.H
        self._dim_x = len(self._x)

    def step(self, method="default"):
        self._t += self._dt
        self._logger.debug('Performing a step for time %r' % str(self._t))
        noise = np.random.randn()
        tspan = np.linspace(self._t, self._t+self._dt, 20)
        dx_dt_bound = partial(self.dx_dt, params=self._params)
        get_B_bound = partial(self.get_B, params=self._params)
        x = sdeint.itoint(dx_dt_bound, get_B_bound, self._x, tspan)[-1, :]
        self._x = x
        self.read_sensor(noise)
        return self._x, self._z

    def read_sensor(self, noise=None):
        if noise is None:
            raise ValueError('In correlated version noise should not be None')
        else:
            self._z = self.hx() * self._dt + self.get_measurement_noise(noise)
        return

    @staticmethod
    def dx_dt(x, t, params):
        dx_dt = np.zeros(10)
        dx_dt[0] = -0.5 * (params.measurement_strength + params.coll_decoherence) * x[0] - x[1] * x[9]
        dx_dt[1] = - 0.5 * params.coll_decoherence * x[1] + x[0] * x[9]
        dx_dt[2] = - 0.5 * params.measurement_strength * x[2]
        dx_dt[3] = - 2 * x[9] * x[6] - (params.measurement_strength + params.coll_decoherence) * (x[3] - x[4]) + (params.measurement_strength + params.coll_decoherence) * x[1] ** 2 - 4 * params.eta * params.measurement_strength * x[6] ** 2
        dx_dt[4] = + 2 * x[9] * x[6] + params.coll_decoherence * (x[3] - x[4]) + params.coll_decoherence * x[0] ** 2 - 4 * params.eta * params.measurement_strength * x[4] ** 2
        dx_dt[5] = + params.measurement_strength * (x[0] ** 2 + x[3] - x[5] - 4 * params.eta * x[7] ** 2)
        dx_dt[6] = + x[9] * (x[3] - x[4]) - params.coll_decoherence * x[0] * x[1] - (2 * params.coll_decoherence + params.measurement_strength / 2) * x[6] - 4 * params.eta * params.measurement_strength * x[4] * x[6]
        dx_dt[7] = - (1 / 2) * (params.coll_decoherence + params.measurement_strength * (1 + 8 * params.eta * x[4])) * x[7] + x[8] * x[9]
        dx_dt[8] = - (1 / 2) * params.coll_decoherence * x[8] - params.measurement_strength * (x[0] * x[1] + 4 * params.eta * x[6] * x[7] + 2 * x[8]) - x[7] * x[9]
        dx_dt[9] = 0
        return dx_dt

    def hx(self):
        return 2 * np.sqrt(self._params.eta*self._params.measurement_strength) * self._H.dot(self._x)

    @staticmethod
    def get_B(x, t, params):
        dW = np.zeros(10)
        dW[0] = 2 * np.sqrt(params.eta * params.measurement_strength) * x[7]
        dW[1] = 2 * np.sqrt(params.eta * params.measurement_strength) * x[5]
        dW[2] = 2 * np.sqrt(params.eta * params.measurement_strength) * x[8]
        dW[3] = params.num_atoms / 4 * np.sin(x[9] * t) * (np.cos(x[9] * t)) ** 2
        dW[4] = -params.num_atoms / 4 * np.sin(x[9] * t) * (np.cos(x[9] * t)) ** 2
        dW[5] = 0
        dW[6] = -params.num_atoms / 16 * (np.cos(x[9] * t) + np.cos(3 * x[9] * t))
        dW[7] = 0
        dW[8] = 0
        dW[9] = 0
        return dW.reshape((10, 1))

    def get_measurement_noise(self, noise):
        return np.array([np.sqrt(self._params.dt * self._params.eta) * noise])
