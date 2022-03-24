# -*- coding: utf-8 -*-
import logging
from scipy.integrate import odeint
import numpy as np

from dynamics.dynamical_model import Model
from noise.gaussian_white_noise import GaussianWhiteNoise


class Atomic_Sensor_Simulation_Model(Model):
    def __init__(self,
                 t,
                 simulation_params,
                 logger=None
                 ):
        self._logger = logger or logging.getLogger(__name__)
        noise = GaussianWhiteNoise(mean=simulation_params.noise.mean,
                                   cov=simulation_params.noise.Q,
                                   dt=simulation_params.dt)
        Model.__init__(self, t, simulation_params, noise, logger=logger)

    def step(self):
        self._t += self._dt
        self._logger.debug('Performing a step for time %r' % str(self._t))
        # x = odeint(Atomic_Sensor_Simulation_Model.dx_dt, self._x, np.linspace(self._t, self._t + self._dt, 10), args=(self,))[-1, :]
        # self._x = x + self._noise.step()
        # # NAIVELY add dx
        dx = np.zeros(self._x.size)
        dx[0] = - self._params.spin_corr_const * self._x[0] * self._dt + self._x[1] * self._x[2] * self._dt
        dx[1] = -self._params.spin_corr_const * self._x[1] * self._dt - self._x[0] * self._x[2] * self._dt
        dx[2] = 0
        self._x += dx + self._noise.step()
        return self._x

    @staticmethod
    def dx_dt(x, t, self):
        dx_dt = np.zeros(x.shape)
        dx_dt[0] = - self._params.spin_corr_const * x[0] + x[1] * x[2]
        dx_dt[1] = - self._params.spin_corr_const * x[1] - x[0] * x[2]
        dx_dt[2] = 0.0
        return dx_dt

