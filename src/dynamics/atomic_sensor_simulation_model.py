# -*- coding: utf-8 -*-
import logging

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
        self._mean_x[0] += -self._params.spin_corr_const*self._x[0]*self._dt+self._x[1]*self._x[2]*self._dt
        self._mean_x[1] += -self._params.spin_corr_const*self._x[1]*self._dt-self._x[0]*self._x[2]*self._dt
        self._x = self._mean_x + self._noise.step()
        return self._x
