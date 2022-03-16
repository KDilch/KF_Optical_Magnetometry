# -*- coding: utf-8 -*-
import logging
import numpy as np

from dynamics.dynamical_model import Model
from noise.gaussian_white_noise import GaussianWhiteNoise


class Hajimolahoseini_Model(Model):
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
        self._mean_x = [self._params.A*(np.cos(self._params.omega*self._t)*np.cos(self._params.omega*self._dt)-np.sin(self._params.omega*self._t)*np.sin(self._params.omega*self._dt)),
                        self._params.A*(np.cos(self._params.omega*self._t)*np.sin(self._params.omega*self._dt)+np.sin(self._params.omega*self._t)*np.sin(self._params.omega*self._dt)),
                        self._params.omega]
        # self._mean_x[0] = self._params.A*np.cos(self._params.omega*self._t)
        # self._mean_x[1] = self._params.A*np.sin(self._params.omega*self._t)
        # self._mean_x[2] = self._params.omega
        self._x = self._mean_x + self._noise.step()
        return self._x
