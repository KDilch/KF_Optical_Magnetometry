# -*- coding: utf-8 -*-
import logging

from measurement.model import MeasurementModel
from noise.gaussian_white_noise import GaussianWhiteNoise


class AtomicSensorMeasurementModel(MeasurementModel):
    def __init__(self,
                 simulation_params,
                 logger=None
                 ):
        self._logger = logger or logging.getLogger(__name__)
        noise = GaussianWhiteNoise(mean=simulation_params.measurement.noise.mean,
                                   cov=[simulation_params.measurement.noise.R],
                                   dt=simulation_params.dt)
        MeasurementModel.__init__(self, simulation_params, noise, logger=logger)

        self._H = self._params.measurement.H
        self._dt = simulation_params.dt
        self._measurement_strength = self._params.measurement.measurement_strength
        self.__dz = simulation_params.x_0[1]
        self.__dz_no_noise = simulation_params.x_0[1]

    def read_sensor(self, state):
        self.__dz_no_noise = self._measurement_strength * self._H.dot(state) * self._dt
        self.__dz = self.__dz_no_noise + self._noise.step()
        return self.__dz

    @property
    def z(self):
        """Returns the value measured in the previous step"""
        return self.__dz

    @property
    def z_no_noise(self):
        """Returns the value measured previously without the measurement noise (but with the intrinsic noise included"""
        return self.__dz_no_noise
