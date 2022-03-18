# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod  # abstract class
import logging


class MeasurementModel(ABC):
    def __init__(self, simulation_params, noise, logger=None):
        self._logger = logger or logging.getLogger(__name__)
        self._params = simulation_params
        self._noise = noise

    @abstractmethod
    def read_sensor(self, state):
        raise NotImplementedError
