# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import logging
from copy import deepcopy


class Model(ABC):
    """An abstract class representing any space_state_model."""
    def __init__(self,
                 t,
                 model_params=None,
                 logger=None):
        """
        :param t:  time
        :param model_params: obj
        :param logger: logger obj
        """
        self._t = t
        self._logger = logger or logging.getLogger(__name__)
        self._x = deepcopy(model_params.x_0)
        self._z = None
        self._dt = model_params.dt
        self._params = model_params

    @property
    def x(self):
        """State vector"""
        return self._x

    @property
    def z(self):
        """Measurement outcome"""
        return self._z

    @property
    def t(self):
        return self._t

    @abstractmethod
    def step(self, method="default"):
        """
        :param method: string representing method of the simulation
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def read_sensor(self, noise=None):
        raise NotImplementedError

    @abstractmethod
    def hx(self):
        raise NotImplementedError
