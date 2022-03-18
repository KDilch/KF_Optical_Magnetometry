# -*- coding: utf-8 -*-
from abc import ABC
import logging
from copy import deepcopy


class Model(ABC):
    """An abstract class representing any dynamical dynamics (can be stochastic)."""
    def __init__(self,
                 t,
                 model_params=None,
                 noise=None,
                 logger=None):
        """
        :param t:  time
        :param model_params: obj
        :param noise:  obj
        :param logger: logger obj
        """

        self._t = t
        self._logger = logger or logging.getLogger(__name__)
        self._x = deepcopy(model_params.x_0)
        self._mean_x = deepcopy(model_params.x_0)
        self._dt = model_params.dt
        self._noise = noise
        self._params = model_params

    @property
    def x(self):
        return self._x

    @property
    def mean_x(self):
        return self._mean_x

    @property
    def t(self):
        return self._t

    def step(self):
        return NotImplementedError
