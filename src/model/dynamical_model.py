# -*- coding: utf-8 -*-
from abc import ABC
import logging


class Model(ABC):
    """An abstract class representing any dynamical model (can be stochastic)."""
    def __init__(self,
                 x,
                 dw,
                 dt,
                 t,
                 logger=None):
        """
        :param x:       state vector
        :param dw:      noise increment term
        :param dt:      time step
        :param t:       time
        :param logger:  logger obj
        """

        self._logger = logger or logging.getLogger(__name__)
        self._x = x
        self._mean_x = x
        self._dt = dt
        self._dw = dw
        self._t = t

    @property
    def x(self):
        return self._x

    @property
    def mean_x(self):
        return self._mean_x

    @property
    def dw(self):
        return self._dw

    @property
    def t(self):
        return self._t

    def step(self):
        return NotImplementedError
