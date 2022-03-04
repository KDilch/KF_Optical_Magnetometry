# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import multivariate_normal
import logging

from noise.noise import Noise


class GaussianWhiteNoise(Noise):
    """
     A class representing Wiener process: dW = sqrt(t)*N(0, Q).
     """

    def __init__(self, mean, cov, dt, logger=None):
        """
        :param cov:
        :param dt:
        :param mean:
        :param logger: instance of logging.Logger or None (if None a new instance of this class will be created)
        """
        self.__logger = logger if logger else logging.getLogger(__name__)
        self.__value = None
        self.__cov = cov
        self.__mean = mean
        self.__dt = dt

    @property
    def value(self):
        return self.__value

    def step(self):
        self.__value = np.sqrt(self.__dt)*multivariate_normal.rvs(mean=self.__mean, cov=self.__cov)
        return self.__value

    def generate(self, num_steps):
        self.__logger.info('Generating Gaussian White Noise for %r steps' % str(num_steps))
        results = np.empty(num_steps)
        for x in range(num_steps):
            self.step()
            results[x] = self.__value
        times = np.arange(0, num_steps * self.__dt, self.__dt)
        return times, results
