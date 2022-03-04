# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod  # abstract class


class Noise(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError
