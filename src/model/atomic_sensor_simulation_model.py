# -*- coding: utf-8 -*-
from model.dynamical_model import Model


class Atomic_Sensor_Simulation_Model(Model):
    def __init__(self,
                 x,
                 dw,
                 dt,
                 t,
                 logger=None):
        Model.__init__(self, x, dw, dt, t, logger=logger)
