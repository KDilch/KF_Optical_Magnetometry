# -*- coding: utf-8 -*-
import logging
import numpy as np
import sdeint

from space_state_model.model import Model


class UnitlessMagnetometerModel(Model):
    """
    Object-oriented simulation class for the unitless magnetometer model.
    Inherits from space_state_model.model.Model.
    """
    def __init__(self, t, simulation_params, logger=None):
        self._logger = logger or logging.getLogger(__name__)
        Model.__init__(self, t, simulation_params, logger=logger)
        
        self._dim_x = len(self._x)
        self._T2 = simulation_params.T2
        self._w01 = simulation_params.w01
        self._dc = simulation_params.dc
        self._tau = simulation_params.tau
        self._sim_type = simulation_params.type
        self._sig_v = simulation_params.sig_v

    def step(self, method="default", num_steps=20):
        """
        Performs one integration step of self._dt.
        """
        t_span = np.linspace(self._t, self._t + self._dt, num_steps)
        dt_grid = self._dt / (num_steps - 1)
        
        # Generates Wiener increments scaled by the grid step size to maintain correct continuous-time noise strength
        dW = np.array([self.get_intrinsic_noise(dt_grid) for _ in t_span[1:]])
        
        # Apply deterministic resets for jump and sine types
        if self._sim_type == "jump":
            if self._t <= 2.29879 / 8:
                self._x[2] = 2 * np.pi * 9800 * self._T2
            elif (self._t >= 2.29879 / 8 and self._t <= 2.29879 / 4):
                self._x[2] = 2 * np.pi * 9100.4 * self._T2
            else:
                self._x[2] = 2 * np.pi * 10000.0 * self._T2
        elif self._sim_type == "sine":
            self._x[2] = 0.1 * self._w01 * np.sin(0.05 * self._w01 * self._t) + 2 * np.pi * 10800.0 * self._T2
            
        f_func = self.f_x_OU if self._sim_type == "OU" else self.f_x
        
        # Stochastic integration using itoSRI2
        x_integrated = sdeint.itoSRI2(f=f_func,
                                      G=self.G,
                                      y0=self._x,
                                      tspan=t_span,
                                      dW=dW)
                                      
        self._x = x_integrated[-1]
        self._t += self._dt
        
        self.read_sensor()
        return self._x, self._z

    def f_x(self, x, t):
        f = np.zeros(3)
        f[0] += -x[0] + x[2] * x[1]
        f[1] += -x[2] * x[0] - x[1]
        f[2] += 0
        return f

    def f_x_OU(self, x, t):
        f = np.zeros(3)
        f[0] += -x[0] + x[2] * x[1]
        f[1] += -x[2] * x[0] - x[1]
        f[2] += (self._w01 - x[2]) / self._tau
        return f

    def G(self, x, t):
        return np.diag([np.sqrt(2), np.sqrt(2), np.sqrt(self._dc)])

    def get_intrinsic_noise(self, dt):
        """Generates dW Wiener increment of the intrinsic noise."""
        return np.array([np.sqrt(dt),
                         np.sqrt(dt),
                         np.sqrt(dt)]) * np.random.randn(3)

    def read_sensor(self, with_noise=False):
        if with_noise:
            # Measurement: J_z + white noise
            self._z = self._x[1] + self._sig_v * np.random.randn()
        else:
            # Measurement: J_z (clean state)
            self._z = self._x[1]
        return self._z

    def hx(self):
        return self._x[1]
