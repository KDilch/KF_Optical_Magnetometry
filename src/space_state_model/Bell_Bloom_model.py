# -*- coding: utf-8 -*-
import logging
import sdeint
import numpy as np

from space_state_model.model import Model


class Bell_Bloom_Magnetometer_Model(Model):
    def __init__(self, t, simulation_params, logger=None):
        self._logger = logger or logging.getLogger(__name__)
        Model.__init__(self, t, simulation_params, logger=logger)
        self._H = simulation_params.measurement.H
        self._F_max = simulation_params.num_atoms/2
        self.omega_pump = simulation_params.omega_pumping
        self.period_pump = (2 * np.pi / self.omega_pump)
        self.half_pump_duration = self.period_pump*0.1/2
        self.n = 0
        self._dim_x = len(self._x)

    def step(self, method="default", num_steps=20):
        self._t += self._dt
        t_span = np.linspace(self._t, self._t + self._dt, num_steps)
        dW = np.array([self.get_intrinsic_noise(self._dt) for _ in t_span[1:]])
        if method == "ito_Euler_Mayurama":
            # dummy approach to verify the stochastic solvers
            # self._x += self.f_x_t(self._x, self.t) + self.get_intrinsic_noise(self._dt)
            x = sdeint.itoEuler(f=self.f_x_t,
                                G=self.G,
                                y0=self._x,
                                tspan=t_span,
                                dW=dW)
            self._x = x[-1]
        else:
            x = sdeint.itoSRI2(f=self.f_x_t,
                                G=self.G,
                                y0=self._x,
                                tspan=t_span,
                                dW=dW)
            self._x = x[-1]

        self.read_sensor()
        return self._x, self._z

    def __pump_rate(self, t):
        if np.abs(np.cos(self.omega_pump * t) - 1) < np.cos(self.omega_pump * 0.9 * self.period_pump):
            return self._params.pump_amplitude
        return 0
        # return 1.
        # if (t > (self.n*self.period_pump - self.half_pump_duration)) and (
        #         t < (self.n*self.period_pump + self.half_pump_duration)):
        #     return 1.
        # elif (t > ((self.n+1)*self.period_pump - self.half_pump_duration)) and (
        #         t < ((self.n+1)*self.period_pump + self.half_pump_duration)):
        #     self.n += 1
        #     return 1.
        # return 0.

    def read_sensor(self):
        self._z = self.hx() * self._dt + self.get_measurement_noise()
        return

    def f_x_t(self, x, t):
        return np.array([- (1 / self._params.T2) * x[0] + x[1] * x[2] - self.__pump_rate(t)*x[0],
                         - (1 / self._params.T2) * x[1] - x[0] * x[2] + self.__pump_rate(t)*(self._F_max-x[1]),
                         0.0])

    def G(self, x, t):
        return np.identity(len(x))

    def hx(self):
        return self._params.measurement.measurement_strength * self._H.dot(self._x)

    def get_intrinsic_noise(self, dt=None):
        """Generates dW Wiener increment of the intrinsic noise."""
        dt_noise = dt if dt is None else self._dt
        return np.array([np.sqrt(dt_noise * self._params.noise.Q_jx),
                        np.sqrt(dt_noise * self._params.noise.Q_jy),
                        np.sqrt(dt_noise * self._params.noise.Q_freq)]) * np.random.randn(self._dim_x)

    def get_measurement_noise(self):
        """Generates dW Wiener increment of the measurement noise."""
        return np.array([np.sqrt(self._params.dt * self._params.measurement.noise.R) * np.random.randn()])
