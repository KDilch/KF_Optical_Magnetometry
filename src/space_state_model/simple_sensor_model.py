# -*- coding: utf-8 -*-
import logging
from scipy.integrate import solve_ivp
import numpy as np

from space_state_model.model import Model


class Simple_CC_Sensor_Model(Model):
    def __init__(self,
                 t,
                 simulation_params,
                 logger=None
                 ):
        self._logger = logger or logging.getLogger(__name__)
        Model.__init__(self, t, simulation_params, logger=logger)
        self._H = simulation_params.measurement.H
        self._dim_x = len(self._x)

    def step(self, method="default"):
        self._t += self._dt
        self._logger.debug('Performing a step for time %r' % str(self._t))
        if  (method == 'naive'):
            # this is just Euler-Mayurama method without any mid-steps
            dx = np.array([-(1/self._params.T2) * self._x[0] * self._dt + self._x[1] * self._x[2] * self._dt,
                           -(1/self._params.T2) * self._x[1] * self._dt - self._x[0] * self._x[2] * self._dt,
                           0.0])
            self._x += dx + self.get_intrinsic_noise()
        elif method == "ito_Euler_Mayurama":
            for i in range(10):
                dt_EM = self._dt/10
                dx = np.array([-(1/self._params.T2) * self._x[0] * dt_EM + self._x[1] * self._x[2] * dt_EM,
                               -(1/self._params.T2) * self._x[1] * dt_EM - self._x[0] * self._x[2] * dt_EM,
                               0.0])
                self._intrinsic_noise = self.get_intrinsic_noise(dt=dt_EM)
                self._x += dx + self._intrinsic_noise
        elif method == "ito_Runge_Kutta":
            for i in range(10):
                dt = self._dt/10
                a = np.array([-(1/self._params.T2) * self._x[0] + self._x[1] * self._x[2],
                               -(1/self._params.T2) * self._x[1] - self._x[0] * self._x[2],
                               0.0])
                y = self._x
                self._intrinsic_noise = self.get_intrinsic_noise(dt=dt) #bdW
                x = y + self._intrinsic_noise + a*dt + 0 #since there is no drift
                self._x = x
        else:
            x = solve_ivp(Simple_CC_Sensor_Model.dx_dt,
                          [self._t, self._t + self._dt],
                          self._x,
                       method=method,
                          dense_output=True,
                       args=(self._params.T2,
                             self._dt,
                             self.get_intrinsic_noise()))
            self._x = x.sol(self._t+self._dt)
        self.read_sensor()
        return self._x, self._z

    def read_sensor(self, noise=None):
        self._z = self.hx() * self._dt + self.get_measurement_noise()
        return

    @staticmethod
    def dx_dt(t, x, T2, dt, intrinsic_noise):
        y = np.array([- (1/T2) * x[0] + x[1] * x[2],
                          - (1/T2) * x[1] - x[0] * x[2],
                          0.0])
        y += intrinsic_noise/dt
        return y

    @staticmethod
    def G(x, t):
        return np.identity(3)

    def hx(self):
        return self._params.measurement.measurement_strength * self._H.dot(self._x)

    def get_intrinsic_noise(self, dt=None):
        self._dW = np.random.randn(self._dim_x)
        if not dt:
            return np.array([np.sqrt(self._params.dt * self._params.noise.Q_jx),
                             np.sqrt(self._params.dt * self._params.noise.Q_jy),
                             np.sqrt(self._params.dt * self._params.noise.Q_freq)]) * self._dW
        else:
            return np.array([np.sqrt(dt * self._params.noise.Q_jx),
                             np.sqrt(dt * self._params.noise.Q_jy),
                             np.sqrt(dt * self._params.noise.Q_freq)]) * self._dW

    def get_measurement_noise(self):
        return np.array([np.sqrt(self._params.dt * self._params.measurement.noise.R) * np.random.randn()])
