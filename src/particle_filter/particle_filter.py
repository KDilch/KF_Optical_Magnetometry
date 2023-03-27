import numpy as np
import scipy


class ParticleFilter(object):

    def __init__(self, model_params, num_particles=1000):
        self._num_particles = num_particles

        # distribute particles randomly with uniform weight
        self._weights = np.empty(num_particles)
        self._weights.fill(1. / num_particles)

        self._x = model_params.x_0
        self._t = model_params.t_0
        self._dt = model_params.dt
        self._model_params = model_params
        self._measurement_strength = model_params.measurement.measurement_strength
        self._dim_x = len(self._x)
        self._dim_z = model_params.measurement.dim_z
        self._F = np.eye(self._dim_x)  # linearized space_state_model
        self._H = model_params.measurement.H
        self._R = model_params.measurement.R
        self._Q = model_params.noise.Q
        self._P = model_params.P0
        self._y = np.zeros((self._dim_z, 1))  # residual

        self._particles = np.empty((self._num_particles, self._dim_x))
        self.create_gaussian_particles()

    def create_gaussian_particles(self):
        for i in range(self._dim_x):
            self._particles[:, i] = self._x[i] + (np.random.randn(self._num_particles) * self._P[i][i])
        return self._particles

    @staticmethod
    def fx(x_0, model_params):
        dx_dt = np.zeros(3)
        dx_dt[0] = - (1 / model_params.T2) * x_0[0] + x_0[1] * x_0[2]
        dx_dt[1] = - (1 / model_params.T2) * x_0[1] - x_0[0] * x_0[2]
        dx_dt[2] = 0
        return dx_dt

    def predict(self):
        """ move according to the model"""
        for i in range(self._num_particles):
            self._particles[i, :] += self.fx(self._x, model_params=self._model_params) * self._dt

    def update(self, z):
        """Update according to the measurement outcome. Landmarks arepossible outcomes."""
        for i in range(self._dim_x):
            distance = np.linalg.norm(self._particles[:, i], axis=0)
            self._weights *= scipy.stats.norm(distance, self._R).pdf(z[i])

        self._weights += 1.e-300  # avoid round-off to zero
        self._weights /= sum(self._weights)  # normalize

    def estimate(self):
        """returns mean and variance of the weighted particles"""
        pos = self._particles[:, 0:2]
        self._x = np.average(pos, weights=self._weights, axis=0)
        self._P = np.average((pos - self._x) ** 2, weights=self._weights, axis=0)
        return self._x, self._P