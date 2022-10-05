import numpy as np
import scipy


class ParticleFilter(object):

    def __init__(self, model_params, mean_prior, cov_prior, dt, num_particles=1000):
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

        self._dz = np.array([model_params.x_0[1]] * self._dim_z)
        self._num_particles = num_particles
        self._dt = dt
        self._mean_prior = mean_prior
        self._dim_state_vec = len(mean_prior)
        self._cov_prior = cov_prior
        self._particles = np.empty((self._num_particles, self._dim_state_vec))

        self.create_gaussian_particles()

    def create_gaussian_particles(self):
        for i in range(self._dim_state_vec):
            self._particles[:, i] = self._mean_prior[i] + (np.random.randn(self._num_particles) * self._cov_prior[i][i])
        return self._particles

    @staticmethod
    def fx(x, model_params):
        raise NotImplementedError('Implement fx function.')

    def predict(self):
        """ move according to the model"""
        for i in range(self._num_particles):
            self._particles[i, :] += self.fx(self._x, model_params=self._model_params) * self._dt

    def update(self, particles, weights, z, R, landmarks):
        """Update according to the measurement outcome"""
        for i, landmark in enumerate(landmarks):
            distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
            weights *= scipy.stats.norm(distance, R).pdf(z[i])

        weights += 1.e-300  # avoid round-off to zero
        weights /= sum(weights)  # normalize

    def estimate(self, particles, weights):
        """returns mean and variance of the weighted particles"""

        pos = particles[:, 0:2]
        mean = np.average(pos, weights=weights, axis=0)
        var = np.average((pos - mean) ** 2, weights=weights, axis=0)
        return mean, var