import numpy as np
import scipy


class MLE_omega(object):
    def __init__(self, N, T2, sigma_sq, t_0, delta_t):
        self.N = N
        self.T2 = T2
        self.sigma_sq = sigma_sq
        self._MLE_sum1 = lambda omega: 0  # sum 2 is a function of signal
        self._MLE_sum2 = lambda omega: 0  # sum 2 doesn't depend on signal
        self.MLE = None
        self._t = t_0
        self._t_0 = t_0
        self._delta_t = delta_t

    def compute_sum_2(self):
        time_arr = np.arange(self._t_0, self._t, self._delta_t)
        return lambda omega: np.sum([self.N ** 2 * np.exp(-(2 * t / self.T2)) * np.sin(omega * t) ** 2 / (8 * self.sigma_sq) for t in time_arr])

    def compute_sum_1(self, signal):
        time_arr = np.arange(self._t_0, self._t, self._delta_t)
        return lambda omega: np.sum([self.N * signal[index] * np.exp(-time_arr[index] / self.T2) * np.sin(omega * time_arr[index]) for index, val in enumerate(time_arr)])

    def approximate_sum_2(self):
        return lambda omega: omega**2/(4*(1/self.T2)**3+4*(1/self.T2)*omega**2)

    def find_MLE(self, signal):
        self._MLE_sum1 = self.compute_sum_1(signal)
        self._MLE_sum2 = self.approximate_sum_2()
        self.MLE = scipy.optimize.minimize_scalar(lambda omega: -(self._MLE_sum1(omega) - self._MLE_sum2(omega)),
                                                  bounds=[0.95, 1.02], method='bounded').x
        self._t += self._delta_t
        return self.MLE
