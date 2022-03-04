from types import SimpleNamespace
import numpy as np
config = SimpleNamespace()

config.simulation = {
    'spin_corr_const': 0.33,
    't_max': 1.,
    'dt': 0.005,
    'x_0': np.array([2., 2., 6.]),  # initial state vector [Jy, Jz, omega]
    'noise': {'Q': np.array([[0.1, 0., 0., 0.],
                             [0., 0.1, 0., 0.],
                             [0., 0., 0.1, 0.],
                             [0., 0., 0., 0.1]]),
              'mean': np.array([0.0, 0.0, 0.0])}
}

config.filter = {
    'dt': 0.005,
    'x_0': None,
    'P0': None,
    'noise': {'Q': np.array([[0.1, 0., 0., 0.],
                             [0., 0.1, 0., 0.],
                             [0., 0., 0.1, 0.],
                             [0., 0., 0., 0.1]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': 100.,
                    'H': np.array([[0., 1., 0.]]),
                    'R': np.array([[0.1]])}
}