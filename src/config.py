from types import SimpleNamespace
import numpy as np
config = SimpleNamespace()

config.simulation = {
    'spin_corr_const': 1.,
    't_max': 1.5,
    'dt': 0.0001,
    'x_0': np.array([6., 6., 6.]),  # initial state vector [Jy, Jz, omega]
    'noise': {'Q': np.array([[0.001, 0., 0.],
                             [0., 0.001, 0.],
                             [0., 0., 0.001]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': 2.,
                    'H': np.array([[0., 1., 0.]]),
                    'noise': {'R': np.array([0.001]),
                              'mean': np.array([0.0])}
                    }
}

config.filter = {
    'dt': 0.0001,
    'spin_corr_const': 1.,
    'x_0': np.array([3., 3., 10.]),
    'P0': np.array([[0.5, 0., 0.],
                    [0., 0.5, 0.],
                    [0., 0., 5.]]),
    'noise': {'Q': np.array([[0.001, 0., 0.],
                             [0., 0.001, 0.],
                             [0., 0., 0.001]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': 2.,
                    'H': np.array([[0., 1., 0.]]),
                    'dim_z': 1,
                    'R': np.array([[0.001]])}
}