from types import SimpleNamespace
import numpy as np
config = SimpleNamespace()

config.simulation = {
    't_max': 6.,
    'dt': 0.0001,
    'dim_measurement': 1,
    'spin_corr_const': 0.0,
    'frequency_decay_rate': 0.0,  # frequency can behave according to OU process
    'x_0': np.array([0., 5., 8.]),  # initial state vector [Jy, Jz, omega]
    't_0': 0,
    'noise': {'Q': np.array([[0.01, 0., 0.],
                             [0., 0.01, 0.],
                             [0., 0., 0.0]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': 1.,
                    'H': np.array([[0., 1., 0.]]),
                    'noise': {'R': np.array([0.001]),
                              'mean': np.array([0.0])}
                    }
}


config.filter_ekf = {
    'dt': 0.0001,
    'spin_corr_const': 0.0,
    'frequency_decay_rate': 0.0,
    'x_0': np.array([0.0, 5.0, 8.1]),
    't_0': 0.,
    'P0': np.array([[0.0, 0., 0.],
                    [0., 0.0, 0.],
                    [0., 0., 0.01]]),
    'noise': {'Q': np.array([[0.01, 0., 0.],
                             [0., 0.01, 0.],
                             [0., 0., 0.0]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': 1.,
                    'H': np.array([[0., 1., 0.]]),
                    'dim_z': 1,
                    'R': np.array([[0.001]])}
}
