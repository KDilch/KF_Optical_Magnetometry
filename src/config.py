from types import SimpleNamespace
import numpy as np
config = SimpleNamespace()

config.simulation = {
    't_max':  100.,
    'dt': 0.001,
    'dim_measurement': 1,
    'decoherence_x': 0.0,
    'decoherence_y': 0.0,
    'frequency_decay_rate': 0.0,  # frequency can behave according to OU process
    'x_0': np.array([0., 5., 8.]),  # initial state vector [Jx, Jy, omega]
    't_0': 0,
    'noise': {'Q_jx': 0.01,
              'Q_jy': 0.01,
              'Q_freq': 0.0},
    'measurement': {'measurement_strength': 1.,
                    'H': np.array([[0., 1., 0.]]),
                    'noise': {'R': 0.001,
                              'mean': 0.0}
                    }
}

config.filter_ekf = {
    'dt': 0.001,
    'decoherence_x': 0.0,
    'decoherence_y': 0.0,
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
