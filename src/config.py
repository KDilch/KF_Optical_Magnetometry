from types import SimpleNamespace
import numpy as np
config = SimpleNamespace()

config.simulation = {
    't_max': 10.,
    'dt': 0.0001,
    'spin_corr_const': 0.0,
    'frequency_decay_rate': 0.0,  # frequency can behave according to OU process
    'x_0': np.array([0., 5., 8.]),  # initial state vector [Jy, Jz, omega]
    't_0': 0,
    'noise': {'Q': np.array([[0.01, 0., 0.],
                             [0., 0.01, 0.],
                             [0., 0., 0.01]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': 10.,
                    'H': np.array([[0., 1., 0.]]),
                    'noise': {'R': np.array([0.001]),
                              'mean': np.array([0.0])}
                    }
}


config.filter_ekf = {
    'dt': 0.0001,
    'spin_corr_const': 0.0,
    'frequency_decay_rate': 0.0,
    'x_0': np.array([0.0, 7.0, 8.05]),
    't_0': 0.,
    'P0': np.array([[0.01, 0., 0.],
                    [0., 0.01, 0.],
                    [0., 0., 0.01]]),
    'noise': {'Q': np.array([[0.01, 0., 0.],
                             [0., 0.01, 0.],
                             [0., 0., 0.01]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': 10.,
                    'H': np.array([[0., 1., 0.]]),
                    'dim_z': 1,
                    'R': np.array([[0.001]])}
}

config.simulationHajimolahoseini = {
    't_max': 6.,
    'dt': 0.0001,
    'A': 1.,
    'omega': 10.,
    'frequency_decay_rate': 0.0,  # frequency can behave according to OU process
    'x_0': np.array([5., 5., 10.]),  # initial state vector [Jy, Jz, omega]
    't_0': 0,
    'noise': {'Q': np.array([[1.0, 0., 0.],
                             [0., 1, 0.],
                             [0., 0., 1.]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': 1.,
                    'H': np.array([[1., 0., 0.]]),
                    'noise': {'R': np.array([0.001]),
                              'mean': np.array([0.0])}
                    }
}

config.filter_ekf_Hajimolahoseini = {
    'dt': 0.0001,
    'A': 1.,
    'omega': 10.,
    'spin_corr_const': 0.0,
    'frequency_decay_rate': 0.,
    'x_0': np.array([5., 5., 8., 1.]),
    't_0': 0,
    'P0': np.array([[1, 0., 0., 0.],
                    [0., 1, 0., 0.],
                    [0., 0., 1, 0.],
                    [0., 0., 0., 0.]]),
    'noise': {'Q': np.array([[1, 0., 0., 0.],
                             [0., 1, 0., 0.],
                             [0., 0., 1, 0],
                             [0., 0., 0., 0.]]),
              'mean': np.array([0.0, 0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': 1.,
                    'H': np.array([[1., 0., 0., 0.]]),
                    'dim_z': 1,
                    'R': np.array([[0.001]])}
}