from types import SimpleNamespace
import numpy as np
config = SimpleNamespace()

config.simulation = {
    't_max':  30.,
    'dt': 0.001,
    'dim_measurement': 1,
    'decoherence_x': 0.0,
    'decoherence_y': 0.0,
    'frequency_decay_rate': 0.0,  # frequency can behave according to OU process
    'x_0': np.array([5., 0., 8.]),  # initial state vector [Jx, Jy, omega]
    't_0': 0,
    'noise': {'Q_m': 0.01,
              'Q_freq': 0.01},
    'measurement': {'measurement_strength': 1., # eta
                    'H': np.array([[0., 1., 0.]])
                    },
    'filter': {}
}

config.filter_ekf = {
    'x_0': np.array([5.0, 0.0, 8.1]),
    'P0': np.array([[0.0, 0., 0.],
                    [0., 0.0, 0.],
                    [0., 0., 0.01]]),
    'noise': {'B': np.identity(3)}}
