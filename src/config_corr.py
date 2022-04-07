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
              'Q_freq': 0.0,
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': 1., # eta
                    'H': np.array([[0., 1., 0.]]),
                    'noise': {'mean': 0.0}
                    }
}

config.filter_ekf = {
    'dt': config.simulation['dt'],
    'decoherence_x': config.simulation['decoherence_x'],
    'decoherence_y': config.simulation['decoherence_y'],
    'frequency_decay_rate': config.simulation['frequency_decay_rate'],
    't_0': config.simulation['t_0'],
    'x_0': np.array([5.0, 0.0, 8.1]),
    'P0': np.array([[0.0, 0., 0.],
                    [0., 0.0, 0.],
                    [0., 0., 0.01]]),
    'noise': {'Q': np.array([[config.simulation['noise']['Q_m'], config.simulation['noise']['Q_m'], 0.],
                             [config.simulation['noise']['Q_m'], config.simulation['noise']['Q_m'], 0.],
                             [0., 0., config.simulation['noise']['Q_freq']]]),
              'mean': np.array([0.0, 0.0, 0.0]),
              'B': np.identity(3),
              'S': np.multiply(np.sqrt(config.simulation['measurement']['measurement_strength']), np.array([[config.simulation['noise']['Q_m'],
                                                                                                            config.simulation['noise']['Q_m'],
                                                                                                            0.]]))
              },
    'measurement': {'measurement_strength': config.simulation['measurement']['measurement_strength'],
                    'H': config.simulation['measurement']['H'],
                    'dim_z': 1,
                    'R': np.array([[config.simulation['measurement']['measurement_strength']*config.simulation['noise']['Q_m']]])
                    }
}
