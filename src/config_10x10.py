from types import SimpleNamespace
import numpy as np

config = SimpleNamespace()

config.params = {
    'num_atoms': 10,
    'larmour_mean': 1.,
    'larmour_std': 0.1,
    't_max': 15.,
    't_0': 0,
    'dt': 0.001,
    'dim_measurement': 1,
    'measurement_strength': 0.1,  # Gamma
    'eta': 1.,
    'coll_decoherence': 0.1,
    'H': np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
}

config.simulation = {
    # initial state vector [Jx, Jy, Jz, vx, vy, vz, cxy, cyz, cxz, omega]
    'x_0': np.array([config.params['num_atoms'] / 2,
                      0.0,
                      0.0,
                      0.0,
                      config.params['num_atoms'] / 4,
                      config.params['num_atoms'] / 4,
                      0.0,
                      0.0,
                      0.0,
                      config.params['larmour_mean']]),
    'Q_freq': 0.0,
}

config.filter_ekf = {
    'x_0': np.array([config.params['num_atoms'] / 2,
                      0.0,
                      0.0,
                      0.0,
                      config.params['num_atoms'] / 4,
                      config.params['num_atoms'] / 4,
                      0.0,
                      0.0,
                      0.0,
                      config.params['larmour_mean']]),
    'P0': np.array([np.zeros(10),
                    np.zeros(10),
                    np.zeros(10),
                    np.zeros(10),
                    np.zeros(10),
                    np.zeros(10),
                    np.zeros(10),
                    np.zeros(10),
                    np.zeros(10),
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, config.params['larmour_std'] ** 2]
                    ]),
    'noise': {'Q': np.array([[1, 0.],
                             [0., config.simulation['Q_freq']]]),
              'S': np.sqrt(config.params['eta'])*np.array([[1.], [0.]])
              },
    'measurement': {'dim_z': 1,
                    'R': np.array([[config.params['measurement_strength'] *
                                    config.params['eta']]])
                    }
}
