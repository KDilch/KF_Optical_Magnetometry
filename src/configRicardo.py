from types import SimpleNamespace
import numpy as np
config = SimpleNamespace()
numAtoms = 10**12
qx = 0.002
qy = 0.002
T2 = 0.087  # in s x100
dt = 0.000001
measurement_strength = 3.5*10**(-1)  #x1000
R = 96.
omega_L = 62000.0  # in Hz

config.simulation = {
    't_max':  0.001,
    'dt': dt,
    'dim_measurement': 1,
    'T2': T2,
    'frequency_decay_rate': 0.0,  # frequency can behave according to OU process
    'x_0': np.array([numAtoms/2., 0., omega_L]),  # initial state vector [Jx, Jy, omega]
    't_0': 0,
    'noise': {'Q_jx': qx*numAtoms/T2,
              'Q_jy': qy*numAtoms/T2,
              'Q_freq': 0.0},
    'measurement': {'measurement_strength': measurement_strength,
                    'H': np.array([[0., 1., 0.]]),
                    'noise': {'R': R,
                              'mean': 0.0}
                    }
}

config.filter_ekf = {
    'dt': dt,
    'T2': T2,
    'frequency_decay_rate': 0.0,
    'inference_method': 'RK23',
    'x_0': np.array([numAtoms/2, 0., omega_L]),
    't_0': 0.,
    'P0': np.array([[0., 0., 0.],
                    [0., numAtoms, 0.],
                    [0., 0., 10000.]]),
    'noise': {'Q': np.array([[qx*numAtoms/T2, 0., 0.],
                             [0., qy*numAtoms/T2, 0.],
                             [0., 0., 0.]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': measurement_strength,
                    'H': np.array([[0., 1., 0.]]),
                    'dim_z': 1,
                    'R': np.array([[R]])}
}
