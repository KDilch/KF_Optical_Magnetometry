from types import SimpleNamespace
import numpy as np
config = SimpleNamespace()
numAtoms = 10.
qx = 0.01
qy = 0.01
qB = 0.0
T2 = 100.
dt = 0.001
freq = 1.
gD = 0.01
R = 0.01
config.simulation = {
    't_max':  500.,
    'dt': dt,
    'dim_measurement': 1,
    'T2': T2,
    'frequency_decay_rate': 0.0,  # frequency can behave according to OU process
    'x_0': np.array([numAtoms/2., 0., freq]),  # initial state vector [Jx, Jy, omega]
    't_0': 0,
    'noise': {'Q_jx': qx*numAtoms/T2,
              'Q_jy': qy*numAtoms/T2,
              'Q_freq': qB},
    'measurement': {'measurement_strength': gD,
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
    'x_0': np.array([numAtoms/2, 0., 2*freq]),
    't_0': 0.,
    'P0': np.array([[qx*numAtoms/T2, 0., 0.],
                    [0., qy*numAtoms/T2, 0.],
                    [0., 0., freq]]),
    'noise': {'Q': np.array([[qx*numAtoms/T2, 0., 0.],
                             [0., qy*numAtoms/T2, 0.],
                             [0., 0., qB]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': gD,
                    'H': np.array([[0., 1., 0.]]),
                    'dim_z': 1,
                    'R': np.array([[R]])}
}
