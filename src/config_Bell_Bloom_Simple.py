from types import SimpleNamespace
import numpy as np
config = SimpleNamespace()
numAtoms = 10.
qx = 0.1
qy = 0.1
T2 = 1000.
dt = 0.001

config.simulation = {
    't_max':  40.,
    'dt': dt,
    'dim_measurement': 1,
    'omega_pumping': 2.,
    'num_atoms': numAtoms,
    'pump_rate': 0.001,
    'T2': T2,
    'frequency_decay_rate': 0.0,  # frequency can behave according to OU process
    'x_0': np.array([0., numAtoms/2., 2.]),  # initial state vector [Jx, Jy, omega]
    't_0': 0,
    'noise': {'Q_jx': qx*numAtoms/T2,
              'Q_jy': qy*numAtoms/T2,
              'Q_freq': 0.0},
    'measurement': {'measurement_strength': 12.,
                    'H': np.array([[0., 1., 0.]]),
                    'noise': {'R': 0.1,
                              'mean': 0.0}
                    }
}

config.filter_ekf = {
    'dt': 5*dt,
    'T2': T2,
    'omega_pumping': 2.,
    'numAtoms': numAtoms,
    'num_atoms': numAtoms,
    'pump_rate': 0.001,
    'frequency_decay_rate': 0.0,
    'inference_method': 'RK23',
    'x_0': np.array([0., numAtoms/2, 2.]),
    't_0': 0.,
    'P0': np.array([[qx*numAtoms/T2, 0., 0.],
                    [0., qy*numAtoms/T2, 0.],
                    [0., 0., 10.]]),
    'noise': {'Q': np.array([[qx*numAtoms/T2, 0., 0.],
                             [0., qy*numAtoms/T2, 0.],
                             [0., 0., 0.]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': 12.,
                    'H': np.array([[0., 1., 0.]]),
                    'dim_z': 1,
                    'R': np.array([[0.1]])}
}
