
from types import SimpleNamespace
import numpy as np
config = SimpleNamespace()
numAtoms = 10**9
qx = 0.002
qy = 0.002
q_omega = 0.
T2 = 0.87  # in ms
dt = 0.000001
measurement_strength = 0.00176656 #0.00000000000177 #1.77*10**(-15)  # for power 500
measurement_strength_renorm = measurement_strength
R = 96. #in pikoampers
omega_L = 62.0  # in Hz /1000
omega_rand = np.random.normal(omega_L, 1)
config.simulation = {
    't_max':  3.,
    'dt': dt,
    'dim_measurement': 1,
    'T2': T2,
    'frequency_decay_rate': 0.0,  # frequency can behave according to OU process
    'x_0': np.array([numAtoms/2., 0., omega_L]),  # initial state vector [Jx, Jy, omega]
    't_0': 0,
    'noise': {'Q_jx': qx*numAtoms/T2,
              'Q_jy': qy*numAtoms/T2,
              'Q_freq': q_omega},
    'measurement': {'measurement_strength': measurement_strength_renorm,
                    'H': np.array([[0., 1., 0.]]),
                    'noise': {'R': R,
                              'mean': 0.0}
                    }
}

config.filter_ekf = {
    'dt': 5*dt,
    'T2': T2,
    'frequency_decay_rate': 0.0,
    'inference_method': 'RK23',
    'x_0': np.array([numAtoms/2, 0., omega_rand]),
    't_0': 0.,
    'P0': np.array([[0., 0., 0.],
                    [0., numAtoms, 0.],
                    [0., 0., 1.]]),
    'noise': {'Q': np.array([[qx*numAtoms/T2, 0., 0.],
                             [0., qy*numAtoms/T2, 0.],
                             [0., 0., q_omega]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': measurement_strength_renorm,
                    'H': np.array([[0., 1., 0.]]),
                    'dim_z': 1,
                    'R': np.array([[R]])}
}
