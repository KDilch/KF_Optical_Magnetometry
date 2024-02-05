from types import SimpleNamespace
import numpy as np

config = SimpleNamespace()
numAtoms = 8.2 * 10. ** 12
qx = 0.002
qy = 0.002
T2 = 0.87  # in ms
dt_sim = 5. * 10 ** (-6)  # in ms # 1.* 10 ** (-6)->also have this simulation
dt_exp = 5. * 10 ** (-4)  # in ms
t_max = 0.43  # in ms
omega_B_initial = 2 * np.pi * 29.07  # in kHz
omega_pump = 2 * np.pi * 29.07  # in kHz
pump_amplitude = 100. #check with theory ? power
measurement_scaling_factor = 0.00000000176656  # gain_scale^(-1)
R = 0.0096

config.simulation = {
    't_max': t_max,
    'dt': dt_sim,
    'dim_measurement': 1,
    'pump_amplitude': pump_amplitude,
    'omega_pumping': omega_pump,
    'num_atoms': numAtoms,
    'T2': T2,
    'x_0': np.array([0., numAtoms / 2., omega_B_initial]),  # initial state vector [Jx, Jy, omega]
    't_0': 0,
    'noise': {'Q_jx': qx * numAtoms / T2,
              'Q_jy': qy * numAtoms / T2,
              'Q_freq': 0.0},
    'measurement': {'measurement_strength': measurement_scaling_factor,
                    'H': np.array([[0., 1., 0.]]),
                    'noise': {'R': R,
                              'mean': 0.0}
                    }
}

config.filter_ekf = {
    'dt': dt_exp,
    'T2': T2,
    'omega_pumping': omega_pump,
    'pump_amplitude': pump_amplitude,
    'num_atoms': numAtoms,
    'inference_method': 'RK23',
    'x_0': np.array([0., numAtoms / 2, omega_B_initial]),
    't_0': 0.,
    'P0': np.array([[qx * numAtoms / T2, 0., 0.],
                    [0., qy * numAtoms / T2, 0.],
                    [0., 0., 100.]]),
    'noise': {'Q': np.array([[qx * numAtoms / T2, 0., 0.],
                             [0., qy * numAtoms / T2, 0.],
                             [0., 0., 0.]]),
              'mean': np.array([0.0, 0.0, 0.0])},
    'measurement': {'measurement_strength': measurement_scaling_factor,
                    'H': np.array([[0., 1., 0.]]),
                    'dim_z': 1,
                    'R': np.array([[R]])}
}
