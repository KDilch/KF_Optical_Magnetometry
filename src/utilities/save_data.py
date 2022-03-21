import pandas as pd
import os
from time import time

def save_simulation_and_est_data(t_arr, xs, xs_est, zs, simulation_params):
    pid = os.getpid()
    dir_path = 'data/raw_data'
    mse0 = (xs[:, 0] - xs_est[:, 0]) ** 2
    mse1 = (xs[:, 1] - xs_est[:, 1]) ** 2
    mse2 = (xs[:, 2] - xs_est[:, 2]) ** 2

    xs_df = pd.DataFrame({'t_arr': t_arr,
                          'x0s': xs[:, 0],
                          'x1s': xs[:, 1],
                          'x2s': xs[:, 2],
                          'x0s_est': xs_est[:, 0],
                          'x1s_est': xs_est[:, 1],
                          'x2s_est': xs_est[:, 2],
                          'MSE_x0': mse0,
                          'MSE_x1': mse1,
                          'MSE_x2': mse2,
                          'zs': zs
                          })
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    xs_df.to_csv(os.path.join(dir_path, 'data_omega%r_spin_corr%r_%r_pid%r.csv' % (simulation_params.x_0[2],
                                                                                  simulation_params.spin_corr_const,
                                                                                  time(),
                                                                                  pid)))

