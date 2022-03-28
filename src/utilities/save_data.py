import pandas as pd
import os
from datetime import datetime


def save_data_simple_simulation(df, params, dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    date = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')

    df.to_csv(os.path.join(dir_name, '%s_pid_%r_omega_%r_decoherence_x_y_%r_%r_dt_%r.csv' % (date,
                                                                                             os.getpid(),
                                                                                             params.x_0[2],
                                                                                             params.decoherence_x,
                                                                                             params.decoherence_y,
                                                                                             params.dt
                                                                                             )))


def prepare_df(time_arr, xs, xs_est=None, P_est=None):
    df = pd.DataFrame({'time': time_arr,
                       'x0s': xs[:, 0],
                       'x1s': xs[:, 1],
                       'x2s': xs[:, 2]
                       })

    if (xs_est is not None) and (P_est is not None):
        mse0 = (xs[:, 0] - xs_est[:, 0]) ** 2
        mse1 = (xs[:, 1] - xs_est[:, 1]) ** 2
        mse2 = (xs[:, 2] - xs_est[:, 2]) ** 2
        df['x0s_est'] = xs_est[:, 0]
        df['x1s_est'] = xs_est[:, 1]
        df['x2s_est'] = xs_est[:, 2]
        df['x0_err_cov'] = P_est[:, 0, 0]
        df['x1_err_cov'] = P_est[:, 1, 1]
        df['x2_err_cov'] = P_est[:, 2, 2]
        df['mse_x0'] = mse0
        df['mse_x1'] = mse1
        df['mse_x2'] = mse2
    return df
