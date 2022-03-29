import pandas as pd
import os
from datetime import datetime
import dask.dataframe as dd


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


def prepare_avg_ddf(ddf):
    time_arr = ddf.groupby('time').time.mean().compute()
    x0s_est_avg = ddf.groupby('time').x0s_est.mean().compute()
    x1s_est_avg = ddf.groupby('time').x1s_est.mean().compute()
    x2s_est_avg = ddf.groupby('time').x2s_est.mean().compute()
    x0s_avg = ddf.groupby('time').x0s.mean().compute()
    x1s_avg = ddf.groupby('time').x1s.mean().compute()
    x2s_avg = ddf.groupby('time').x2s.mean().compute()
    x0s_est_avg_err = ddf.groupby('time').x0_err_cov.mean().compute()
    x1s_est_avg_err = ddf.groupby('time').x1_err_cov.mean().compute()
    x2s_est_avg_err = ddf.groupby('time').x2_err_cov.mean().compute()
    x0s_avg_err = ddf.groupby('time').mse_x0.mean().compute()
    x1s_avg_err = ddf.groupby('time').mse_x1.mean().compute()
    x2s_avg_err = ddf.groupby('time').mse_x2.mean().compute()
    list_of_avg_series = [time_arr,
                          x0s_est_avg,
                          x1s_est_avg,
                          x2s_est_avg,
                          x0s_avg,
                          x1s_avg,
                          x2s_avg,
                          x0s_est_avg_err,
                          x1s_est_avg_err,
                          x2s_est_avg_err,
                          x0s_avg_err,
                          x1s_avg_err,
                          x2s_avg_err]
    avg_ddf = list_of_avg_series[0].to_frame()
    for el in list_of_avg_series[1:]:
        avg_ddf[el.name] = el
    return avg_ddf


def save_data_avg_simple_simulation(df, dir_name, num_reps):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    date = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')

    df.to_csv(os.path.join(dir_name, 'avgs_%s_pid_%r_num_reps_%r.csv' % (date,
                                                                         os.getpid(),
                                                                         num_reps
                                                                         ))
              )
