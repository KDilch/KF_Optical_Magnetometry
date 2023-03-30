import pandas as pd
import os
from datetime import datetime

from tqdm.dask import TqdmCallback


def save_data_simple_simulation(df, params, dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    date = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')

    df.to_csv(os.path.join(dir_name, '%s_pid_%r_omega_%r_T2_%r_dt_%r.csv' % (date,
                                                                             os.getpid(),
                                                                             params.x_0[2],
                                                                             params.T2,
                                                                             params.dt
                                                                             )))


def prepare_df(time_arr, xs, zs, xs_est=None, P_est=None, P_ss=None, ftt_freq=None, autocorr_freq=None, periodogram=None):
    df = pd.DataFrame({'time': time_arr,
                       'zs': zs,
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
        if P_ss is not None:
            df['ss_x0'] = P_est[:, 0, 0]
            df['ss_x1'] = P_est[:, 1, 1]
            df['ss_x2'] = P_est[:, 2, 2]
        if ftt_freq is not None:
            df['fft_x2'] = ftt_freq
            df['fft_x2_err_sq'] = (xs[:, 2] - ftt_freq) ** 2
        if autocorr_freq is not None:
            df['autocorr_x2'] = autocorr_freq
            df['autocorr_x2_err_sq'] = (xs[:, 2] - autocorr_freq) ** 2
        if periodogram is not None:
            df['periodogram_x2'] = periodogram
            df['periodogram_x2_err_sq'] = (xs[:, 2] - periodogram) ** 2
    return df

def prepare_df_from_inference(time_arr, df, xs_est=None, P_est=None, P_ss=None):
    df = pd.DataFrame({'time': time_arr,
                       'zs': df.zs,
                       'x0s': df.x0s,
                       'x1s': df.x1s,
                       'x2s': df.x2s
                       })

    if (xs_est is not None) and (P_est is not None):
        from decimal import Decimal

        mse0 = (df.x0s - xs_est[:, 0]) ** 2
        mse1 = (df.x1s - xs_est[:, 1]) ** 2
        mse2 = (df.x2s - xs_est[:, 2]) ** 2
        df['x0s_est'] = xs_est[:, 0]
        df['x1s_est'] = xs_est[:, 1]
        df['x2s_est'] = xs_est[:, 2]
        df['x0_err_cov'] = P_est[:, 0, 0]
        df['x1_err_cov'] = P_est[:, 1, 1]
        df['x2_err_cov'] = P_est[:, 2, 2]
        df['mse_x0'] = mse0
        df['mse_x1'] = mse1
        df['mse_x2'] = mse2
        if P_ss is not None:
            df['ss_x0'] = P_est[:, 0, 0]
            df['ss_x1'] = P_est[:, 1, 1]
            df['ss_x2'] = P_est[:, 2, 2]
    return df


def prepare_avg_ddf(ddf):
    groupedByTime = ddf.groupby('time')
    time_arr = groupedByTime.time.mean().compute()
    with TqdmCallback(desc="Calculating x0_est avg"):
        x0s_est_avg = groupedByTime.x0s_est.mean().compute()
    with TqdmCallback(desc="Calculating x1_est avg"):
        x1s_est_avg = groupedByTime.x1s_est.mean().compute()
    with TqdmCallback(desc="Calculating x2_est avg"):
        x2s_est_avg = groupedByTime.x2s_est.mean().compute()
    with TqdmCallback(desc="Calculating x0 avg"):
        x0s_avg = groupedByTime.x0s.mean().compute()
    with TqdmCallback(desc="Calculating x1 avg"):
        x1s_avg = groupedByTime.x1s.mean().compute()
    with TqdmCallback(desc="Calculating x2 avg"):
        x2s_avg = groupedByTime.x2s.mean().compute()
    with TqdmCallback(desc="Calculating x0 avg cov"):
        x0s_est_avg_err = groupedByTime.x0_err_cov.mean().compute()
    with TqdmCallback(desc="Calculating x1 avg cov"):
        x1s_est_avg_err = groupedByTime.x1_err_cov.mean().compute()
    with TqdmCallback(desc="Calculating x2 avg cpv"):
        x2s_est_avg_err = groupedByTime.x2_err_cov.mean().compute()
    with TqdmCallback(desc="Calculating x0 avg err"):
        x0s_avg_err = groupedByTime.mse_x0.mean().compute()
    with TqdmCallback(desc="Calculating x1 avg err"):
        x1s_avg_err = groupedByTime.mse_x1.mean().compute()
    with TqdmCallback(desc="Calculating x2 avg err"):
        x2s_avg_err = groupedByTime.mse_x2.mean().compute()
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

    df.to_csv(os.path.join(dir_name, 'avg_%s_pid_%r_num_reps_%r.csv' % (date,
                                                                        os.getpid(),
                                                                        num_reps
                                                                        ))
              )
