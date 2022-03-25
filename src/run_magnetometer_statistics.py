#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from munch import DefaultMunch
from copy import deepcopy
import numpy as np
from time import time
import pandas as pd
from multiprocessing import Pool

from utilities.config_util import import_config_from_path
from run_magnetometer import run__magnetometer
from plots import plot_avg_mse_from_dataframes, plot_avg_xs_from_dataframes,\
    plot_xs_sim_ekf_cont, plot_mse_sim_ekf_cont, plot_avg_omega_with_fft_from_dataframes, plot_avg_freq_from_dataframes, plot_avg_mse_loglog_from_dataframes


def run__magnetometer_statistics(*args):
    # Logger for storing errors and logs in separate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-magnetometer-statistics command.')

    logger.info('Loading a config file from path %r' % args[0].config)
    config = import_config_from_path(args[0].config)
    simulation_params = DefaultMunch.fromDict(config.simulation)

    logger.info('Setting simulation parameters to delta_t_simulation = %r, t_max=%r.' %
                (str(simulation_params.dt),
                 str(simulation_params.t_max)
                 )
                )
    num_trajectories = 10

    args_list = [deepcopy(args) for i in range(num_trajectories)]

    num_iter = int(simulation_params.t_max/simulation_params.dt)
    columns_simulation = ['xs_%r' % i for i in range(num_trajectories)]
    columns_estimator = ['xs_est_%r' % i for i in range(num_trajectories)]
    columns_avgMSE = ['MSE_%r' % i for i in range(num_trajectories)]
    columns_cov_ekf = ['cov_ekf_%r' % i for i in range(num_trajectories)]
    simulation_x0_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_simulation)
    ekf_x0_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_estimator)
    ekf_cov_x0_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_cov_ekf)
    mse_x0_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_avgMSE)
    simulation_x1_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_simulation)
    ekf_x1_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_estimator)
    ekf_cov_x1_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_cov_ekf)
    mse_x1_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_avgMSE)
    simulation_x2_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_simulation)
    ekf_x2_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_estimator)
    mse_x2_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_avgMSE)
    ekf_cov_x2_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_cov_ekf)


    # #MULTIPROCESSING
    pool = Pool(10)
    with pool:
        result = pool.starmap(run__magnetometer, args_list)
    xs, xs_est, zs, P_ekf = list(zip(*result))

    for i in range(num_trajectories):
        x0s_df = pd.DataFrame({'xs_%r' % i: xs[0][:, 0]})
        x0s_est_df = pd.DataFrame({'xs_est_%r' % i: xs_est[0][:, 0]})
        x1s_df = pd.DataFrame({'xs_%r' % i: xs[0][:, 1]})
        x1s_est_df = pd.DataFrame({'xs_est_%r' % i: xs_est[0][:, 1]})
        x2s_df = pd.DataFrame({'xs_%r' % i: xs[0][:, 2]})
        x2s_est_df = pd.DataFrame({'xs_est_%r' % i: xs_est[0][:, 2]})
        mse0_ekf_df = pd.DataFrame({'cov_ekf_%r' % i: P_ekf[0][:, 0, 0]})
        mse1_ekf_df = pd.DataFrame({'cov_ekf_%r' % i: P_ekf[0][:, 1, 1]})
        mse2_ekf_df = pd.DataFrame({'cov_ekf_%r' % i: P_ekf[0][:, 2, 2]})
        mse0 = (xs[0][:, 0] - xs_est[0][:, 0])**2
        mse0_df = pd.DataFrame({'MSE_%r' % i: mse0})
        mse1 = (xs[0][:, 1] - xs_est[0][:, 1]) ** 2
        mse1_df = pd.DataFrame({'MSE_%r' % i: mse1})
        mse2 = (xs[0][:, 2] - xs_est[0][:, 2]) ** 2
        mse2_df = pd.DataFrame({'MSE_%r' % i: mse2})
        simulation_x0_data['xs_%r' % i] = x0s_df['xs_%r' % i]
        ekf_x0_data['xs_est_%r' % i] = x0s_est_df['xs_est_%r' % i]
        mse_x0_data['MSE_%r' % i] = mse0_df['MSE_%r' % i]
        simulation_x1_data['xs_%r' % i] = x1s_df['xs_%r' % i]
        ekf_x1_data['xs_est_%r' % i] = x1s_est_df['xs_est_%r' % i]
        mse_x1_data['MSE_%r' % i] = mse1_df['MSE_%r' % i]
        simulation_x2_data['xs_%r' % i] = x2s_df['xs_%r' % i]
        ekf_x2_data['xs_est_%r' % i] = x2s_est_df['xs_est_%r' % i]
        mse_x2_data['MSE_%r' % i] = mse2_df['MSE_%r' % i]
        ekf_cov_x0_data['cov_ekf_%r' % i] = mse0_ekf_df['cov_ekf_%r' % i]
        ekf_cov_x1_data['cov_ekf_%r' % i] = mse1_ekf_df['cov_ekf_%r' % i]
        ekf_cov_x2_data['cov_ekf_%r' % i] = mse2_ekf_df['cov_ekf_%r' % i]

    time_arr = np.arange(0, simulation_params.t_max, simulation_params.dt)
    plot_avg_xs_from_dataframes(time_arr,
                                simulation_x0_data,
                                simulation_x1_data,
                                simulation_x2_data,
                                ekf_x0_data,
                                ekf_x1_data,
                                ekf_x2_data,
                                simulation_params)

    plot_avg_mse_from_dataframes(time_arr,
                                 mse_x0_data,
                                 mse_x1_data,
                                 mse_x2_data,
                                 ekf_cov_x0_data,
                                 ekf_cov_x1_data,
                                 ekf_cov_x2_data,
                                 simulation_params)

    plot_avg_freq_from_dataframes(time_arr,
                                  simulation_x2_data,
                                  ekf_x2_data,
                                  mse_x2_data,
                                  ekf_cov_x2_data,
                                  simulation_params)

    plot_avg_mse_loglog_from_dataframes(time_arr,
                                 mse_x0_data,
                                 mse_x1_data,
                                 mse_x2_data,
                                 ekf_cov_x0_data,
                                 ekf_cov_x1_data,
                                 ekf_cov_x2_data,
                                 simulation_params)

    # plot_avg_omega_with_fft_from_dataframes(time_arr,
    #                                         simulation_x2_data,
    #                                         ekf_x2_data,
    #                                         x2s_fft_df,
    #                                         x2s_fft_of_ekf_df,
    #                                         mse_x2_data,
    #                                         mse2_fft_df,
    #                                         mse2_fft_ekf_df,
    #                                         simulation_params)

    simulation_x0_data.to_csv('data/raw_data/sim_x0_omega%r_spin_corr%r_%r_num_iter%r.csv' % (simulation_params.x_0[2],
                                                                                              simulation_params.spin_corr_const,
                                                                                              time(),
                                                                                              mse_x0_data.ndim))
    ekf_x0_data.to_csv('data/raw_data/ekf_x0_omega%r_spin_corr%r_%r_num_iter%r.csv' % (simulation_params.x_0[2],
                                                                                       simulation_params.spin_corr_const,
                                                                                       time(),
                                                                                       mse_x0_data.ndim))
    mse_x0_data.to_csv('data/raw_data/mse_x0_omega%r_spin_corr%r_%r_num_iter%r.csv' % (simulation_params.x_0[2],
                                                                            simulation_params.spin_corr_const,
                                                                            time(),
                                                                            mse_x0_data.ndim))
    simulation_x1_data.to_csv('data/raw_data/sim_x1_omega%r_spin_corr%r_%r_num_iter%r.csv' % (simulation_params.x_0[2],
                                                                            simulation_params.spin_corr_const,
                                                                            time(),
                                                                            mse_x0_data.ndim))
    ekf_x1_data.to_csv('data/raw_data/ekf_x1_omega%r_spin_corr%r_%r_num_iter%r.csv' % (simulation_params.x_0[2],
                                                                            simulation_params.spin_corr_const,
                                                                            time(),
                                                                            mse_x0_data.ndim))
    mse_x1_data.to_csv('data/raw_data/mse_x2_omega%r_spin_corr%r_%r_num_iter%r.csv' % (simulation_params.x_0[2],
                                                                            simulation_params.spin_corr_const,
                                                                            time(),
                                                                            mse_x0_data.ndim))
    simulation_x2_data.to_csv('data/raw_data/sim_x2_omega%r_spin_corr%r_%r_num_iter%r.csv' % (simulation_params.x_0[2],
                                                                            simulation_params.spin_corr_const,
                                                                            time(),
                                                                            mse_x0_data.ndim))
    mse_x2_data.to_csv('data/raw_data/mse_x2_omega%r_spin_corr%r_%r_num_iter%r.csv' % (simulation_params.x_0[2],
                                                                            simulation_params.spin_corr_const,
                                                                            time(),
                                                                            mse_x0_data.ndim))

