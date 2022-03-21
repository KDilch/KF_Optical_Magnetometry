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
    plot_xs_sim_ekf_cont, plot_mse_sim_ekf_cont, plot_avg_omega_with_fft_from_dataframes


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
    num_trajectories = 5

    args_list = [deepcopy(args) for i in range(num_trajectories)]

    num_iter = int(simulation_params.t_max/simulation_params.dt)
    columns_simulation = ['xs_%r' % i for i in range(num_trajectories)]
    columns_estimator = ['xs_est_%r' % i for i in range(num_trajectories)]
    columns_avgMSE = ['MSE_%r' % i for i in range(num_trajectories)]
    simulation_x0_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_simulation)
    ekf_x0_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_estimator)
    mse_x0_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_avgMSE)
    simulation_x1_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_simulation)
    ekf_x1_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_estimator)
    mse_x1_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_avgMSE)
    simulation_x2_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_simulation)
    ekf_x2_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_estimator)
    mse_x2_data = pd.DataFrame(data=np.empty((num_iter, num_trajectories)), columns=columns_avgMSE)

    # #MULTIPROCESSING
    pool = Pool(8)
    with pool:
        result = pool.starmap(run__magnetometer, args_list)
    xs, xs_est, x_fft_est, x_fft_from_ekf_est, zs = list(zip(*result))



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


