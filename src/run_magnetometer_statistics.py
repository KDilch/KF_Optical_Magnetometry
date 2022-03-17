#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from munch import DefaultMunch
import numpy as np
import pandas as pd

from utilities.config_util import import_config_from_path
from run_magnetometer import run__magnetometer


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
    num_trajectories = 20
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
    for i in range(num_trajectories):
        xs, xs_est = run__magnetometer(*args)
        x0s_df = pd.DataFrame({'xs_%r' % i: xs[:, 0]})
        x0s_est_df = pd.DataFrame({'xs_est_%r' % i: xs_est[:, 0]})
        x1s_df = pd.DataFrame({'xs_%r' % i: xs[:, 1]})
        x1s_est_df = pd.DataFrame({'xs_est_%r' % i: xs_est[:, 1]})
        x2s_df = pd.DataFrame({'xs_%r' % i: xs[:, 2]})
        x2s_est_df = pd.DataFrame({'xs_est_%r' % i: xs_est[:, 2]})
        mse0 = (xs[:, 0] - xs_est[:, 0])**2
        mse0_df = pd.DataFrame({'MSE_%r' % i: mse0})
        mse1 = (xs[:, 1] - xs_est[:, 1]) ** 2
        mse1_df = pd.DataFrame({'MSE_%r' % i: mse1})
        mse2 = (xs[:, 2] - xs_est[:, 2]) ** 2
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

    mean_mse = mse_x2_data.mean(axis=1)
    time = np.arange(0, simulation_params.t_max, simulation_params.dt)
    import matplotlib.pyplot as plt
    plt.plot(time, mean_mse)
    plt.show()


    # ekf_data.append(xs_est)