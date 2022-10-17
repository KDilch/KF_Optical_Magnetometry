#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
import os
import tqdm
import dask.dataframe as dd
from tqdm.dask import TqdmCallback
from munch import DefaultMunch
from copy import deepcopy

from utilities.config_util import import_config_from_path
from kalman_filter.continuous.simple_model_ekf import MagnetometerEKF
from kalman_filter.continuous.simple_model_cd_ekf import CD_EKF
from plots import plot_simple_model
from utilities.save_data import save_data_simple_simulation, prepare_df_from_inference, prepare_df


def run__magnetometer_inference(*args):
    # Logger for storing errors and logs in separate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-inference command.')
    data_path = os.path.join(args[0].data_path, '*.csv')

    logger.info('Loading a config file from path %r' % args[0].config)
    config = import_config_from_path(args[0].config)
    simulation_params = DefaultMunch.fromDict(deepcopy(config.simulation))
    filter_params_ekf = DefaultMunch.fromDict(deepcopy(config.filter_ekf))

    ddf = dd.read_csv(data_path)
    config = import_config_from_path(args[0].config)
    filter_params_ekf = DefaultMunch.fromDict(deepcopy(config.filter_ekf))

    if args[0].cc_ekf:
        ekf = MagnetometerEKF(model_params=filter_params_ekf)
        # CREATE A TIME ARRAY====================================================
        time_arr = np.arange(0, simulation_params.t_max, simulation_params.dt)

        x_ekf_est = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr])
        P_ekf_est = np.array([np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr])

        for index, z in enumerate(tqdm.tqdm(ddf.zs, desc='pid:%r' % os.getpid())):
            ekf.predict_update(z)
            x_ekf_est[index] = ekf.x_est
            P_ekf_est[index] = ekf.P_est

        df = prepare_df_from_inference(time_arr, df=ddf, xs_est=x_ekf_est, P_est=P_ekf_est)
        if args[0].save_data:
            save_data_simple_simulation(df, simulation_params, args[0].output_path + '/csv_inference')
        if args[0].save_plots:
            plot_simple_model(df,
                              dir_name=args[0].output_path + '/plots_inference',
                              params=simulation_params,
                              simulation=True,
                              ekf=args[0].cc_ekf,
                              err=args[0].cc_ekf,
                              err_loglog=args[0].cc_ekf,
                              show=False,
                              save=True)

    if args[0].cd_ekf:
        ekf = CD_EKF(model_params=filter_params_ekf)
        # CREATE A TIME ARRAY====================================================
        every_nth_z = np.int(np.floor_divide(filter_params_ekf.dt, simulation_params.dt))

        time_arr_ekf = np.arange(0, simulation_params.t_max, filter_params_ekf.dt)
        x_ekf_est = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr_ekf])
        P_ekf_est = np.array([np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr_ekf])
        x_filter_freq = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr_ekf])
        z_filter_freq = np.zeros(len(time_arr_ekf))
        index_ekf = 0
        for index, el in enumerate(tqdm.tqdm(ddf.iterrows(), desc='pid:%r' % os.getpid())):
            if index % every_nth_z == 0:
                ekf.predict_update(el[1]['zs']/simulation_params.dt)
                x_ekf_est[index_ekf] = ekf.x_est
                P_ekf_est[index_ekf] = ekf.P_est
                x_filter_freq[index_ekf] = np.array([el[1]['x0s'], el[1]['x1s'], el[1]['x2s']])
                z_filter_freq[index_ekf] = el[1]['zs']
                index_ekf += 1

        df = prepare_df(time_arr_ekf, xs=x_filter_freq, zs=z_filter_freq, xs_est=x_ekf_est, P_est=P_ekf_est)
        if args[0].save_data:
            save_data_simple_simulation(df, simulation_params, args[0].output_path + '/csv_inference')
        if args[0].save_plots:
            plot_simple_model(df,
                              dir_name=args[0].output_path + '/plots_inference_cd',
                              params=simulation_params,
                              simulation=True,
                              ekf=args[0].cd_ekf,
                              err=args[0].cd_ekf,
                              err_loglog=args[0].cd_ekf,
                              show=False,
                              save=True)

