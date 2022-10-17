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
from plots import plot_simple_model
from utilities.save_data import save_data_simple_simulation, prepare_df_from_inference


def run__magnetometer_inference(*args):
    # Logger for storing errors and logs in separate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-inference command.')
    data_path = os.path.join(args[0].data_path, '*.csv')

    logger.info('Loading a config file from path %r' % args[0].config)
    config = import_config_from_path(args[0].config)
    simulation_params = DefaultMunch.fromDict(deepcopy(config.simulation))
    filter_params_ekf = DefaultMunch.fromDict(deepcopy(config.filter_ekf))
    ekf = MagnetometerEKF(model_params=filter_params_ekf)

    ddf = dd.read_csv(data_path)
    config = import_config_from_path(args[0].config)
    filter_params_ekf = DefaultMunch.fromDict(deepcopy(config.filter_ekf))
    # CREATE A TIME ARRAY====================================================
    time_arr = np.arange(0, simulation_params.t_max, simulation_params.dt)

    x_ekf_est = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr])
    P_ekf_est = np.array([np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr])

    if args[0].cc_ekf:
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
        pass
    if args[0].save_data:
        pass
    if args[0].save_plots:
        pass

