#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
import os
import tqdm
import pandas as pd
from munch import DefaultMunch
from copy import deepcopy
import glob  # for matching a path REGEX
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
    file_list = glob.glob(data_path)

    logger.info('Loading a config file from path %r' % args[0].config)
    config = import_config_from_path(args[0].config)
    simulation_params = DefaultMunch.fromDict(deepcopy(config.simulation))
    filter_params_ekf = DefaultMunch.fromDict(deepcopy(config.filter_ekf))

    for file in file_list:
        df = pd.read_csv(file)
        if args[0].cc_ekf:
            ekf = MagnetometerEKF(model_params=filter_params_ekf)
            # CREATE A TIME ARRAY====================================================
            time_arr = np.arange(0, simulation_params.t_max, simulation_params.dt)

            # CREATE A Estimator and Covariance ARRAY====================================================
            x_ekf_est = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr])
            P_ekf_est = np.array([np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr])
            P_ss = np.array([np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr])

            for index, z in enumerate(tqdm.tqdm(df.zs, desc='pid:%r' % os.getpid())):
                ekf.predict_update(z)
                x_ekf_est[index] = ekf.x_est
                P_ekf_est[index] = ekf.P_est
                if args[0].ekf_ss:
                    P_ss[index] = ekf.steady_cov

            df_output = prepare_df_from_inference(time_arr, df=df, xs_est=x_ekf_est, P_est=P_ekf_est, P_ss=P_ss)
            if args[0].save_data:
                save_data_simple_simulation(df_output, simulation_params, args[0].output_path + '/csv_inference_cc_ekf')
            if args[0].save_plots:
                plot_simple_model(df_output,
                                  dir_name=args[0].output_path + '/plots_inference_cc_ekf',
                                  params=simulation_params,
                                  simulation=True,
                                  ekf=args[0].cc_ekf,
                                  err=args[0].cc_ekf,
                                  err_loglog=args[0].cc_ekf,
                                  show=False,
                                  save=True)

        if args[0].cd_ekf:
            ekf = CD_EKF(model_params=filter_params_ekf)
            # CREATE A TIME ARRAY AND COMPUTE SAMPLING FREQUENCY ====================================================
            every_nth_z = int(filter_params_ekf.dt/simulation_params.dt)
            time_arr_ekf = np.arange(0, simulation_params.t_max, filter_params_ekf.dt)

            # CREATE A Estimator and Covariance and simulation at filter frequency ARRAY================================
            x_ekf_est = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr_ekf])
            P_ekf_est = np.array(
                [np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr_ekf])
            P_ss = np.array([np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr_ekf])
            x_filter_freq = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr_ekf])
            z_filter_freq = np.zeros(len(time_arr_ekf))
            index_ekf = 0
            for index, element in enumerate(tqdm.tqdm(df.zs, desc='pid:%r' % os.getpid())):
                if index % every_nth_z == 0:
                    ekf.predict_update(element/simulation_params.dt, Phi_Q_method=False)
                    x_ekf_est[index_ekf] = ekf.x_est
                    P_ekf_est[index_ekf] = ekf.P_est
                    x_filter_freq[index_ekf] = np.array([df.x0s[index], df.x1s[index], df.x2s[index]])
                    z_filter_freq[index_ekf] = element
                    if args[0].ekf_ss:
                        pass
                    index_ekf += 1

            df_output = prepare_df(time_arr_ekf, xs=x_filter_freq, zs=z_filter_freq, xs_est=x_ekf_est, P_est=P_ekf_est, P_ss=P_ss)
            if args[0].save_data:
                save_data_simple_simulation(df_output, simulation_params, args[0].output_path +
                                            '/csv_inference_cd_ekf_sampling_%r' % filter_params_ekf.dt)
            if args[0].save_plots:
                plot_simple_model(df_output,
                                  dir_name=args[0].output_path + '/plots_inference_cd_ekf_filter_dt_%r'
                                           % filter_params_ekf.dt,
                                  params=simulation_params,
                                  simulation=True,
                                  ekf=args[0].cd_ekf,
                                  err=args[0].cd_ekf,
                                  err_loglog=args[0].cd_ekf,
                                  show=False,
                                  save=True)
