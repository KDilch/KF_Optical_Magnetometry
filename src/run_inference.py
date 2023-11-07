#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

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
from freq_inference import freq_from_autocorr, freq_from_fft, freq_from_periodogram, ipFFT
from plots import plot_simple_model
from MLE_omega import MLE_omega
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
            MLE_obj = MLE_omega(filter_params_ekf.numAtoms,
                                  filter_params_ekf.T2,
                                  filter_params_ekf.noise.Q[1][1]+filter_params_ekf.measurement.R[0][0],
                                  filter_params_ekf.dt, filter_params_ekf.dt)
            # CREATE A TIME ARRAY====================================================
            time_arr = np.arange(0, simulation_params.t_max, simulation_params.dt)

            # CREATE A Estimator and Covariance ARRAY====================================================
            x_ekf_est = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr])
            P_ekf_est = np.array([np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr])
            P_ss = np.array([np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr])
            MLE_omega_est = np.array(np.zeros(len(time_arr)))

            for index, z in enumerate(tqdm.tqdm(df.zs, desc='pid:%r' % os.getpid())):
                ekf.predict_update(z, compute_ss=args[0].ekf_ss)
                x_ekf_est[index] = ekf.x_est
                P_ekf_est[index] = ekf.P_est
                if args[0].ekf_ss:
                    P_ss[index] = ekf.steady_cov
                # MLE estimator implementation (uncomment to RUN)
                if index > 1: #and index<3000:
                    MLE_omega_est[index] = MLE_obj.find_MLE(x_ekf_est[0:index, 1])
                else:
                    MLE_omega_est[index] = 0.0


            df_output = prepare_df_from_inference(time_arr, df=df, xs_est=x_ekf_est, P_est=P_ekf_est, P_ss=P_ss, MLE=MLE_omega_est)
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

            # CREATE ARRAY TO STORE OUTPUT DATA
            fft_est_Larmour = None
            ipfft_est_Larmour = None
            autocorr_est_Larmour = None
            periodogram_est_Larmour = None
            if args[0].fft:
                fft_est_Larmour = np.zeros(len(time_arr_ekf))
            if args[0].ipfft:
                ipfft_est_Larmour = np.zeros(len(time_arr_ekf))
            if args[0].autocorrelation:
                autocorr_est_Larmour = np.zeros(len(time_arr_ekf))
            if args[0].periodogram:
                periodogram_est_Larmour = np.zeros(len(time_arr_ekf))

            index_ekf = 0
            for index, element in enumerate(tqdm.tqdm(df.zs, desc='pid:%r' % os.getpid())):
                if index % every_nth_z == 0:
                    ekf.predict_update(element/simulation_params.dt, Phi_Q_method=False)
                    x_ekf_est[index_ekf] = ekf.x_est
                    P_ekf_est[index_ekf] = ekf.P_est
                    x_filter_freq[index_ekf] = np.array([df.x0s[index], df.x1s[index], df.x2s[index]])
                    z_filter_freq[index_ekf] = element
                    if args[0].ekf_ss:
                        P_ss[index_ekf] = ekf.steady_cov

                    # Other methods of inference (start only after 20 initial points)
                    if index_ekf > 10000:
                        if args[0].ipfft:
                            ipfft_est_Larmour[index_ekf] = ipFFT(x_ekf_est[0:index_ekf, 1],
                                                                 1./filter_params_ekf.dt,
                                                                 10.0)
                        if args[0].fft:
                            fft_est_Larmour[index_ekf] = freq_from_fft(x_ekf_est[0:index_ekf, 1],
                                                                       1./filter_params_ekf.dt,
                                                                       window_name=None)

                        if args[0].autocorrelation:
                            autocorr_est_Larmour[index_ekf] = freq_from_autocorr(x_ekf_est[index_ekf-10000:index_ekf, 1],
                                                                                 1./filter_params_ekf.dt,
                                                                                 window_name=None)

                        if args[0].periodogram:
                            periodogram_est_Larmour[index_ekf] = freq_from_periodogram(x_ekf_est[index_ekf-10000:index_ekf, 1],
                                                                                       1./filter_params_ekf.dt,
                                                                                       window_name=None)

                    index_ekf += 1

            df_output = prepare_df(time_arr_ekf,
                                   xs=x_filter_freq,
                                   zs=z_filter_freq,
                                   xs_est=x_ekf_est,
                                   P_est=P_ekf_est,
                                   P_ss=P_ss,
                                   ftt_freq=fft_est_Larmour,
                                   autocorr_freq=autocorr_est_Larmour,
                                   periodogram=periodogram_est_Larmour,
                                   ipfft_freq=ipfft_est_Larmour)

            if args[0].save_data:
                save_data_simple_simulation(df_output, simulation_params, args[0].output_path +
                                            '/csv_inference_cd_ekf_sampling_%r' % filter_params_ekf.dt)
            if args[0].save_plots:
                plot_simple_model(df_output,
                                  dir_name=args[0].output_path + '/plots_inference_cd_ekf_filter_dt_%r' % filter_params_ekf.dt,
                                  params=simulation_params,
                                  simulation=True,
                                  ekf=args[0].cd_ekf,
                                  err=args[0].cd_ekf,
                                  err_loglog=args[0].cd_ekf,
                                  show=False,
                                  save=True)
