#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
from copy import deepcopy
from munch import DefaultMunch

from utilities.config_util import import_config_from_path
from dynamics.atomic_sensor_simulation_model import Atomic_Sensor_Simulation_Model
from measurement.atomicsensormeasurementmodel import AtomicSensorMeasurementModel
from kalman_filter.continuous.magnetometer_ekf import MagnetometerEKF
from utilities.fft import perform_discrete_fft
from utilities.time_arr import initialize_time_arrays
from plots import plot_mse_sim_ekf_cont, plot_xs_sim_ekf_cont


def run__magnetometer(*args):
    # Logger for storing errors and logs in separate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-frequency-extractor command.')

    logger.info('Loading a config file from path %r' % args[0].config)
    config = import_config_from_path(args[0].config)
    simulation_params = DefaultMunch.fromDict(deepcopy(config.simulation))

    logger.info('Setting simulation parameters to delta_t_simulation = %r, t_max=%r.' %
                (str(simulation_params.dt),
                 str(simulation_params.t_max)
                 )
                )
    filter_params_ekf = DefaultMunch.fromDict(deepcopy(config.filter_ekf))
    logger.info('Setting filter parameters to delta_t_filter = %r.' %
                (str(filter_params_ekf.dt)
                 )
                )

    logger.info('Setting initial state vec to  [%r].' %
                (str(config.simulation['x_0'])))

    logger.info('Setting initial ekf state vec to  [%r].' %
                (str(config.filter_ekf['x_0'])))

    logger.info('Setting filter Q, H and R to Q = %r, H = %r, R = %r' %
                (str(filter_params_ekf.noise.Q),
                 str(filter_params_ekf.measurement.H),
                 str(filter_params_ekf.measurement.R)
                 )
                )

    # continuous measurement for now
    time_arr_simulation, time_arr_filter, every_nth_z = initialize_time_arrays(simulation_params, filter_params_ekf)

    # SIMULATE THE DYNAMICS=====================================================
    simulation_dynamical_model = Atomic_Sensor_Simulation_Model(t=0,
                                                                simulation_params=simulation_params)

    xs = np.array([np.array((simulation_dynamical_model.step())) for _ in time_arr_simulation])

    measurement_model = AtomicSensorMeasurementModel(simulation_params)

    dz_s = np.array([np.array((measurement_model.read_sensor(_))) for _ in xs])  # noisy measurement
    dz_s_filter_freq = dz_s[::every_nth_z]

    # KALMAN FILTER====================================================
    logger.info("Initializing ekf_magnetometer")

    ekf = MagnetometerEKF(model_params=filter_params_ekf)

    x_ekf_est = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr_filter])
    x_fft_est = np.array([0 for _ in time_arr_filter])
    x_fft_from_ekf_est = np.array([0 for _ in time_arr_filter])

    P_ekf_est = np.array(
        [np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr_filter])

    for index, time in enumerate(time_arr_filter):
        z = dz_s_filter_freq[index]

        ekf.predict_update(z)
        x_ekf_est[index] = ekf.x_est
        P_ekf_est[index] = ekf.P_est

        if index >= 1000:
            freq_z, ampl_z = perform_discrete_fft(simulation_params, dz_s_filter_freq[0:index])
            freq_x1_ekf, ampl_x1_ekf = perform_discrete_fft(simulation_params, x_ekf_est[:, 1])
            x_fft_est[index] = abs(2*np.pi*freq_z[np.where(ampl_z == np.amax(ampl_z))][-1])
            x_fft_from_ekf_est[index] = abs(2 * np.pi * freq_x1_ekf[np.where(ampl_x1_ekf == np.amax(ampl_x1_ekf))][-1])
        else:
            x_fft_est[index] = simulation_params.x_0[2]
            x_fft_from_ekf_est[index] = simulation_params.x_0[2]


    # freqs, xs_fft = perform_discrete_fft(simulation_params, xs)
    # plot_mse_sim_ekf_cont(time_arr_simulation, xs, x_ekf_est, simulation_params)
    # plot_xs_sim_ekf_cont(time_arr_simulation, xs, time_arr_filter, x_ekf_est, simulation_params)

    return xs, x_ekf_est, x_fft_est, x_fft_from_ekf_est, dz_s
