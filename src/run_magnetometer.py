#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
from copy import deepcopy
from munch import DefaultMunch

from utilities.config_util import import_config_from_path
from space_state_model.simple_sensor_model import Simple_CC_Sensor_Model
from kalman_filter.continuous.simple_model_ekf import MagnetometerEKF
from plots import plot_simple_model
from utilities.save_data import save_data_simple_simulation, prepare_df


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

    # continuous space_state_model and measurement for now
    # CREATE A TIME ARRAY====================================================
    num_iter_simulation = np.intc(np.floor_divide(simulation_params.t_max,
                                                  simulation_params.dt))

    time_arr = np.arange(0, simulation_params.t_max, simulation_params.dt)
    # INITIALIZE THE MODEL=====================================================
    simulation_dynamical_model = Simple_CC_Sensor_Model(t=0,
                                                        simulation_params=simulation_params)
    ekf = MagnetometerEKF(model_params=filter_params_ekf)

    # ALLOCATE MEMORY FOR THE ARRAYS=====================================================
    xs = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr])
    z_s = np.array([np.zeros_like(simulation_params.measurement.noise.mean) for _ in time_arr])
    x_ekf_est = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr])
    P_ekf_est = np.array([np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr])

    # RUN THE SIMULATION, PERFORM THE MEASUREMENT AND FILTER
    for index, time in enumerate(time_arr):
        # SIMULATION AND MEASUREMENT==============================
        xs[index], z_s[index] = simulation_dynamical_model.step(method=args[0].method)
        # KALMAN FILTER===========================================
        if args[0].ekf:
            ekf.predict_update(z_s[index])
            x_ekf_est[index] = ekf.x_est
            P_ekf_est[index] = ekf.P_est
    if args[0].ekf:
        df = prepare_df(time_arr, xs, xs_est=x_ekf_est, P_est=P_ekf_est)
    else:
        df = prepare_df(time_arr, xs)

    if args[0].save_data:
        save_data_simple_simulation(df, simulation_params, args[0].output_path+'/data/csv')
    if args[0].save_plots:
        plot_simple_model(df,
                          dir_name=args[0].output_path+'/data/plots',
                          params=simulation_params,
                          simulation=True,
                          ekf=args[0].ekf,
                          err=args[0].ekf,
                          err_loglog=args[0].ekf,
                          show=False,
                          save=True)
    return df
