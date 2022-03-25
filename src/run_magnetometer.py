#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
from copy import deepcopy
from munch import DefaultMunch

from utilities.config_util import import_config_from_path
from space_state_model.atomic_sensor_simulation_model import Atomic_Sensor_Simulation_Model
from measurement.atomicsensormeasurementmodel import AtomicSensorMeasurementModel
from kalman_filter.continuous.magnetometer_ekf import MagnetometerEKF
from plots import plot_mse_sim_ekf_cont, plot_xs_sim_ekf_cont, plot_est_cov, plot_fx_diff


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
    simulation_dynamical_model = Atomic_Sensor_Simulation_Model(t=0,
                                                                simulation_params=simulation_params)
    measurement_model = AtomicSensorMeasurementModel(simulation_params)
    ekf = MagnetometerEKF(model_params=filter_params_ekf)

    # ALLOCATE MEMORY FOR THE ARRAYS=====================================================
    xs = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr])
    dz_s = np.array([np.zeros_like(simulation_params.measurement.noise.mean) for _ in time_arr])
    x_ekf_est = np.array([np.zeros_like(filter_params_ekf.x_0) for _ in time_arr])
    P_ekf_est = np.array(
        [np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr])

    # RUN THE SIMULATION, PERFORM THE MEASUREMENT AND FILTER
    for index, time in enumerate(time_arr):
        xs[index] = simulation_dynamical_model.step()
        dz_s[index] = measurement_model.read_sensor(xs[index])
        ekf.predict_update(dz_s[index])
        x_ekf_est[index] = ekf.x_est
        P_ekf_est[index] = ekf.P_est

    return xs, x_ekf_est, dz_s, P_ekf_est

