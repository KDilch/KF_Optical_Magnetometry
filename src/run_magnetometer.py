#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
from munch import DefaultMunch

from utilities.config_util import import_config_from_path
from dynamics.atomic_sensor_simulation_model import Atomic_Sensor_Simulation_Model
from measurement.atomicsensormeasurementmodel import AtomicSensorMeasurementModel
from kalman_filter.continuous.magnetometer_ekf import MagnetometerEKF


def run__magnetometer(*args):

    # Logger for storing errors and logs in separate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-frequency-extractor command.')

    logger.info('Loading a config file from path %r' % args[0].config)
    config = import_config_from_path(args[0].config)
    simulation_params = DefaultMunch.fromDict(config.simulation)

    logger.info('Setting simulation parameters to delta_t_simulation = %r, t_max=%r.' %
                (str(simulation_params.dt),
                 str(simulation_params.t_max)
                 )
                )
    filter_params = DefaultMunch.fromDict(config.filter)
    logger.info('Setting filter parameters to delta_t_filter = %r.' %
                (str(filter_params.dt)
                 )
                )

    logger.info('Setting initial state vec to  [%r].' %
                (str(config.simulation['x_0'])))

    logger.info('Setting filter Q, H and R to Q = %r, H = %r, R = %r' %
                (str(filter_params.noise.Q),
                 str(filter_params.measurement.H),
                 str(filter_params.measurement.R)
                 )
                )

    # continuous measurement for now
    num_iter_simulation = np.intc(np.floor_divide(simulation_params.t_max,
                                                  simulation_params.dt))
    num_iter_filter = np.intc(np.floor_divide(simulation_params.t_max,
                                              filter_params.dt))
    logger.info('Number of filter iterations and number of simulation iterations is respectively  [%r, %r].' %
                (str(num_iter_filter), str(num_iter_simulation)))

    every_nth_z = np.intc(np.floor_divide(num_iter_simulation, num_iter_filter))

    time_arr_simulation = np.arange(0, simulation_params.t_max, simulation_params.dt)
    time_arr_filter = np.arange(0, simulation_params.t_max, filter_params.dt)

    # SIMULATE THE DYNAMICS=====================================================
    simulation_dynamical_model = Atomic_Sensor_Simulation_Model(t=0,
                                                                simulation_params=simulation_params)

    xs = np.array([np.array((simulation_dynamical_model.step())) for _ in time_arr_simulation])

    measurement_model = AtomicSensorMeasurementModel(simulation_params)

    zs = np.array([np.array((measurement_model.read_sensor(_))) for _ in xs])  # noisy measurement
    zs_filter_freq = zs[::every_nth_z]

    # KALMAN FILTER====================================================
    logger.info("Initializing ekf")
    ekf = MagnetometerEKF(model_params=filter_params)
    x_ekf_est = np.array([np.zeros_like(filter_params.x_0) for _ in time_arr_filter])

    P_ekf_est = np.array([np.zeros((len(filter_params.x_0), len(filter_params.x_0))) for _ in time_arr_filter])

    for index, time in enumerate(time_arr_filter):
        z = zs_filter_freq[index]

        ekf.predict_update(z)
        x_ekf_est[index] = ekf.x_est
        P_ekf_est[index] = ekf.P_est

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(time_arr_simulation, xs[:, 1], time_arr_filter, x_ekf_est[:, 1])
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('sim and ekf')
    axs[0].grid(True)
    plt.legend()

    axs[1].plot(time_arr_simulation, xs[:, 2], time_arr_filter, x_ekf_est[:, 2])
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('s1 and s2')
    axs[1].grid(True)

    axs[2].plot(time_arr_simulation, xs[:, 0], time_arr_filter, x_ekf_est[:, 0])
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('s1 and s2')
    axs[2].grid(True)
    plt.show()
