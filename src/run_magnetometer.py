#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
from scipy.fft import fft, fftfreq
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
    filter_params_ekf = DefaultMunch.fromDict(config.filter_ekf)
    logger.info('Setting filter parameters to delta_t_filter = %r.' %
                (str(filter_params_ekf.dt)
                 )
                )

    logger.info('Setting initial state vec to  [%r].' %
                (str(config.simulation['x_0'])))

    logger.info('Setting filter Q, H and R to Q = %r, H = %r, R = %r' %
                (str(filter_params_ekf.noise.Q),
                 str(filter_params_ekf.measurement.H),
                 str(filter_params_ekf.measurement.R)
                 )
                )

    # continuous measurement for now
    num_iter_simulation = np.intc(np.floor_divide(simulation_params.t_max,
                                                  simulation_params.dt))
    num_iter_filter = np.intc(np.floor_divide(simulation_params.t_max,
                                              filter_params_ekf.dt))
    logger.info('Number of filter iterations and number of simulation iterations is respectively  [%r, %r].' %
                (str(num_iter_filter), str(num_iter_simulation)))

    every_nth_z = np.intc(np.floor_divide(num_iter_simulation, num_iter_filter))

    time_arr_simulation = np.arange(0, simulation_params.t_max, simulation_params.dt)
    time_arr_filter = np.arange(0, simulation_params.t_max, filter_params_ekf.dt)

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

    P_ekf_est = np.array(
        [np.zeros((len(filter_params_ekf.x_0), len(filter_params_ekf.x_0))) for _ in time_arr_filter])

    for index, time in enumerate(time_arr_filter):
        z = dz_s_filter_freq[index]

        ekf.predict_update(z)
        x_ekf_est[index] = ekf.x_est
        P_ekf_est[index] = ekf.P_est

    #TODO MOVE TH FFT TO A DIFFERENT FILE
    SAMPLE_RATE = 1/simulation_params.dt
    NUM_SAMPLES = int(simulation_params.t_max*SAMPLE_RATE)
    # x_fft = np.abs(fft(x_ekf_est[:, 1]))
    x_fft = np.abs(fft(xs[:, 1]))

    # get the list of frequencies
    frequencies = fftfreq(NUM_SAMPLES, simulation_params.dt)
    logger.info("Fourier trnasporm frequencies are %s" % str(2*np.pi*frequencies[np.where(x_fft == np.amax(x_fft))]))

    import matplotlib.pyplot as plt

    # fig = plt.figure(2, figsize=(15, 6))
    # plt.clf()
    # plt.plot(frequencies, np.abs(x_fft), lw=1.0, c='paleturquoise')
    # plt.xlabel("frequency [Hz]")
    # plt.ylabel("amplitude [a.u.]")
    # plt.xlim(-10, 10)
    # plt.title(r"$|\mathcal{F}(A_{signal})|$")
    # plt.show()

    # fig, axs = plt.subplots(3, 1)
    # axs[0].plot(time_arr_simulation, xs[:, 1], time_arr_filter, x_ekf_est[:, 1])
    # axs[0].set_xlabel('time')
    # axs[0].set_ylabel('Jz')
    # axs[0].grid(True)
    #
    # axs[1].plot(time_arr_simulation, xs[:, 2], time_arr_filter, x_ekf_est[:, 2])
    # axs[1].set_xlabel('time')
    # # axs[1].axhline(y=abs(2*np.pi*frequencies[np.where(x_fft == np.amax(x_fft))][-1]), color='r', linestyle='-')
    # axs[1].set_ylabel('larmour')
    # axs[1].grid(True)
    #
    # axs[2].plot(time_arr_simulation, xs[:, 0], time_arr_filter, x_ekf_est[:, 0])
    # axs[2].set_xlabel('time')
    # axs[2].set_ylabel('Jx')
    # axs[2].grid(True)
    # plt.show()
