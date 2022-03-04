#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
from munch import DefaultMunch

from utilities.config_util import import_config_from_path
from dynamics.atomic_sensor_simulation_model import Atomic_Sensor_Simulation_Model
from measurement.atomicsensormeasurementmodel import AtomicSensorMeasurementModel


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
    import matplotlib.pyplot as plt
    plt.style.use('_mpl-gallery')

   # plot
    fig, ax = plt.subplots()

    ax.plot(time_arr_simulation, xs[:, 0], linewidth=2.0)

    plt.show()
    fig, ax = plt.subplots()

    ax.plot(time_arr_simulation, xs[:, 1], linewidth=2.0)

    plt.show()
    fig, ax = plt.subplots()

    ax.plot(time_arr_simulation, xs[:, 2], linewidth=2.0)

    plt.show()
    # x_filter_freq = sensor.state_vec_full_history[::every_nth_z]

    # KALMAN FILTER====================================================
    # unscented_kf_model = Unscented_KF(fx=compute_fx_at_time_t(0),
    #                                   Q=linear_kf_model.Q_delta,
    #                                   hx=hx,
    #                                   R=R / config.filter['dt_filter'],
    #                                   Gamma=state.Gamma_control_evolution_matrix,
    #                                   u=state.u_control_vec,
    #                                   z0=[zs[0]],
    #                                   dt=config.filter['dt_filter'],
    #                                   x0=linear_kf_model.x0,
    #                                   P0=linear_kf_model.P0)

    # extended_kf_model = Extended_KF(Q=Q,
    #                                 H=H,
    #                                 R_delta=R / config.filter['dt_filter'],
    #                                 Gamma=state.Gamma_control_evolution_matrix,
    #                                 u=state.u_control_vec,
    #                                 z0=[zs[0]],
    #                                 dt=config.filter['dt_filter'],
    #                                 num_terms=3,
    #                                 time_arr=time_arr_filter
    #                                 )
    #
    # # RUN FILTERPY KALMAN FILTER
    # logger.info("Initializing linear_kf_filterpy Kalman Filter")
    #
    # # logger.info("Initializing unscented_kf_filterpy Unscented Filter")
    # # unscented_kf_filterpy = unscented_kf_model.initialize_filterpy()
    # # unscented_kf_history_manager = Filter_History_Manager(unscented_kf_filterpy, num_iter_filter)
    #
    # logger.info("Initializing extended_kf_filterpy Unscented Filter")
    # extended_kf_filterpy = extended_kf_model.initialize_filterpy()
    # extended_kf_history_manager = Filter_History_Manager(extended_kf_filterpy, num_iter_filter)
    #
    # for index, time in enumerate(time_arr_filter):
    #     z = zs_filter_freq[index]
    #     # unscented_kf_filterpy.predict(fx=compute_fx_at_time_t(time))
    #     # unscented_kf_filterpy.update(z)
    #     # unscented_kf_history_manager.add_entry(index)
    #
    #     extended_kf_filterpy.predict()
    #     print(H, "H")
    #     extended_kf_filterpy.update(z, HJacobian=lambda x: H, Hx=lambda x: np.dot(H, x))
    #     extended_kf_history_manager.add_entry(index)
    #
    # # PLOTS=========================================================
    # # Get history data from sensor state class and separate into blocks using "zip".
    # x1_full_history, x2_full_history, x3_full_history = zip(*sensor.state_vec_full_history)
    #
    # labels = ['Extended kf', 'Exact data']
    # labels_err = ['Extended kf err', 'Steady state']
    #
    # if np.any([True]):
    #     logger.info("Plotting data x1")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([
    #                                              time_arr_filter,
    #                                              time_arr]),
    #                                    np.array([extended_kf_history_manager.x1s,
    #                                              x1_full_history]),
    #                                    labels=labels,
    #                                    bools=[True,
    #                                          True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="x1", is_show=True, is_legend=True)
    #
    #     # plot error for atoms jy
    #     logger.info("Plotting error x1")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([
    #                                                                           time_arr_filter]),
    #                                                                 np.array([extended_kf_history_manager.x1s_err_post,
    #                                                                           ]),
    #                                                                 labels=labels_err,
    #                                                                 bools=[True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Squared error x1", is_show=True, is_legend=True)
    #
    #     # plot atoms jz
    #     logger.info("Plotting data x2")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([
    #                                                                           time_arr_filter,
    #                                                                           time_arr]),
    #                                                                 np.array([extended_kf_history_manager.x2s,
    #                                                                           x2_full_history]),
    #                                                                 labels=labels,
    #                                                                 bools=[True,
    #                                                                        True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="x2", is_show=True, is_legend=True)
    #
    #     # plot error for atoms jz
    #     logger.info("Plotting error x2")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([
    #                                                                           time_arr_filter]),
    #                                                                 np.array([extended_kf_history_manager.x2s_err_post,
    #                                                                          ]),
    #                                                                 labels=labels_err,
    #                                                                 bools=[True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Squared error x2", is_show=True, is_legend=True)
    #
    #     logger.info("Plotting data x3")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([
    #                                                                           time_arr_filter,
    #                                                                           time_arr]),
    #                                                                 np.array([
    #                                                                           extended_kf_history_manager.x3s,
    #                                                                           x3_full_history]),
    #                                                                 labels=labels,
    #                                                                 bools=[True,
    #                                                                        True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="x3", is_show=True, is_legend=True)
    #
    #     # plot error for light q
    #     logger.info("Plotting error q")
    #     xs_sel, ys_sel, labels_sel = generate_data_arr_for_plotting(np.array([
    #
    #                                                                           time_arr_filter]),
    #                                                                 np.array([
    #                                                                           extended_kf_history_manager.x3s_err_post,
    #                                                                          ]),
    #                                                                 labels=labels_err,
    #                                                                 bools=[True])
    #     plot_data(xs_sel, ys_sel, data_labels=labels_sel, title="Squared error x3", is_show=True, is_legend=True)
