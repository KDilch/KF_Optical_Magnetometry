import numpy as np


def initialize_time_arrays(simulation_params, filter_params_ekf):
    num_iter_simulation = np.intc(np.floor_divide(simulation_params.t_max,
                                                  simulation_params.dt))
    num_iter_filter = np.intc(np.floor_divide(simulation_params.t_max,
                                              filter_params_ekf.dt))

    every_nth_z = np.intc(np.floor_divide(num_iter_simulation, num_iter_filter))

    time_arr_simulation = np.arange(0, simulation_params.t_max, simulation_params.dt)
    time_arr_filter = np.arange(0, simulation_params.t_max, filter_params_ekf.dt)
    return time_arr_simulation, time_arr_filter, every_nth_z
