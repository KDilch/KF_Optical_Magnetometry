#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from munch import DefaultMunch
from copy import deepcopy
import numpy as np
from multiprocessing import Pool

from utilities.config_util import import_config_from_path
from run_magnetometer import run__magnetometer


def run__magnetometer_statistics(*args):
    # Logger for storing errors and logs in separate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of run-magnetometer-statistics command.')

    logger.info('Loading a config file from path %r' % args[0].config)
    config = import_config_from_path(args[0].config)
    simulation_params = DefaultMunch.fromDict(config.simulation)

    logger.info('Setting simulation parameters to delta_t_simulation = %r, t_max=%r.' %
                (str(simulation_params.dt),
                 str(simulation_params.t_max)
                 )
                )
    num_trajectories = np.intc(args[0].num_reps)

    args_list = [deepcopy(args) for i in range(num_trajectories)]

    # MULTIPROCESSING
    pool = Pool(10)
    with pool:
        pool.starmap(run__magnetometer, args_list)

    # TODO save a copy of config file

