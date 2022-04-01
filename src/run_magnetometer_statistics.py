#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tqdm.auto import tqdm as std_tqdm
import logging
from datetime import datetime
import shutil
import os
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
    shutil.copyfile(args[0].config, os.path.join(args[0].output_path, 'config_%s' % datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')))
    config = import_config_from_path(args[0].config)
    simulation_params = DefaultMunch.fromDict(config.simulation)

    logger.info('Setting simulation parameters to delta_t_simulation = %r, t_max=%r.' %
                (str(simulation_params.dt),
                 str(simulation_params.t_max)
                 )
                )
    num_reps = np.intc(args[0].num_reps)

    args_list = [deepcopy(args) for _ in range(num_reps)]

    # MULTIPROCESSING
    pool = Pool(10)
    with pool:
        pool.starmap(run__magnetometer, args_list)
