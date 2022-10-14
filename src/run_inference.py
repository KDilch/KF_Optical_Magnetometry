#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import glob
import dask.dataframe as dd
from munch import DefaultMunch
from copy import deepcopy
from utilities.config_util import import_config_from_path


def run__magnetometer_inference(*args):
    # Logger for storing errors and logs in separate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of analyze-magnetometer-statistics command.')
    data_path = os.path.join(args[0].data_path, '*.csv')
    ddf = dd.read_csv(data_path)
    config = import_config_from_path(args[0].config)
    filter_params_ekf = DefaultMunch.fromDict(deepcopy(config.filter_ekf))
    if args[0].cc_ekf:
        pass
    if args[0].cd_ekf:
        pass
    if args[0].save_data:
        pass
    if args[0].save_plots:
        pass

