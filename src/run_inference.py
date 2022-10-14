#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import glob
import dask.dataframe as dd
from utilities.save_data import prepare_avg_ddf, save_data_avg_simple_simulation
from plots import plot_simple_model_avg


def run__magnetometer_inference(*args):
    # Logger for storing errors and logs in separate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of analyze-magnetometer-statistics command.')
    data_path = os.path.join(args[0].data_path, '*.csv')
    ddf = dd.read_csv(data_path)
    if args[0].cc_ekf:
        pass
    if args[0].cd_ekf:
        pass
    if args[0].save_data:
        pass
    if args[0].save_plots:
        pass

