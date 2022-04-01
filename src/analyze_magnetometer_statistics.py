#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import glob
import dask.dataframe as dd
from utilities.save_data import prepare_avg_ddf, save_data_avg_simple_simulation
from plots import plot_simple_model_avg


def analyze__magnetometer_statistics(*args):
    # Logger for storing errors and logs in separate file, creates separate folder
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of analyze-magnetometer-statistics command.')
    data_path = os.path.join(args[0].data_path, '*.csv')
    number_reps = len(glob.glob(data_path))
    logger.info('Found %r data files.' % number_reps)
    ddf = dd.read_csv(data_path)
    avgs_ddf = prepare_avg_ddf(ddf)
    if args[0].save_data:
        save_data_avg_simple_simulation(avgs_ddf, args[0].output_path, number_reps)
    if args[0].save_plots:
        plot_simple_model_avg(avgs_ddf, dir_name=args[0].output_path, num_reps=number_reps)

