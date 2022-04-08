import argparse

from run_tests import run__test
from run_magnetometer import run__magnetometer
from run_magnetometer_statistics import run__magnetometer_statistics
from analyze_magnetometer_statistics import analyze__magnetometer_statistics
from run_magnetometer_corr import run__magnetometer_corr
from run_10x10 import run__10x10_corr


def initialize_parsers():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='Atomic Magnetometer Simulation')
    # parser.add_argument('--working_dir', action='store', help='foo help')
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "run-tests" command
    tests_parser = subparsers.add_parser('run-tests', help='Run unit tests')
    tests_parser.set_defaults(func=run__test)

    # RUN MAGNETOMETER=========================================================================================
    simulation_parser = subparsers.add_parser('run-magnetometer', help='Run atomic sensor simulation')
    simulation_parser.add_argument('-o',
                                   '--output_path',
                                   action='store',
                                   help='A string representing path where the output should be saved.',
                                   default='./')
    simulation_parser.add_argument('--config',
                                   action='store',
                                   help='A string representing a module name of a config file. Config is a python file.',
                                   default='config')

    simulation_parser.add_argument('--ekf',
                                   action='store_true',
                                   help='Bool specifying if you want to save plots',
                                   default=False)
    simulation_parser.add_argument('--save_data',
                                   action='store_true',
                                   help='Bool specifying if you want to save the data in a file',
                                   default=False)
    simulation_parser.add_argument('--save_plots',
                                   action='store_true',
                                   help='Bool specifying if you want to save the data in a file',
                                   default=False)
    simulation_parser.add_argument('--method',
                                   action='store',
                                   help='A string representing a method used to solve SDEs.',
                                   default='default')
    simulation_parser.set_defaults(func=run__magnetometer)
    # RUN STATISTICS=========================================================================================
    simulation_parser_stat = subparsers.add_parser('run-magnetometer-statistics',
                                                   help='Run atomic sensor simulation with reps')
    simulation_parser_stat.add_argument('-o',
                                        '--output_path',
                                        action='store',
                                        help='A string representing path where the output should be saved.',
                                        default='./')
    simulation_parser_stat.add_argument('--simulation_type',
                                        action='store',
                                        help='A string representing a simulation type - simple/corr.',
                                        default="simple")
    simulation_parser_stat.add_argument('--config',
                                        action='store',
                                        help='A string representing a module name of a config file. Config is a python file.',
                                        default="C:/Users\Klaudia\Documents\Python_projects\kalman_filters_in_magnetometry\src\config.py")

    simulation_parser_stat.add_argument('--ekf',
                                        action='store_true',
                                        help='Bool specifying if you want to save plots',
                                        default=False)

    simulation_parser_stat.add_argument('--num_reps',
                                        action='store',
                                        help='A string representing a number of repetitions.',
                                        default='config')

    simulation_parser_stat.add_argument('--save_plots',
                                        action='store_true',
                                        help='Bool specifying if you want to save plots',
                                        default=False)
    simulation_parser_stat.add_argument('--save_data',
                                        action='store_true',
                                        help='Bool specifying if you want to save the data in a file',
                                        default=False)
    simulation_parser_stat.add_argument('--method',
                                        action='store',
                                        help='A string representing a method used to solve SDEs.',
                                        default='default')
    simulation_parser_stat.set_defaults(func=run__magnetometer_statistics)
    # RUN SIMPLE MODEL ANALYSIS=========================================================================================
    simulation_parser_stat = subparsers.add_parser('run-magnetometer-analysis', help='Run atomic sensor data analysis')
    simulation_parser_stat.add_argument('-o',
                                        '--output_path',
                                        action='store',
                                        help='A string representing path where the output should be saved.',
                                        default='./')
    simulation_parser_stat.add_argument('--data_path',
                                        action='store',
                                        help='A string representing a module name of a config file. Config is a python file.',
                                        default='config')
    simulation_parser_stat.add_argument('--ekf',
                                        action='store_true',
                                        help='Bool specifying if you want to save plots',
                                        default=False)
    simulation_parser_stat.add_argument('--save_plots',
                                        action='store_true',
                                        help='Bool specifying if you want to save plots',
                                        default=False)
    simulation_parser_stat.add_argument('--save_data',
                                        action='store_true',
                                        help='Bool specifying if you want to save the data in a file',
                                        default=False)
    simulation_parser_stat.set_defaults(func=analyze__magnetometer_statistics)
    # RUN SIMPLE MODEL CORRELATED SIMULATION================================================================
    simulation_parser_corr = subparsers.add_parser('run-magnetometer-corr',
                                                   help='Run atomic sensor simple model correlated simulation')
    simulation_parser_corr.add_argument('-o',
                                        '--output_path',
                                        action='store',
                                        help='A string representing path where the output should be saved.',
                                        default='./')
    simulation_parser_corr.add_argument('--config',
                                        action='store',
                                        help='A string representing a module name of a config file. Config is a python file.',
                                        default='config')

    simulation_parser_corr.add_argument('--ekf',
                                        action='store_true',
                                        help='Bool specifying if you want to save plots',
                                        default=False)
    simulation_parser_corr.add_argument('--save_data',
                                        action='store_true',
                                        help='Bool specifying if you want to save the data in a file',
                                        default=False)
    simulation_parser_corr.add_argument('--save_plots',
                                        action='store_true',
                                        help='Bool specifying if you want to save the data in a file',
                                        default=False)
    simulation_parser_corr.add_argument('--method',
                                        action='store',
                                        help='A string representing a method used to solve SDEs.',
                                        default='default')
    simulation_parser_corr.set_defaults(func=run__magnetometer_corr)

    # RUN 10x10 MODEL CORRELATED SIMULATION================================================================
    simulation_parser_10x10corr = subparsers.add_parser('run-magnetometer10x10-corr',
                                                   help='Run atomic sensor simple model correlated simulation')
    simulation_parser_10x10corr.add_argument('-o',
                                        '--output_path',
                                        action='store',
                                        help='A string representing path where the output should be saved.',
                                        default='./')
    simulation_parser_10x10corr.add_argument('--config',
                                        action='store',
                                        help='A string representing a module name of a config file. Config is a python file.',
                                        default='config')
    simulation_parser_10x10corr.add_argument('--ekf',
                                        action='store_true',
                                        help='Bool specifying if you want to save plots',
                                        default=False)
    simulation_parser_10x10corr.add_argument('--save_data',
                                        action='store_true',
                                        help='Bool specifying if you want to save the data in a file',
                                        default=False)
    simulation_parser_10x10corr.add_argument('--save_plots',
                                        action='store_true',
                                        help='Bool specifying if you want to save the data in a file',
                                        default=False)
    simulation_parser_10x10corr.add_argument('--method',
                                        action='store',
                                        help='A string representing a method used to solve SDEs.',
                                        default='default')
    simulation_parser_10x10corr.set_defaults(func=run__10x10_corr)

    return parser
