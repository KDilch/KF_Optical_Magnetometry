import argparse

from run_tests import run__test
from run_magnetometer import run__magnetometer
from run_magnetometer_statistics import run__magnetometer_statistics


def initialize_parsers():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='Atomic Magnetometer Simulation')
    # parser.add_argument('--working_dir', action='store', help='foo help')
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "run-tests" command
    tests_parser = subparsers.add_parser('run-tests', help='Run unit tests')
    tests_parser.set_defaults(func=run__test)

    # create the parser for the "run-magnetometer" command
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

    simulation_parser.add_argument('--save_plots',
                                   action='store_true',
                                   help='Bool specifying if you want to save plots',
                                   default=False)
    simulation_parser.add_argument('--save_data_file',
                                   action='store_true',
                                   help='Bool specifying if you want to save the data in a file',
                                   default=False)
    simulation_parser.set_defaults(func=run__magnetometer)

    simulation_parser_stat = subparsers.add_parser('run-magnetometer-statistics', help='Run atomic sensor simulation')
    simulation_parser_stat.add_argument('-o',
                                   '--output_path',
                                   action='store',
                                   help='A string representing path where the output should be saved.',
                                   default='./')
    simulation_parser_stat.add_argument('--config',
                                   action='store',
                                   help='A string representing a module name of a config file. Config is a python file.',
                                   default='config')

    simulation_parser_stat.add_argument('--save_plots',
                                   action='store_true',
                                   help='Bool specifying if you want to save plots',
                                   default=False)
    simulation_parser_stat.add_argument('--save_data_file',
                                   action='store_true',
                                   help='Bool specifying if you want to save the data in a file',
                                   default=False)
    simulation_parser_stat.set_defaults(func=run__magnetometer_statistics)

    return parser
