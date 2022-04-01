import os
import json
import logging.config
import logging
from importlib import import_module


def import_config_from_path(module_name):
    filename = os.path.basename(module_name)
    module_object = import_module(str(os.path.splitext(filename)[0]))
    return getattr(module_object, 'config')


def load_logging_config(default_path='logging.json', default_level=logging.INFO):
    """Setup logging configuration
    """
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        info_file_path = os.path.dirname(config['handlers']['info_file_handler']['filename'])
        error_file_path = os.path.dirname(config['handlers']['error_file_handler']['filename'])
        if not os.path.exists(info_file_path):
            os.makedirs(info_file_path)
        if not os.path.exists(error_file_path):
            os.makedirs(error_file_path)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    return
