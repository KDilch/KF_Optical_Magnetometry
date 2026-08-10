import os
import json
import logging.config
import logging
import importlib.util
import os
from pathlib import Path
from importlib import import_module


def import_config_from_path(module_path):
    # Convert to absolute path
    module_path = Path(module_path).resolve()

    # Extract module name from filename (e.g. config_file.py -> config_file)
    module_name = module_path.stem

    # Load the module from file
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Return the 'config' attribute (if it exists)
    return getattr(module, 'config')


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
