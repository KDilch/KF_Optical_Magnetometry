#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

from utilities.namespace_util import stringify_namespace
from utilities.config_util import load_logging_config
from utilities.argparse_parsers import initialize_parsers

def main():
    # setup a logger
    load_logging_config()
    logger = logging.getLogger(__name__)
    logger.info('Starting execution of the simulation.')
    parser = initialize_parsers()

    # parse some argument lists
    args = parser.parse_args()
    args.func(args)
    logger.info('Parsed input arguments %r' % stringify_namespace(args))
    return 0


if __name__ == "__main__":
    main()
