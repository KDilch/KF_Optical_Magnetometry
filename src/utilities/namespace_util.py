#!/usr/bin/env python
# -*- coding: utf-8 -*-

def stringify_namespace(namespace):
    """
    :param namespace:
    :return: string
    """
    __str = ''
    for arg in namespace.__dict__:
        if arg:
            __str += " --" + arg + ": " + str(namespace.__dict__[arg])
    return __str

