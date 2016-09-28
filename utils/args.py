#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Imports
import os.path
import argparse

# ==============================================================================
from utils.log import createAndInitLogger
logger = createAndInitLogger(__name__)

# ==============================================================================
class StoreValidFilePath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.isfile(values):
            parser.error("'%s' does not exist or is not a file." % values)
        setattr(namespace, self.dest, values)

class StoreValidFilePaths(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for filename in values:
            if not os.path.isfile(filename):
                parser.error("'%s' does not exist or is not a file." % filename)
        setattr(namespace, self.dest, values)

class StoreValidFilePathOrStdin(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not values == '-' and not os.path.isfile(values):
            parser.error("'%s' does not exist or is not a file." % values)
        setattr(namespace, self.dest, values)

class StoreValidDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.isdir(values):
            parser.error("'%s' does not exist or is not a directory." % values)
        setattr(namespace, self.dest, values)

class StoreExistingOrCreatableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) is not str:
            parser.error("'%s' does not represent a directory path." % values)
        if os.path.exists(values) and not os.path.isdir(values):
            parser.error("'%s' is not a directory." % values)
        if os.path.exists(values) and not os.access(values, os.W_OK):
            parser.error("'%s' is not writable." % values)
        if not os.path.exists(values):
            try:
                os.makedirs(values)
            except OSError as e: 
                parser.error("'%s' cannot be created." % values)
        setattr(namespace, self.dest, values)


class StoreWidthHeight(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # try to split on 'x' and get two parts
        ERR_FORMAT_STR = "'%s' does not represent a shape with WWWxHHH (width + 'x' + height) format."
        if not 'x' in  values:
            parser.error(ERR_FORMAT_STR % values)
        parts = values.split('x')
        if len(parts) != 2:
            parser.error(ERR_FORMAT_STR % values)
        # convert to int and store as tuple (width, height)
        intparts = []
        for p in parts:
            try:
                intparts.append(int(p))
            except Exception, e:
                parser.error("'%s' does not represent a pixel value." % p)
                # raise called
        setattr(namespace, self.dest, (intparts[0], intparts[1]))


class StoreIntZeroPositive(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        intval = None
        try:
            intval = int(values)
        except:
            parser.error("'%s' cannot be coerced to an int value." % values)
            
        if intval < 0:
            parser.error("'%s' does not represent an int >= 0." % values)
        setattr(namespace, self.dest, intval)


class Store0to1float(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        floatval = None
        try:
            floatval = float(values)
        except:
            parser.error("'%s' cannot be coerced to a float value." % values)
            
        if floatval < 0 or floatval > 1.0:
            parser.error("'%s' does not represent a float within [0.0; 1.0]." % values)
        setattr(namespace, self.dest, floatval)


def dumpArgs(args, logger=logger):
    logger.debug("Arguments:")
    for (k, v) in args.__dict__.items():
        logger.debug("    %-20s = %s" % (k, v))
