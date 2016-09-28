#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Imports
import logging
import argparse
import os.path
import sys
import io
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats


# ==============================================================================
# SegEval Tools suite imports
from utils.args import *
from utils.log import *

# ==============================================================================
logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
PROG_VERSION = "0.1"
PROG_NAME = "SmartDoc Results Analyzer"
PROG_NAME_SHORT = "smartdoc_ji"

ERRCODE_OK = 0
ERRCODE_NOFILE = 10


def barPlt(arr, sems, title, xlab, ylab):
    ind = np.arange(len(arr)) # create the x-axis
    fig = plt.figure()
    ax = plt.subplot(111) # we create a variable for the subplot so we can access the axes
    # set the top and right axes to invisible
    # matplotlib calls the axis lines spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # turn off ticks where there is no spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    bar_plt = plt.bar(ind, arr, color = 'grey', align = 'center', width = 0.5) # width argument slims down the bars
    plt.hold(True)
    plt.errorbar(ind, arr, sems, elinewidth = 2, ecolor = 'black', fmt = None, capsize=7.5) # add the errorbars
    plt.ylabel(ylab, fontsize = 18)
    plt.xticks(ind, xlab, fontsize = 18)
    plt.title(title, fontsize = 24)
    plt.show(bar_plt)


# ==============================================================================
def main(argv=None):
    # Option parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=PROG_NAME, 
        version=PROG_VERSION)

    parser.add_argument('-d', '--debug', 
        action="store_true", 
        help="Activate debug output.")

    parser.add_argument('input_file', 
        action=StoreValidFilePathOrStdin,
        help='CSV file containing Jaccard Index measures for each frame (or - for stdin).')

    parser.add_argument('output_dir', 
        action=StoreExistingOrCreatableDir,
        help='Place where results should be stored.')

    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # Logger activation
    initLogger(logger)
    output_prettyprint = False
    if args.debug:
        logger.setLevel(logging.DEBUG)
        output_prettyprint = True
    
    # -----------------------------------------------------------------------------
    # Output log header
    programHeader(logger, PROG_NAME, PROG_VERSION)
    logger.debug(DBGSEP)
    dumpArgs(args, logger)
    logger.debug(DBGSEP)

    # -----------------------------------------------------------------------------
    logger.debug("Starting up")
    
    # --------------------------------------------------------------------------
    logger.debug("--- Process started. ---")

    data = pd.read_csv(args.input_file, 
                       delim_whitespace=True,
                       names=["method","background","document", "frame", "ji"])


    table = pd.pivot_table(data, values=['ji'], index=['background'], columns=['method'], aggfunc=np.average, margins=True)
    table.plot(kind='bar')
    print table;

    table2 = pd.pivot_table(data, values=['ji'], index=['background'], columns=[], aggfunc=np.average, margins=True)
    table2.plot(kind='bar')
    print table2;

    data[['docclass']] = data[['document']].applymap(lambda d: d.split("0")[0])
    table3 = pd.pivot_table(data, values=['ji'], index=['docclass'], columns=['method'], aggfunc=np.average, margins=True)
    table3.plot(kind='bar')
    print table3;

    table4 = pd.pivot_table(data, values=['ji'], index=['docclass'], columns=[], aggfunc=np.average, margins=True)
    table4.plot(kind='bar')
    print table4;

    data.boxplot(column='ji', by=u'method', fontsize=9, vert=False)
    print "Data:"
    print data[['method', 'ji']].groupby('method').describe()

    plt.figure()

    methods = data['method'].unique()
    
    results = {}

    for m in map(str, methods):
        values = data[data['method'] == m]['ji']
        n, min_max, mean, var, skew, kurt = stats.describe(values)
        std = math.sqrt(var)
        R = stats.norm.interval(0.95,loc=mean,scale=std/math.sqrt(len(values))) 
        results[m] = (mean, R)

    print "CI:"
    to_plot = []
    for (m, (mean, (cil,cih))) in results.items():
        print  m, '\t', mean, '\t', cil, '\t', cih
        to_plot.append([m, mean, cil, cih])

    fig = plt.gca()
    mean_values, ci = zip(*[r for r in results.itervalues()])
    bar_labels = [str(m) for m in results.iterkeys()]

    ci2 = [(cih - m) for (m, (cil, cih)) in results.itervalues()]

    # plot bars
    x_pos = list(range(len(to_plot)))
    # plt.bar(x_pos, mean_values, yerr=ci, align='center', alpha=0.5)
    # plt.bar(x_pos, mean_values, yerr=ci2, align='center', alpha=0.5)
    plt.barh(x_pos, mean_values, xerr=ci2, align='center', alpha=0.5)

    # set height of the y-axis
    plt.xlim([0,1])

    # set axes labels and title
    plt.xlabel('Jaccard Index')
    plt.yticks(x_pos, bar_labels)
    plt.title('Overvall Evaluation')

    # axis formatting
    fig.axes.get_yaxis().set_visible(True)
    fig.spines["top"].set_visible(False)  
    fig.spines["right"].set_visible(False)  
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                    labelbottom="on", left="on", right="off", labelleft="on")  

    plt.show()
    plt.savefig(os.path.join(args.output_dir, "meth_vs_perf.pdf"))





    logger.debug("--- Process complete. ---")
    # --------------------------------------------------------------------------

    logger.debug("Clean exit.")
    logger.debug(DBGSEP)
    return ERRCODE_OK
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())


