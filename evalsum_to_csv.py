#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Imports
import logging
import argparse
import os
import os.path
import sys
import fileinput
import itertools # chain
import csv

from dexml import ParseError

# ==============================================================================
# SegEval Tools suite imports
from utils.args import *
from utils.log import *
from models.models import *

# ==============================================================================
logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
PROG_VERSION = "0.1"
PROG_NAME = "Evaluation result and summary to CSV extractor"
PROG_NAME_SHORT = "Evalsum2CSV"
XML_VERSION_MIN = 0.2
XML_VERSION_MAX = 0.3

E_OK = 0
E_NOFILE = 10


# ==============================================================================
def read_results_from_file(eval_file):
    current_mdl = None
    try:
        try:
            current_mdl = EvalResult.loadFromFile(eval_file)
            logger.debug("Got EvalResult file.")
        except dexml.ParseError:
            current_mdl = EvalSummary.loadFromFile(eval_file)
            logger.debug("Got EvalSummary file.")
    except Exception, e:
        logger.error("File '%s' is not a valid segmentation evaluation file." % eval_file)
        logger.error("\t Is it a '*.segeval.xml' or a '*.evalsummary.xml' file?")
        raise e
    return current_mdl.global_results


# ==============================================================================
def main(argv=None):
    # Option parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Extract segmentation evaluation results or summaries and generate a CSV file.', 
        version=PROG_VERSION)


    parser.add_argument('-d', '--debug', 
        action="store_true", 
        help="Activate debug output.")

    parser.add_argument('-f', '--files-from', metavar="FILE_LIST", 
        action=StoreValidFilePathOrStdin,
        help="File containing the list of files to merge, or '-' to use standard input. \
              Will be read BEFORE files specified on command line.")

    parser.add_argument('files', 
        action=StoreValidFilePaths,
        metavar='result_file', 
        nargs='*',
        help='EvalSummary or SegEval files containing global results to merge.')

    parser.add_argument('-o', '--output-file', required=True,
        help="MANDATORY path to output file.")

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
    # Create file name generator
    file_iter = None
    files_in_list = []
    if args.files_from:
        files_in_list = (line.rstrip("\n") for line in fileinput.input([args.files_from]))
    file_iter = itertools.chain(files_in_list, args.files)

    # Prepare output file
    with open(args.output_file, "wb") as ofile:
        csv_writer = csv.writer(ofile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        # Output header
        header = [
            "filename",                      # string
            "mean_segmentation_precision",   # float
            "mean_segmentation_recall",      # float
            "detection_precision",           # float
            "detection_recall",              # float
            "mean_jaccard_index_smartdoc",   # float
            "mean_jaccard_index_segonly",    # float
            "count_total_frames",            # int
            "count_true_accepted_frames",    # int
            "count_true_rejected_frames",    # int
            "count_false_accepted_frames",   # int
            "count_false_rejected_frames"]   # int
        logger.info("\t".join(header))
        csv_writer.writerow(header)

    # --------------------------------------------------------------------------
        logger.debug("--- Process started. ---")
        # Loop over files
        file_count = 0
        for eval_file in file_iter:
            logger.debug("Processing file '%s'" % eval_file)
            # Try to read either EvalResult or EvalSummary
            res_cur = read_results_from_file(eval_file)
            # Payload
            res_lst = [
                eval_file,
                res_cur.mean_segmentation_precision,
                res_cur.mean_segmentation_recall,
                res_cur.detection_precision,
                res_cur.detection_recall,
                res_cur.mean_jaccard_index_smartdoc,
                res_cur.mean_jaccard_index_segonly,
                res_cur.count_total_frames,
                res_cur.count_true_accepted_frames,
                res_cur.count_true_rejected_frames,
                res_cur.count_false_accepted_frames,
                res_cur.count_false_rejected_frames]
            # Output (log and file)
            logger.info("\t".join(map(str, res_lst)))
            csv_writer.writerow(res_lst)
            # Stats
            file_count += 1

        logger.debug("--- Process complete. ---")
    # --------------------------------------------------------------------------

    # Test for empty task and trap
    if file_count == 0:
        logger.error("No file processed.")
        logger.error("\t Use '-h' option to review program synopsis.")
        return E_NOFILE

    # else
    logger.debug("%d files processed." % file_count)
    logger.debug("Clean exit.")
    logger.debug(DBGSEP)
    return E_OK
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())

