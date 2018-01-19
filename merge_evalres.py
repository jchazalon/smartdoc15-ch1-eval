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
from collections import namedtuple

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
PROG_VERSION = "0.4"
PROG_NAME = "Segmentation Evaluation Result Merger"
PROG_NAME_SHORT = "SegEval"
XML_VERSION_MIN = 0.3
XML_VERSION_MAX = 0.3

ERRCODE_OK = 0
ERRCODE_NOFILE = 10

# ==============================================================================
# Lightweight structure to store results and merge them in a convenient way
evalres = namedtuple("evalres", ["mean_segmentation_precision", # None means undefined
                                 "mean_segmentation_recall",    # None means undefined
                                 "mean_detection_precision",    # None means undefined / redundant
                                 "mean_detection_recall",       # None means undefined / redundant
                                 "mean_jaccard_index_smartdoc", # None means undefined
                                 "mean_jaccard_index_segonly",  # None means undefined
                                 "count_total_frames",          # float at this level
                                 "count_true_accepted_frames",  # float at this level
                                 "count_true_rejected_frames",  # float at this level
                                 "count_false_accepted_frames", # float at this level
                                 "count_false_rejected_frames"])# float at this level

res_init = evalres(None, None, None, None, None, None, 0.0, 0.0, 0.0, 0.0, 0.0)


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
def res_model_to_tuple(result_model):
    cta = result_model.count_true_accepted_frames
    cf = result_model.count_total_frames
    cr = result_model.count_true_accepted_frames + result_model.count_false_accepted_frames
    res = evalres(
        result_model.mean_segmentation_precision if cta > 0 else None,
        result_model.mean_segmentation_recall if cta > 0 else None,
        result_model.detection_precision if cf > 0 else None,
        result_model.detection_recall if cf > 0 else None,
        result_model.mean_jaccard_index_smartdoc if cf > 0 else None,
        result_model.mean_jaccard_index_segonly if cr > 0 else None,
        float(result_model.count_total_frames),
        float(result_model.count_true_accepted_frames),
        float(result_model.count_true_rejected_frames),
        float(result_model.count_false_accepted_frames),
        float(result_model.count_false_rejected_frames))
    return res

def getOrDefault(value, default):
    # TODO add warning if using default
    return value if value is not None else default

def res_tuple_to_model(result_tuple):
    mdl = EvalSummary(
            version="0.3",
            software_used=Software(name=PROG_NAME_SHORT, version=PROG_VERSION))
    mdl.global_results = GlobalEvalResults()
    # Force to 0.0 only at the end of the process, so as not to loose information.
    mdl.global_results.mean_segmentation_precision = getOrDefault(result_tuple.mean_segmentation_precision, 0.0)
    mdl.global_results.mean_segmentation_recall    = getOrDefault(result_tuple.mean_segmentation_recall, 0.0)
    mdl.global_results.detection_precision         = getOrDefault(result_tuple.mean_detection_precision, 0.0)
    mdl.global_results.detection_recall            = getOrDefault(result_tuple.mean_detection_recall, 0.0)
    mdl.global_results.mean_jaccard_index_smartdoc = getOrDefault(result_tuple.mean_jaccard_index_smartdoc, 0.0)
    mdl.global_results.mean_jaccard_index_segonly  = getOrDefault(result_tuple.mean_jaccard_index_segonly, 0.0)
    mdl.global_results.count_total_frames          = int(result_tuple.count_total_frames)
    mdl.global_results.count_true_accepted_frames  = int(result_tuple.count_true_accepted_frames)
    mdl.global_results.count_true_rejected_frames  = int(result_tuple.count_true_rejected_frames)
    mdl.global_results.count_false_accepted_frames = int(result_tuple.count_false_accepted_frames)
    mdl.global_results.count_false_rejected_frames = int(result_tuple.count_false_rejected_frames)
    return mdl

# ==============================================================================
def merge_res_tuples(res1, res2):
    '''evalres x evalres ---> evalres'''
    # If any of two contains zero frames, return the other
    if res1.count_total_frames == 0:
        return res2
    if res2.count_total_frames == 0:
        return res1

    # Now both res contain frames

    # First merge counters
    count_total_frames          = res1.count_total_frames          + res2.count_total_frames
    count_true_accepted_frames  = res1.count_true_accepted_frames  + res2.count_true_accepted_frames
    count_true_rejected_frames  = res1.count_true_rejected_frames  + res2.count_true_rejected_frames
    count_false_accepted_frames = res1.count_false_accepted_frames + res2.count_false_accepted_frames
    count_false_rejected_frames = res1.count_false_rejected_frames + res2.count_false_rejected_frames

    # Segmentation precision and recall
    mean_segmentation_precision = None
    mean_segmentation_recall = None
    if count_true_accepted_frames > 0:
        mean_segmentation_precision = (
            ( getOrDefault(res1.mean_segmentation_precision, 0.0) * res1.count_true_accepted_frames
            + getOrDefault(res2.mean_segmentation_precision, 0.0) * res2.count_true_accepted_frames)
            / count_true_accepted_frames)
        mean_segmentation_recall = (
            ( getOrDefault(res1.mean_segmentation_recall, 0.0) * res1.count_true_accepted_frames
            + getOrDefault(res2.mean_segmentation_recall, 0.0) * res2.count_true_accepted_frames)
            / count_true_accepted_frames)
    else:
        logger.warn("No frame accepted while merging. Mean segmentation precision and recall left undefined.")

    # Detection precision and recall (adapted from eval_seg)
    count_expected  = count_true_accepted_frames + count_false_rejected_frames
    count_retrieved = count_true_accepted_frames + count_false_accepted_frames

    mean_detection_precision = None
    if count_retrieved > 0:
        mean_detection_precision = count_true_accepted_frames / count_retrieved
    else:
        logger.warn("No frame accepted while merging. Mean detection precision left undefined.")

    mean_detection_recall = None
    if count_expected > 0:
        mean_detection_recall = count_true_accepted_frames / count_expected
    else:
        logger.error("Cannot compute full sample recall if nothing is expected! Mean detection recall left undefined.")


    # Jaccard index
    mean_jaccard_index_smartdoc = None
    if count_total_frames > 0:
        mean_jaccard_index_smartdoc = (
            ( getOrDefault(res1.mean_jaccard_index_smartdoc, 0.0) * res1.count_total_frames
            + getOrDefault(res2.mean_jaccard_index_smartdoc, 0.0) * res2.count_total_frames)
            / count_total_frames)
    else:
        logger.error("No frame in sample. Mean Jaccard index (smartdoc variant) left undefined.")

    mean_jaccard_index_segonly = None
    if count_retrieved > 0:
        mean_jaccard_index_segonly = (
            ( getOrDefault(res1.mean_jaccard_index_segonly, 0.0) 
                * (res1.count_true_accepted_frames + res1.count_false_accepted_frames) 
            + getOrDefault(res2.mean_jaccard_index_segonly, 0.0) 
                * (res2.count_true_accepted_frames + res2.count_false_accepted_frames) )
            / count_retrieved)
    else:
        logger.error("No retreived frame in sample. Mean Jaccard index (segonly variant) left undefined.")

    # Prepare result
    res_agg = evalres(
        mean_segmentation_precision,
        mean_segmentation_recall,
        mean_detection_precision,
        mean_detection_recall,
        mean_jaccard_index_smartdoc,
        mean_jaccard_index_segonly,
        count_total_frames,
        count_true_accepted_frames,
        count_true_rejected_frames,
        count_false_accepted_frames,
        count_false_rejected_frames)
    # All done
    return res_agg


# ==============================================================================
def main(argv=None):
    # Option parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Merge page segmentation evaluation results.', 
        version=PROG_VERSION)


    parser.add_argument('-d', '--debug', 
        action="store_true", 
        help="Activate debug output.")
    parser.add_argument('-o', '--output-file', 
        help="Optional path to output file.")


    parser.add_argument('-f', '--files-from', metavar="FILE_LIST", 
        action=StoreValidFilePathOrStdin,
        help="File containing the list of files to merge, or '-' to use standard input. \
              Will be read BEFORE files specified on command line.")

    parser.add_argument('files', 
        action=StoreValidFilePaths,
        metavar='result_file', 
        nargs='*',
        help='EvalSummary or SegEval files containing global results to merge.')

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

    # --------------------------------------------------------------------------
    logger.debug("--- Process started. ---")
    # Init variables
    res_agg = res_init
    # Loop over files
    file_count = 0
    for eval_file in file_iter:
        logger.debug("Processing file '%s'" % eval_file)
        # Try to read either EvalResult or EvalSummary
        res_cur = res_model_to_tuple(read_results_from_file(eval_file))
        # Merge evaluation results
        res_agg = merge_res_tuples(res_cur, res_agg)
        # Logging
        logger.debug(
            "\t %d new frames (total is %d)",
            res_cur.count_total_frames,
            res_agg.count_total_frames)
        logger.debug(
            "\t AFTER: mean_segmentation_precision=%f ; mean_segmentation_recall  =%f",
            getOrDefault(res_agg.mean_segmentation_precision, 0.0),
            getOrDefault(res_agg.mean_segmentation_recall, 0.0))
        logger.debug(
            "\t        mean_detection_precision   =%f ; mean_detection_recall     =%f",
            getOrDefault(res_agg.mean_detection_precision, 0.0),
            getOrDefault(res_agg.mean_detection_recall, 0.0))
        logger.debug(
            "\t        mean_jaccard_index_smartdoc=%f ; mean_jaccard_index_segonly=%f",
            getOrDefault(res_agg.mean_jaccard_index_smartdoc, 0.0),
            getOrDefault(res_agg.mean_jaccard_index_segonly, 0.0))
        # Stats
        file_count += 1

    logger.debug("--- Process complete. ---")

    # --------------------------------------------------------------------------
    # Test for empty task and trap
    if file_count == 0:
        logger.error("No file processed. Output file will be useless so it is deactivated.")
        logger.error("\t Use '-h' option to review program synopsis.")
        return ERRCODE_NOFILE

    # else
    # Final output
    aggreg_mdl = res_tuple_to_model(res_agg)
    gr_mdl = aggreg_mdl.global_results
    logger.debug("------------------------------")
    logger.debug("Final results")
    logger.debug("------------------------------")
    logger.debug("Segmentation quality:")
    logger.info("\tmean segmentation precision  = %f", getOrDefault(gr_mdl.mean_segmentation_precision, 0.0))
    logger.info("\tmean segmentation recall     = %f", getOrDefault(gr_mdl.mean_segmentation_recall, 0.0))
    logger.debug("------------------------------")
    logger.debug("Detection quality:")
    logger.info("\tmean detection precision = %f", getOrDefault(gr_mdl.detection_precision, 0.0))
    logger.info("\tmean detection recall    = %f", getOrDefault(gr_mdl.detection_recall, 0.0))
    logger.debug("------------------------------")
    logger.debug("Jaccard index:")
    logger.info("\tmean ji smartdoc = %f", getOrDefault(gr_mdl.mean_jaccard_index_smartdoc, 0.0))
    logger.info("\tmean ji seg only = %f", getOrDefault(gr_mdl.mean_jaccard_index_segonly, 0.0))
    logger.debug("------------------------------")
    logger.debug("Frame counts:")
    logger.info("\ttotal_frames   = %d", gr_mdl.count_total_frames)
    logger.info("\ttrue_accepted  = %d", gr_mdl.count_true_accepted_frames)
    logger.info("\ttrue_rejected  = %d", gr_mdl.count_true_rejected_frames)
    logger.info("\tfalse_accepted = %d", gr_mdl.count_false_accepted_frames)
    logger.info("\tfalse_rejected = %d", gr_mdl.count_false_rejected_frames)
    logger.debug("- - - - - - - - - - - - - - - ")
    logger.debug("Note:")
    logger.debug("\texpected  = true_accept + false_reject = %d", (gr_mdl.count_true_accepted_frames + gr_mdl.count_false_rejected_frames))
    logger.debug("\tretrieved = true_accept + false_accept = %d", (gr_mdl.count_true_accepted_frames + gr_mdl.count_false_accepted_frames))
    logger.debug("------------------------------")
    logger.debug("")

    # Export the XML structure to file if needed
    if args.output_file is not None:
        aggreg_mdl.exportToFile(args.output_file, pretty_print=output_prettyprint)

    logger.debug("Clean exit.")
    logger.debug(DBGSEP)
    return ERRCODE_OK
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())

