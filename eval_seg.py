#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Test values (included in xml examples)
# ~3/4 coverage
# object_coord_gt = np.float32([[200, 200], [100, 800], [600, 800], [600, 300]])
# object_coord_test = np.float32([[300, 200], [100, 600], [500, 700], [600, 200]])
# perfect match
# object_coord_gt = np.float32([[200, 200], [100, 800], [600, 800], [600, 300]])
# object_coord_test = object_coord_gt
# no match
# object_coord_gt = np.float32([[200, 200], [100, 800], [600, 800], [600, 300]])
# object_coord_test = np.float32([[300, 100], [300, 200], [600, 200], [600, 100]])

# ==============================================================================
# Imports
import logging
import argparse
import os
import os.path
import sys
from collections import namedtuple

# from pprint import pprint as pp
import cv2
import numpy as np
import Polygon
import Polygon.Utils
import Polygon.IO # dbg

# ==============================================================================
# SegEval Tools suite imports
from utils.args import *
from utils.log import *
from models.models import *
from utils.polygon import *

# ==============================================================================
# (re)define logger after "from ... import *" (potential overwrite otherwise)
logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
PROG_VERSION = "0.4"
PROG_NAME = "Mobile Segmentation Evaluation Tool for Videos"
XML_VERSION_MIN = 0.2
XML_VERSION_MAX = 0.3


# ==============================================================================
def main(argv=None):
    # -----------------------------------------------------------------------------
    # Parser definition
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Evaluate the page segmentation results for a given video sequence.', 
        version=PROG_VERSION,
        epilog="""Segmentation and detection precision and recall are computed separately."""
    )

    parser.add_argument('groundtruth_file',
        action=StoreValidFilePath,
        help="File containing ground truth segmentation and object reference.")
    parser.add_argument('testresult_file', 
        action=StoreValidFilePath,
        help="File containing ground truth segmentation and object reference.")
    parser.add_argument('-d', '--debug', 
        action="store_true", 
        help="Activate debug output.")
    parser.add_argument('-o', '--output-file', 
        help="Optionnal path to output file.")

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

    # Create result model
    evalRes_mdl = EvalResult(
            version="0.3",
            software_used=Software(
                    name="SegEval",
                    version="0.3"))

    evalRes_mdl.source_files = EvalSourceFiles(
                groundtruth_file=args.groundtruth_file,
                segresult_file=args.testresult_file)

    # Let's go
    # --------------------------------------------------------------------------
    logger.debug("--- Process started. ---")
    # --------------------------------------------------------------------------

    # Input files models
    gt_mdl = GroundTruth.loadFromFile(args.groundtruth_file)
    test_mdl = SegResult.loadFromFile(args.testresult_file)

    # read ref object shape
    target_width = gt_mdl.object_shape.width
    target_height = gt_mdl.object_shape.height
    # Referential: (0,0) at TL, x > 0 toward right and y > 0 toward bottom
    # Corner order: TL, BL, BR, TR
    # object_coord_target = np.float32([[0, 0], [0, target_height], [target_width, target_height], [target_width, 0]])
    object_coord_target = np.float32([[0, target_height], [0, 0], [target_width, 0], [target_width, target_height]])


    if len(gt_mdl.segmentation_results) != len(test_mdl.segmentation_results):
        err = "ERROR: Number of frames is different in ground truth and test result XML files."
        logger.error(err)
        raise Exception(err)

    # Check reject case and compute geometric match
    frame_prec_acc = 0.0
    frame_rec_acc = 0.0
    frame_ji_smartdoc_acc = 0.0
    frame_ji_seg_acc = 0.0
    jaccard_index_smartdoc = 0.0
    jaccard_index_segonly = 0.0
    count_true_accept = 0
    count_false_accept = 0
    count_true_reject = 0
    count_false_reject = 0

    error_selfintersections_count = 0 # polygons self-intersection count (errors)

    for idx in range(0, len(gt_mdl.segmentation_results)):
        frame_gt = gt_mdl.segmentation_results[idx]
        frame_test = test_mdl.segmentation_results[idx]
        fidx = idx+1
        # logger.error("frame %03d" % fidx) # dbg

        fr = FrameEvalResult(index=fidx)

        # TODO change vocabulary? use containsObject(ref)? build another joining generator?
        rej_gt = frame_gt.rejected
        rej_test = frame_test.rejected

        if rej_gt and rej_test:
            logger.debug("frame %03d: correct reject \t# in gt and test" % fidx)
            count_true_reject += 1
            fr.match_type = TRUE_REJECTED_STR
            jaccard_index = 1.0
            fr.jaccard_index_smartdoc = 1.0
            frame_ji_smartdoc_acc  += jaccard_index
        elif not rej_gt and rej_test:
            logger.debug("frame %03d: false reject \t# in test but not in gt" % fidx)
            count_false_reject += 1
            fr.match_type = FALSE_REJECTED_STR
            # jaccard_index = 0.0
            fr.jaccard_index_smartdoc = 0.0
            # frame_ji_smartdoc_acc  += jaccard_index
        elif rej_gt and not rej_test:
            logger.debug("frame %03d: false accept \t# in gt but not in test" % fidx)
            count_false_accept += 1
            fr.match_type = FALSE_ACCEPTED_STR
            # jaccard_index = 0.0
            fr.jaccard_index_smartdoc = 0.0
            # frame_ji_smartdoc_acc  += jaccard_index
        else:
        # not rej_gt and not rej_test => we have to compare unwarped shapes
            logger.debug("frame %03d: true accept \t# in gt and in test" % fidx)
            count_true_accept +=1
            fr.match_type = TRUE_ACCEPTED_STR

            object_coord_gt = np.float32([[frame_gt.points['tl'].x,
                                           frame_gt.points['tl'].y,
                                           frame_gt.points['bl'].x,
                                           frame_gt.points['bl'].y,
                                           frame_gt.points['br'].x,
                                           frame_gt.points['br'].y,
                                           frame_gt.points['tr'].x,
                                           frame_gt.points['tr'].y]])
            object_coord_test = np.float32([[frame_test.points['tl'].x,
                                             frame_test.points['tl'].y,
                                             frame_test.points['bl'].x,
                                             frame_test.points['bl'].y,
                                             frame_test.points['br'].x,
                                             frame_test.points['br'].y,
                                             frame_test.points['tr'].x,
                                             frame_test.points['tr'].y]])

            # 1/ Compute Ĥ = perfect homography from gt frame coordinates to target coordinates
            H = cv2.getPerspectiveTransform(object_coord_gt.reshape(-1, 1, 2), object_coord_target.reshape(-1, 1, 2))
            
            # 2/ Apply to test result to project in target referential
            test_coords = cv2.perspectiveTransform(object_coord_test.reshape(-1, 1, 2), H)

            # 3/ Compute intersection between target region and test result region
            # poly = Polygon.Polygon([(0,0),(1,0),(0,1)])
            poly_target = Polygon.Polygon(object_coord_target.reshape(-1,2))
            poly_test = Polygon.Polygon(test_coords.reshape(-1,2))
            poly_inter = None

            area_target = area_test = area_inter = area_union = 0.0
            # (sadly, we must check for self-intersecting polygons which mess the interection computation)
            # logger.error("~~~ poly_target %s" % poly_target) # dbg
            if isSelfIntersecting(poly_target):
                msg = "frame %03d: Ground truth polygon is self intersecting. Aborting evaluation." % fidx
                logger.error(msg)
                raise ValueError(msg)
            area_target = poly_target.area()

            # logger.error("~~~ poly_test %s" % poly_test) # dbg
            if isSelfIntersecting(poly_test) :
                reject_shape = True
                error_selfintersections_count += 1
                logger.warning("frame %03d: Test result polygon is self intersecting. Assuming null surfaces instead." % fidx)
                # TODO log errors and suspicious frames in result file!
            else :
                # logger.error("~~~ poly_inter %s" % poly_inter) # dbg
                poly_inter = poly_target & poly_test
                # Polygon.IO.writeSVG('_tmp/polys-%03d.svg'%fidx, [poly_target, poly_test, poly_inter]) # dbg
                # poly_inter should not self-intersect, but may have more than 1 contour
                area_test = poly_test.area()
                area_inter = poly_inter.area()

                # Little hack to cope with float precision issues when dealing with polygons:
                #   If intersection area is close enough to target area or GT area, but slighlty >,
                #   then fix it, assuming it is due to rounding issues.
                area_min = min(area_target, area_test)
                if area_min < area_inter and area_min * 1.0000000001 > area_inter :
                    area_inter = area_min
                    logger.debug("Capping area_inter.")
                
            area_union = area_test + area_target - area_inter

            # Polygon.IO.writeSVG('polys.svg', [poly_target, poly_test, poly_inter]) # dbg

            # 4-5/ Compute segmentation precision and recall
            precision_frame = 0.0
            recall_frame    = 0.0
            if area_test == 0:
                # Actually, it's only the precision which is undefined, but we can extend the domain
                # considering the limit:
                # lim_(x->0) 0/x = 0 ## http://www.wolframalpha.com/input/?i=lim+0%2Fx+as+x-%3E0
                logger.warning("frame %03d: Test area surface is null. Setting segmentation precision and recall to 0." % fidx)
            else:
                precision_frame = area_inter / area_test
                recall_frame = area_inter / area_target
                if area_target < area_inter or area_test < area_inter:
                    msg = "frame %03d: area_inter is bigger than area_target or area_test." % (fidx, )
                    logger.error(msg)
                    logger.debug("area_target = %f (%s) ; area_test = %f (%s) ; area_inter = %f (%s)" 
                            % (area_target, float.hex(area_target), area_test, float.hex(area_test), area_inter, float.hex(area_inter)))
                    raise ValueError(msg)
                if precision_frame < 0.0 or precision_frame > 1.0:
                    msg = "frame %03d: precision_frame = %f not in [0.0, 1.0]." % (fidx, precision_frame)
                    logger.error(msg)
                    logger.debug("area_target = %f (%s) ; area_test = %f (%s) ; area_inter = %f (%s)" 
                            % (area_target, float.hex(area_target), area_test, float.hex(area_test), area_inter, float.hex(area_inter)))
                    raise ValueError(msg)
                if recall_frame < 0.0 or recall_frame > 1.0:
                    msg = "frame %03d: recall_frame = %f not in [0.0, 1.0]." % (fidx, recall_frame)
                    logger.error(msg)
                    logger.debug("area_target = %f (%s) ; area_test = %f (%s) ; area_inter = %f (%s)" 
                            % (area_target, float.hex(area_target), area_test, float.hex(area_test), area_inter, float.hex(area_inter)))
                    raise ValueError(msg)

                # assert (0.0 <= precision_frame and precision_frame <= 1.0), "Segmentation precision must be in [0.0, 1.0]."
                # assert (0.0 <= recall_frame and recall_frame <= 1.0), "Segmentation recall must be in [0.0, 1.0]."
            jaccard_index = area_inter / area_union
            fr.segmentation_precision = precision_frame
            fr.segmentation_recall = recall_frame
            fr.jaccard_index_segonly = jaccard_index
            fr.jaccard_index_smartdoc = jaccard_index
            fr.surfaces = SegSurfaces(
                            test=area_test,
                            intersection=area_inter)

            frame_prec_acc += precision_frame
            frame_rec_acc  += recall_frame
            frame_ji_smartdoc_acc  += jaccard_index
            frame_ji_seg_acc += jaccard_index
            logger.debug("\tsegmentation quality: prec=%f ; rec=%f ; ji=%f" 
                % (precision_frame, recall_frame, jaccard_index))
        # /if true accept
        # Keep current result
        evalRes_mdl.frame_results.append(fr)
    # --------------------------------------------------------------------------

    # Prepare final score
    # evalRes_mdl.results.global_score MUST be configured after
    count_total = len(evalRes_mdl.frame_results)
    evalRes_mdl.global_results = GlobalEvalResults()
    evalRes_mdl.global_results.count_total_frames = count_total
    evalRes_mdl.global_results.count_true_accepted_frames = count_true_accept
    evalRes_mdl.global_results.count_true_rejected_frames = count_true_reject
    evalRes_mdl.global_results.count_false_accepted_frames = count_false_accept
    evalRes_mdl.global_results.count_false_rejected_frames = count_false_reject

    # Detection precision/recall for full sample (sequence of frames)
    count_expected = count_true_accept + count_false_reject
    count_retrieved = count_true_accept + count_false_accept

    evalRes_mdl.global_results.detection_precision = 0.0
    if count_retrieved > 0:
        evalRes_mdl.global_results.detection_precision = float(count_true_accept) / count_retrieved
    else:
        logger.warn("No frame accepted. Full sample precision set to %f" % evalRes_mdl.global_results.detection_precision)

    evalRes_mdl.global_results.detection_recall = 0.0
    if count_expected > 0:
        evalRes_mdl.global_results.detection_recall = float(count_true_accept) / count_expected
    else:
        logger.error("Cannot compute full sample recall if ground truth contains no accepted frame! Recall set to %f"
            % evalRes_mdl.global_results.detection_recall)

    # Precision/recall averaged for frames
    evalRes_mdl.global_results.mean_segmentation_precision = 0.0
    evalRes_mdl.global_results.mean_segmentation_recall    = 0.0
    if count_true_accept > 0:
        evalRes_mdl.global_results.mean_segmentation_precision = frame_prec_acc / count_true_accept
        evalRes_mdl.global_results.mean_segmentation_recall    = frame_rec_acc / count_true_accept
    else:
        logger.warn("Cannot compute mean segmentation precision and recall if nothing was accepted! Precision set to %f ; recall set to %f" 
            % (evalRes_mdl.global_results.mean_segmentation_precision, evalRes_mdl.global_results.mean_segmentation_recall))

    # Jaccard index averaged
    evalRes_mdl.global_results.mean_jaccard_index_smartdoc = frame_ji_smartdoc_acc / count_total

    if count_retrieved > 0:
        evalRes_mdl.global_results.mean_jaccard_index_segonly  = frame_ji_seg_acc / count_retrieved
    else:
        evalRes_mdl.global_results.mean_jaccard_index_segonly  = 0.0
        logger.error("Cannot compute JI for segmentation if nothing was accepted! ji_segonly set to %f" 
                % evalRes_mdl.global_results.mean_jaccard_index_segonly)


    # --------------------------------------------------------------------------
    logger.debug("--- Process complete. ---")

    # Final output
    logger.debug("------------------------------")
    logger.debug("Final results")
    logger.debug("------------------------------")
    logger.debug("Segmentation quality:")
    logger.info("\tmean frame precision  = %f" % evalRes_mdl.global_results.mean_segmentation_precision)
    logger.info("\tmean frame recall     = %f" % evalRes_mdl.global_results.mean_segmentation_recall)
    logger.debug("------------------------------")
    logger.debug("Detection quality:")
    logger.info("\tfull sample precision = %f" % evalRes_mdl.global_results.detection_precision)
    logger.info("\tfull sample recall    = %f" % evalRes_mdl.global_results.detection_recall)
    logger.debug("------------------------------")
    logger.debug("Jaccard index:")
    logger.info("\tmean ji smartdoc = %f" % evalRes_mdl.global_results.mean_jaccard_index_smartdoc)
    logger.info("\tmean ji seg only = %f" % evalRes_mdl.global_results.mean_jaccard_index_segonly)
    logger.debug("------------------------------")
    logger.debug("Frame counts:")
    logger.info("\ttotal_frames   = %d" % evalRes_mdl.global_results.count_total_frames)
    logger.info("\ttrue_accepted  = %d" % evalRes_mdl.global_results.count_true_accepted_frames)
    logger.info("\ttrue_rejected  = %d" % evalRes_mdl.global_results.count_true_rejected_frames)
    logger.info("\tfalse_accepted = %d" % evalRes_mdl.global_results.count_false_accepted_frames)
    logger.info("\tfalse_rejected = %d" % evalRes_mdl.global_results.count_false_rejected_frames)
    logger.debug("- - - - - - - - - - - - - - - ")
    logger.debug("Note:")
    logger.debug("\texpected  = true_accept + false_reject = %d" % (count_true_accept + count_false_reject))
    logger.debug("\tretrieved = true_accept + false_accept = %d" % (count_true_accept + count_false_accept))
    logger.debug("")
    if error_selfintersections_count > 0:
        logger.warning("Seg. results contain self-intersecting polygons in %d frame(s)."
                        % error_selfintersections_count)
        logger.warning("\tResults assume seg. prec. and rec. = 0 for those cases.")
        logger.debug("")


    # Export the XML structure to file if needed
    if args.output_file is not None:
        evalRes_mdl.exportToFile(args.output_file, pretty_print=output_prettyprint)

    logger.debug("Clean exit.")
    logger.debug(DBGSEP)
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())
