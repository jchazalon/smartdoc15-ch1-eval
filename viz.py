#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Visualization tool for segmentation files and videos.

Given a ground-truth file or a segmentation result and a video, allows to 
navigate threw the video with segmentation overlaid.

Future work:
- auto discover sample from 'source_sample_file' (+ datasetroot)
- (make video an option, and keep minimalist control)
- allow manual editing
- allow manual inpainting
- play/pause
- better UI
- application framework (resource management, main loop abstracted, etc.)
- turn into a command
'''

# ==============================================================================
# Imports
import logging
import argparse
import os
import os.path
import sys

import cv2
import numpy as np

# ==============================================================================
# mobileSeg Tools suite imports
from utils.args import *
from utils.cache import lru_cache
from utils.log import initLogger
from utils.io import VideoSeeker

from models.models import *

# ==============================================================================
logger = logging.getLogger(__name__)

# ==============================================================================

EXITCODE_OK = 0
EXITCODE_UNKERR = 254


class VizController(object):
    CACHE_STEP_SIZE = 10

    def __init__(self, videofile, segfile):
        self._seektrackbarname = "seek trackbar"
        self._ratiotrackbarname = "ratio trackbar"
        self._videofile = videofile
        self._segfile = segfile
        self._winname = "Viz - vid( %s ) seg( %s )" % (os.path.basename(videofile), os.path.basename(segfile))

        # Model
        self._seeker = VideoSeeker(videofile)

        self._datamdl = None ## added to seeker
            # Subclass of SegResult with segmentation_results in both cases
        try:
            self._datamdl = GroundTruth.loadFromFile(segfile)
        except:
            try:
                self._datamdl = SegResult.loadFromFile(segfile)
            except:
                err = "Cannot load '%s', format not supported." % segfile
                logger.error(err)
                raise Exception(err)

        src = self._datamdl.source_sample_file
        if os.path.splitext(os.path.basename(src))[0] != os.path.splitext(os.path.basename(videofile))[0]:
            logger.warning("Video file '%s' does not seem to be the sample file '%s' was created from." 
                                %(videofile, segfile))
            logger.warning("\texpected sample file is: '%s'" % src)

        self._segres = self._datamdl.segmentation_results

        # Views
        cv2.namedWindow(self._winname) # auto resize
        cv2.createTrackbar(self._seektrackbarname,
                           self._winname,
                           0,
                           self._seeker.frame_count - 1,
                           self._set_current_frameId)

        cv2.createTrackbar(self._ratiotrackbarname,
                           self._winname,
                           100, # = 100%
                           300, # = 300%
                           self._trackbar_ratio_to_ratio)

        # Inner state
        self._finished = False
        self._ratio = 1.0
        self._refresh_required = True
        self._current_frameId = 0

    def release(self):
        # cv2.destroyAllWindows()
        cv2.destroyWindow(self._winname)
        self._seeker.release()
        self._seeker = None
        # self._getFrame.clear()


    @property
    def ratio(self):
        return self._ratio
    @ratio.setter
    def ratio(self, value):
        if value < 0 or value > 10:
            raise ValueError("ratio must be >0 and <=10, got '%f'" % value)
        self._ratio = value
        self._refresh_required = True

    def _trackbar_ratio_to_ratio(self, value):
        if value < 1:
            cv2.setTrackbarPos(self._ratiotrackbarname, self._winname, 1)
        else:
            real_val = float(value) / 100
            self.ratio = real_val


    def _getCurrentFrame(self):
        return self._getFrame(self.current_frameId)

    def _getFrame(self, frame_id):
        return self._seeker.getFrame(frame_id)

    def _get_current_frameId(self):
        return self._current_frameId
    def _set_current_frameId(self, value):
        if self._current_frameId != value:
            self._current_frameId = value
            self._refresh_required = True
    current_frameId = property(_get_current_frameId, _set_current_frameId)


    def _displayCurrentFrame(self):
        while self._refresh_required:
            self._refresh_required = False
            frame = self._getCurrentFrame()
            self._overlaySegmentation(frame) ## added to seeker

            ratio = self._ratio
            frame_scaled = None
            if ratio == 1.0:
                frame_scaled = frame
            else:
                frame_scaled = cv2.resize(frame, tuple(map(lambda x: int(x * ratio), self._seeker.frame_size)))

            cv2.imshow(self._winname, frame_scaled)


    def _overlaySegmentation(self, frame):
        fres = self._segres[self.current_frameId]
        if (fres.index - 1) != self.current_frameId: # fres ids start at 1
            logger.warning("Video @f%04d out of sync with seg @f%04d" % (fres.index, self.current_frameId))

        if fres.rejected:
            # Simply draw circle
            (frame_height, frame_width, _depth) = frame.shape
            cv2.circle(frame, (frame_width/2, frame_height/2), 20, (0, 0, 255), 10)
        else:
            # Draw polygon
            object_shape = np.float32([[fres.points['tl'].x, fres.points['tl'].y],
                                       [fres.points['bl'].x, fres.points['bl'].y],
                                       [fres.points['br'].x, fres.points['br'].y],
                                       [fres.points['tr'].x, fres.points['tr'].y]])
            cv2.polylines(frame, [np.int32(object_shape)], True, (0, 255, 0), 2)


    def _onForward(self):
        logger.debug("Forward")
        if self.current_frameId < self._seeker.frame_count - 1:
            self.current_frameId += 1
            cv2.setTrackbarPos(self._seektrackbarname, self._winname, self.current_frameId)
            self._refresh_required = True


    def _onBackward(self):
        logger.debug("Backward")
        if self.current_frameId > 0:
            self.current_frameId -= 1
            cv2.setTrackbarPos(self._seektrackbarname, self._winname, self.current_frameId)
            self._refresh_required = True

    def _onQuit(self):
        logger.info("Quit requested.")
        self._finished = True


    def main_loop(self):
        logger.debug("Entered application main loop.")
        iterWithoutRepaintCount = 0

        while not self._finished:
            # Repainting management
            if self._refresh_required:
                self._displayCurrentFrame()
                iterWithoutRepaintCount = 0

            # Keybard event active wait
            key = cv2.waitKey(100)
            if key == -1:
                iterWithoutRepaintCount += 1
                if iterWithoutRepaintCount >= 50:
                    self._refresh_required = True
                    logger.debug("(auto refresh requested)")
            else:
                key &= 0xFF
                if key == ord('q'):
                    self._onQuit()
                elif key in [ord('f'), ord('n')]:
                    self._onForward()
                elif key in [ord('b'), ord('d')]:
                    self._onBackward()
                else:
                    logger.debug("No action for key '%c'" % key)


        return EXITCODE_OK


# ==============================================================================
# ==============================================================================
def main(argv=None):
    # Option parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Visualizer for segmentation result or ground-truth files.')

    parser.add_argument('input_video', action=StoreValidFilePath)
    parser.add_argument('seg_file', action=StoreValidFilePath)


    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    initLogger(logger, debug=False)
    dumpArgs(args, logger)

    # --------------------------------------------------------------------------
    # Prepare process
    logger.debug("Starting up")

    app = VizController(args.input_video, args.seg_file)

    # Let's test video processing
    # --------------------------------------------------------------------------
    logger.debug("--- Process started. ---")
    # --------------------------------------------------------------------------
    ret = EXITCODE_UNKERR
    try:
        ret = app.main_loop()
    finally:
        app.release()

    # --------------------------------------------------------------------------
    logger.debug("--- Process complete. ---")
    # --------------------------------------------------------------------------

    return ret

# ==============================================================================
# ==============================================================================
if __name__ == "__main__":
    ret = main(sys.argv)
    if ret is not None:
        sys.exit(ret)
