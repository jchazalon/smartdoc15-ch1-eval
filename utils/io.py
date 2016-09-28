#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
# Imports
import os
import os.path
import sys
import glob

import cv,cv2
import numpy as np

# ==============================================================================
# mobileSeg Tools suite imports

# ==============================================================================
from utils.log import createAndInitLogger
logger = createAndInitLogger(__name__)
# logger = createAndInitLogger(__name__, debug=True)

# ==============================================================================

class FrameData(object): 
    """
    Simple container for frame data.
    May be extended later with more data: timestamp, sensors, etc.
    """
    def __init__(self, mat, basefile, index=-1):
        """FrameData x cv2.Mat x str x int ---> None"""
        self._mat = mat
        self._basefile = basefile
        self._index = index

    @property
    def mat(self):
        """FrameData ---> cv2.Mat"""
        return self._mat

    @property
    def basefile(self):
        return self._basefile

    @property
    def index(self):
        """Index of the current frame in the sample. Starts at 1, 0 or less means unknown."""
        return self._index
    
    

class FrameDataFromMat(FrameData):
    def __init__(self, mat, basefile=None, index=-1):
        super(FrameDataFromMat, self).__init__(mat, basefile, index)


class FrameDataFromFile(FrameData):
    def __init__(self, image_path, index=-1):
        super(FrameDataFromFile, self).__init__(cv2.imread(os.path.abspath(image_path)), image_path, index)


#######

class FrameSequence(object):
    # TODO add seek methods to allow navigation for GT tool (see demo_video_seek)
    def __iter__(self):
        return self

    def next(self):
        raise NotImplementedError("Cannot call FrameSequence.next() directly (abstract). Must use concrete subclass.")

    def release(self):
        pass


class FrameSequenceFromVideo(FrameSequence):
    def __init__(self, videofile):
        if not os.path.exists(videofile):
            err = "'%s' does not exist." % videofile
            logger.error(err)
            raise IOError(err)

        self._videofile = videofile
        self._videocap = cv2.VideoCapture(videofile)
        self._frame_count = int(self._videocap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self._frame_size = (int(self._videocap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(self._videocap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

        logger.info("Input video informations:")
        logger.info("\tframe_count = %d" % self._frame_count)
        logger.info("\tframe_size = %dx%d" % self._frame_size)

        self._cfid = 0
        self._prevRes = True


    def next(self):
        if self._prevRes and self._cfid < self._frame_count:
            self._prevRes, frame = self._videocap.read()
            self._cfid += 1
            if self._prevRes:
                return FrameDataFromMat(frame, "%s?fid=%04d" % (self._videofile, self._cfid), self._cfid)
        # else
        raise StopIteration

    def release(self):
        self._videocap.release()
        self._videocap = None


class FrameSequenceFromGlob(FrameSequence):
    def __init__(self, globbing):
        self._frames = sorted(glob.glob(globbing))
        self._frame_count = len(self._frames)

        if self._frame_count == 0:
            err = "'%s' does not match any file." % globbing
            logger.error(err)
            raise IOError(err)

        logger.info("Input video informations:")
        logger.info("\tframe_count = %d" % self._frame_count)

        self._cfid = 0

    def next(self):
        if self._cfid < self._frame_count:
            res = FrameDataFromFile(self._frames[self._cfid], self._cfid)
            self._cfid += 1
            return res
        # else
        raise StopIteration    


def frameIteratorFromInput(input_sample):
    '''
    str ---> FrameSequence

    Given an input filename (video) or globbing (sequence of image files), generate a frame iterator.
    '''
    # TODO raise an exception if video codec is not available
    frames = None
    if input_sample.endswith((".mp4", ".avi")):
        frames = FrameSequenceFromVideo(input_sample)
    else:
        frames = FrameSequenceFromGlob(input_sample)
    return frames


# ==============================================================================
# ==============================================================================
# TODO FrameFilesSeeker

# ==============================================================================
# ==============================================================================
class VideoSeeker(object):
    """
    Wrapper over OpenCV VideoCapture object.
    Enables video seeking while preventing from desynchronization (due to the 
    lack of key frame management, apparently).

    NOT adapted for long video sequences.
    """

    # Init and release

    def __init__(self, videofile, cache_size=30):
        assert cache_size > 0, "cache_size must be > 0"
        self._vin = cv2.VideoCapture(videofile)
        self._filename = videofile
        # Video information
        self._frame_count = int(self._vin.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self._frame_size = (int(self._vin.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), 
                            int(self._vin.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        self._fps = self._vin.get(cv2.cv.CV_CAP_PROP_FPS)
        self._frame_duration = 1000 / self._fps
        self._maxtime = self._frame_duration * self._frame_count
        # debug output
        logger.debug("Input video informations (from OCV codec):")
        logger.debug("\tfile: '%s'" % self._filename)
        logger.debug("\tframe_count = %d" % self._frame_count)
        logger.debug("\tframe_size = %dx%d" % self._frame_size)
        logger.debug("\tfps = %0.2f" % self._fps)
        logger.debug("\tframe_duration = %0.2f" % self._frame_duration)
        logger.debug("\tmaxtime = %0.2f" % self._maxtime)
        # Cache init
        cache_len = min(cache_size, self._frame_count)
        self._cache = [self._readCurrentFrameCodecCheck() for _i in range(cache_len)]
        # The cache contains the `cache_len` elements before (not including) self._current_pos

    def release(self):
        logger.debug("Releasing resources.")
        self._vin.release()


    # Visible properties
    @property
    def frame_size(self):
        return self._frame_size

    @property
    def frame_count(self):
        return self._frame_count


    # Seeking property and methods
    def current_pos(self):
        return self._current_pos()

    @property
    def _current_pos(self):
        return int(self._vin.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
    @_current_pos.setter
    def _current_pos(self, value):
        # Seeking by frame id or timestamp seems to be both as slow, and become equally slower at the end of the stream
        if self._current_pos != value:
            self._seek_by_frame_posframes(value)
    
    def _checkFrameId(self, frame_id):
        return 0 <= frame_id and frame_id < self._frame_count

    def _checkRaiseFrameId(self, frame_id):
        if not self._checkFrameId(frame_id):
            raise ValueError("Invalid frame id: '%d', [0 ; frame_count (%d)[" % (frame_id, self._frame_count))

    def _seek_by_frame_posframes(self, frame_id):
        self._checkRaiseFrameId(frame_id)
        if self._current_pos < frame_id:
            self._forwardUntil(frame_id)
        elif self._current_pos > frame_id:
            self._rewind()
            self._forwardUntil(frame_id)
        else:
            pass

    def _rewind(self):
        if not self._vin.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0):
            raise IOError("Error while setting frame id to 0")

    def _forwardUntil(self, frame_id):
        self._checkRaiseFrameId(frame_id)
        if frame_id < self._current_pos:
            raise Exception("Illegal state, cannot call _forwardUntil with target frame_id (%d) > _current_pos (%d)." % (frame_id, self._current_pos))
        while self._current_pos < frame_id: self.getFrame(self._current_pos)


    # Codec read methods
    def _readCurrentFrameCodec(self):
        '''Assumes `self._currentFrame` is valid.'''
        logger.debug("io.VideoSeeker._readCurrentFrameCodec() %s @ %d" % (os.path.basename(self._filename), self._current_pos))
        res, frame = self._vin.read()
        # cursor auto advance is automatic here
        # self._current_pos -= 1
        if not res:
            raise IOError("Error while reading frame")
        return frame

    def _readCurrentFrameCodecCheck(self):
        if not self._checkFrameId(self._current_pos):
            raise Exception("Invalid current position in stream.")
        return self._readCurrentFrameCodec()


    # Public read method
    def getFrame(self, frame_id):
        if self._isInCache(frame_id):
            return self._readCache(frame_id)
        else:
            self._current_pos = frame_id
            mat = self._readCurrentFrameCodecCheck()
            self._updateCache(frame_id, mat)
            return mat


    # Cache management methods
    def _isInCache(self, frame_id):
        return self._cache_first <= frame_id and frame_id <= self._cache_last

    @property
    def _cache_first(self):
        return self._current_pos - len(self._cache)
    
    @property
    def _cache_last(self):
        return self._current_pos - 1

    def _cacheIndex(self, frame_id):
        return frame_id % len(self._cache)

    def _updateCache(self, frame_id, mat):
        if not self._isInCache(frame_id):
            raise ValueError("frame id %d is not in cache range (%d -> %d), cannot store its value." % (frame_id, self._cache_first, self._cache_last))
        logger.debug("io.VideoSeeker._updateCache %s @ %d" % (os.path.basename(self._filename), frame_id))
        self._cache[self._cacheIndex(frame_id)] = mat

    def _readCache(self, frame_id):
        if not self._isInCache(frame_id):
            raise ValueError("frame id %d is not in cache range (%d -> %d), cannot read its value." % (frame_id, self._cache_first, self._cache_last))
        logger.debug("io.VideoSeeker._readCache %s @ %d" % (os.path.basename(self._filename), frame_id))
        return self._cache[self._cacheIndex(frame_id)]


    # Iteration methods 
    def __iter__(self):
        self._rewind()
        return self

    def hasNext(self):
        return self._current_pos < self._frame_count

    def next(self):
        if self._checkFrameId(self._current_pos):
            frame = self.getFrame(self._current_pos) # fetches next uncached frame
            return frame
        else:
            raise StopIteration


