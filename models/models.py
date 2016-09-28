#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains the models used in the mobile segmentation evaluation suite.
The classes defined here only contain structural information and mapping to XML representation.
Operations on those structures are defined in separate packages ("processing", in particular).
"""

import logging
import os
import os.path
import sys
import datetime

import lxml.etree as etree  # import xml.etree as etree
import lxml.sax   as sax  # import xml.sax   as sax
from xml.dom.pulldom import SAX2DOM

import dexml
from dexml import fields

# ==============================================================================
logger = logging.getLogger(__name__)

# ==============================================================================
# Models

# To speed up dexml, I used lxml instead of minidom to parse and pretty print 
# xml.
# Reference: http://lxml.de/sax.html#interfacing-with-pulldom-minidom
# To pass a dom object directly to the MyModel.parse function:
####
# import lxml.etree
# import lxml.sax
# from xml.dom.pulldom import SAX2DOM
# tree = lxml.etree.parse(f)
# handler = SAX2DOM()
# lxml.sax.saxify(tree, handler)
# dom = handler.document
#
# mdl = MyModel.parse(dom)
####
# To pretty print:
####
# root = lxml.etree.parse(FILE)
# string = lxml.etree.tostring(root, xml_declaration=True, encoding='utf-8', pretty_print=True)
####


# dexml extensions / hacks
# ------------------------------------------------------------------------------
class RestrictedString(fields.Value):
    """Field representing a string value which is restricted 
    to a set a legal values.
    """

    class arguments(fields.Value.arguments):
        case_sensitive = False

    def __init__(self, *legal_values, **kwds):
        super(RestrictedString,self).__init__(**kwds)
        self.legal_values = []
        for lv in legal_values:
            if not self.case_sensitive:
                lv = str(lv).lower()
            self.legal_values.append(lv)


    def _check_restrictions(self, val):
        testval = val
        if not self.case_sensitive:
            testval = testval.lower()
        return testval in self.legal_values

    def __set__(self,instance,value):
        if not self._check_restrictions(value):
            raise ValueError("Illegal value '%s' for restricted string (%s)." 
                             % (value, self.legal_values))
        instance.__dict__[self.field_name] = value

    def parse_value(self,val):
        if not self._check_restrictions(val):
            raise dexml.ParseError("Illegal value '%s' for restricted string (%s)." 
                             % (val, self.legal_values))
        return val


# GENERAL components
# ------------------------------------------------------------------------------
# TODO make a Version component with proper semantic version numbers?

class Software(dexml.Model):
    """Describe the software used to produce a file."""
    class meta:
        tagname = "software_used"
    name    = fields.String()
    version = fields.String()

class Size(dexml.Model):
    """Simple size abstraction. Floats used, beware!"""
    width  = fields.Float()
    height = fields.Float()

class MainModel(dexml.Model):
    """Basis for main models. Manages automatically the generation timestamp.
    Version "number" and optionnal software information are still to provide."""
    version   = fields.String()
    generated = fields.String() # Could be a date object at some point
    software_used  = fields.Model(Software, required=False)

    def __init__(self,**kwds):
        super(MainModel, self).__init__(**kwds)
        self.version   = "0.3"
        self.generated = datetime.datetime.now().isoformat() # warning: no TZ info here

    def exportToFile(self, filename, pretty_print=False):
        out_str = self.render(encoding="utf-8") # no pretty=pretty_print here, done by lxml
        if pretty_print:
            # lxml pretty print should be way faster than the minidom one used in dexml
            root = etree.fromstring(out_str)
            out_str = etree.tostring(root, xml_declaration=True, encoding='utf-8', pretty_print=True)
        logger.debug("Output file's content:\n%s" % out_str)
        # print "Output file's content:\n%s" % out_str
        with open(os.path.abspath(filename), "wb") as out_f:
            out_f.write(out_str)

    @classmethod 
    def loadFromFile(cls, filename):
        path_file = os.path.abspath(filename)
        if not os.path.isfile(path_file):
            err = "Error: '%s' does not exist or is not a file." % filename
            logger.error(err)
            raise Exception(err)

        # Note: parsing a file directly with dexml/minidom is supposedly slower, si I used lxml one, 
        #       but I did not benchmark it.
        tree = etree.parse(path_file)
        handler = SAX2DOM()
        sax.saxify(tree, handler)
        dom = handler.document

        # In case, you can pass the filename to parse() here to skip lxml
        mdl = cls.parse(dom)
        return mdl

    # def __repr__(self):
    #     return "%s(%r)" % (self.__class__, self.__dict__)

# Sample
# ------------------------------------------------------------------------------
class FrameSize(Size):
    class meta:
        tagname = "frame_size"

class Frame(dexml.Model):
    """Store information about a frame to process: index in original stream, filename of extracted version, etc."""
    class meta:
        tagname = "frame"
    index     = fields.Integer(required=False)
    time      = fields.Float(required=False)
    filename  = fields.String() # can be relative to root given to a tool on command line

class Sample(MainModel):
    """Main model for raw test samples (sequences of frames) used to benchmark tools."""
    class meta:
        tagname = "sample"
    frame_size = fields.Model(FrameSize)
    frames     = fields.List(Frame, tagname="frames")


# SegResult
# ------------------------------------------------------------------------------
class Pt(dexml.Model):
    """Simple point class."""
    class meta:
        tagname = "point"
    name = RestrictedString("tl","bl","br","tr")
    x = fields.Float()
    y = fields.Float()

class FrameSegResult(dexml.Model):
    """Tracker output for a given frame. Only 1 object can be detected for now."""
    class meta:
        tagname = "frame"
    index     = fields.Integer(required=False)
    rejected  = fields.Boolean()
    points    = fields.Dict(Pt, key='name', unique=True)

    # def get_tl(self): return self.points['tl']
    # def set_tl(self, value): self.points['tl'] = value
    # tl = property(get_tl, set_tl)
    # def get_bl(self): return self.points['bl']
    # def set_bl(self, value): self.points['bl'] = value
    # bl = property(get_bl, set_bl)
    # def get_br(self): return self.points['br']
    # def set_br(self, value): self.points['br'] = value
    # br = property(get_br, set_br)
    # def get_tr(self): return self.points['tr']
    # def set_tr(self, value): self.points['tr'] = value
    # tr = property(get_tr, set_tr)



class SegResult(MainModel):
    """Main model for tracker output when run on a sample."""
    class meta:
        tagname = "seg_result"
    # FIXME relative or absolute? Should be **relative** to dataset root
    source_sample_file   = fields.String(tagname="source_sample_file")
    segmentation_results = fields.List(FrameSegResult, tagname="segmentation_results")


# GroundTruth
# ------------------------------------------------------------------------------
class ObjectShape(Size):
    class meta:
        tagname = "object_shape"


class GroundTruth(SegResult):
    """Main model for ground truthing tool output when run on a sample.
    Like segResult, but with a reference object to facilitate evaluation."""
    class meta:
        tagname = "ground_truth"
    object_shape = fields.Model(ObjectShape)


# EvalResult
# ------------------------------------------------------------------------------

class EvalSourceFiles(dexml.Model):
    class meta:
        tagname = "source_files"
    groundtruth_file = fields.String()
    segresult_file   = fields.String()

class SegSurfaces(dexml.Model):
    class meta:
        tagname = "seg_surfaces"
    # groundtruth = fields.Float() # redundant
    test         = fields.Float()
    intersection = fields.Float()

# Frame result types
TRUE_ACCEPTED_STR = "true_accepted"
TRUE_REJECTED_STR = "true_rejected"
FALSE_ACCEPTED_STR = "false_accepted"
FALSE_REJECTED_STR = "false_rejected"

class FrameEvalResult(dexml.Model):
    class meta:
        tagname = "frame"
    index      = fields.Integer(required=False)
    match_type = RestrictedString(TRUE_ACCEPTED_STR, TRUE_REJECTED_STR, FALSE_ACCEPTED_STR, FALSE_REJECTED_STR)
    segmentation_precision  = fields.Float(default=0.0, required=False, tagname="segmentation_precision")
    segmentation_recall     = fields.Float(default=0.0, required=False, tagname="segmentation_recall")
    jaccard_index_smartdoc  = fields.Float(default=0.0, required=False, tagname="jaccard_index_smartdoc")
    jaccard_index_segonly   = fields.Float(default=0.0, required=False, tagname="jaccard_index_segonly")
    surfaces   = fields.Model(SegSurfaces, required=False)

class GlobalEvalResults(dexml.Model):
    class meta:
        tagname = "global_results"
    detection_precision         = fields.Float(tagname="detection_precision")
    detection_recall            = fields.Float(tagname="detection_recall")
    mean_segmentation_precision = fields.Float(tagname="mean_segmentation_precision")
    mean_segmentation_recall    = fields.Float(tagname="mean_segmentation_recall")
    mean_jaccard_index_smartdoc = fields.Float(tagname="mean_jaccard_index_smartdoc")
    mean_jaccard_index_segonly  = fields.Float(tagname="mean_jaccard_index_segonly")
    count_total_frames          = fields.Integer(tagname="count_total_frames")
    count_true_accepted_frames  = fields.Integer(tagname="count_true_accepted_frames")
    count_true_rejected_frames  = fields.Integer(tagname="count_true_rejected_frames")
    count_false_accepted_frames = fields.Integer(tagname="count_false_accepted_frames")
    count_false_rejected_frames = fields.Integer(tagname="count_false_rejected_frames")


class EvalResult(MainModel):
    """Main model for evaluation result for a given sample."""
    class meta:
        tagname = "eval_results"
    source_files   = fields.Model(EvalSourceFiles)
    frame_results  = fields.List(FrameEvalResult, tagname="frame_results")
    global_results = fields.Model(GlobalEvalResults)


# EvalSummary
# ------------------------------------------------------------------------------
class EvalSummary(MainModel):
    """Main model for evaluation summary for a whole experiment."""
    class meta:
        tagname = "eval_summary"
    global_results = fields.Model(GlobalEvalResults)


# Experiment
# ------------------------------------------------------------------------------
class Test(dexml.Model):
    """Describes what a simple sample (list of frames) test consists in:
    a sample + an optionnal model of the object to track + optionnal 
    ground-truth."""
    class meta:
        tagname = "test"
    id                = fields.String(required=False)
    sample_file       = fields.String(tagname="sample_file")
    model_file        = fields.String(required=False, tagname="model_file")
    ground_truth_file = fields.String(required=False, tagname="ground_truth_file")
    # TODO intégrer directement le contenu des fichiers ici ?


class Experiment(MainModel):
    """Main model for defining an experimental setup, without results."""
    class meta:
        tagname = "experiment"
    test_set = fields.List(Test, tagname="test_set")



# InteractionCheckpoints
# ------------------------------------------------------------------------------
# class AnnotatedFrame(dexml.Model):
#     class meta:
#         tagname = "frame"
#     index     = fields.Integer()
#     marker_points    = fields.Dict(Pt, key='name', unique=True)
#     object_points    = fields.Dict(Pt, key='name', unique=True)


class InteractionCheckpoints(MainModel):
    """Main model for storing interaction information produced during ground truth annotation."""
    class meta:
        tagname = "interaction_checkpoints"
    marker_points    = fields.Dict(Pt, key='name', unique=True, tagname="marker_points")
    object_points    = fields.Dict(Pt, key='name', unique=True, tagname="object_points")

    # annotated_frames = fields.List(AnnotatedFrame, tagname="annotated_frames")

