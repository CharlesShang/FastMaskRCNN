# --------------------------------------------------------
# Mask RCNN
# Written by CharlesShang@github
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from . import anchor
from . import roi
from . import mask
from . import sample

def anchor_encoder(gt_boxes, all_anchors, height, width, stride, scope='AnchorEncoder'):
  
  with tf.name_scope(scope) as sc:
    labels, rois, bbox_targets, bbox_inside_weights = \
      tf.py_func(anchor.encode,
                 [gt_boxes, all_anchors, height, width, stride],
                 [tf.float32, tf.float32, tf.float32, tf.float32])
    labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')
    rois = tf.convert_to_tensor(rois, name='rois')
    bbox_targets = tf.convert_to_tensor(bbox_targets, name='bbox_targets')
    bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
  
  return labels, rois, bbox_targets, bbox_inside_weights


def anchor_decoder(boxes, scores, all_anchors, ih, iw, scope='AnchorDecoder'):
  
  with tf.name_scope(scope) as sc:
    final_boxes, classes, scores = \
      tf.py_func(anchor.decode,
                 [boxes, scores, all_anchors, ih, iw],
                 [tf.float32, tf.int32, tf.float32])
    final_boxes = tf.convert_to_tensor(final_boxes, name='boxes')
    classes = tf.convert_to_tensor(classes, name='classes')
    scores = tf.convert_to_tensor(scores, name='scores')
  
  return final_boxes, classes, scores


def roi_encoder(gt_boxes, rois, num_classes, scope='ROIEncoder'):
  
  with tf.name_scope(scope) as sc:
    labels, rois, bbox_targets, bbox_inside_weights = \
      tf.py_func(roi.encode,
                [gt_boxes, rois, num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32])
    labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')
    rois = tf.convert_to_tensor(rois, name='rois')
    bbox_targets = tf.convert_to_tensor(bbox_targets, name='bbox_targets')
    bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
  
  return labels, rois, bbox_targets, bbox_inside_weights


def roi_decoder(boxes, scores, rois, ih, iw, scope='ROIDecoder'):
  
  with tf.name_scope(scope) as sc:
    final_boxes, classes, scores = \
      tf.py_func(roi.decode,
                 [boxes, scores, rois],
                 [tf.float32, tf.int32, tf.float32])
    final_boxes = tf.convert_to_tensor(final_boxes, name='boxes')
    classes = tf.convert_to_tensor(classes, name='classes')
    scores = tf.convert_to_tensor(scores, name='scores')
    
  return final_boxes, classes, scores

def mask_encoder(gt_masks, gt_boxes, rois, num_classes, pooled_width, pooled_height, scope='MaskEncoder'):
  
  with tf.name_scope(scope) as sc:
    rois, classes, mask_targets, mask_inside_weights = \
      tf.py_func(mask.encode,
                 [gt_boxes, rois, num_classes],
                 [tf.float32, tf.int32, tf.int32, tf.float32])
    classes = tf.convert_to_tensor(classes, name='classes')
    rois = tf.convert_to_tensor(rois, name='rois')
    mask_targets = tf.convert_to_tensor(mask_targets, name='mask_targets')
    mask_inside_weights = tf.convert_to_tensor(mask_inside_weights, name='mask_inside_weights')
  
  return rois, classes, mask_targets, mask_inside_weights

def mask_decoder(mask_targets, rois, classes, ih, iw, scope='MaskDecoder'):
  
  with tf.name_scope(scope) as sc:
    Mask = \
      tf.py_func(mask.decode,
                 [mask_targets, rois, classes, ih, iw,],
                 [tf.float32])
    final_boxes = tf.convert_to_tensor(Mask, name='MaskImage')
  
  return Mask


def sample_wrapper(boxes, scores, PHASE = 'TEST', scope='SampleBoxes'):
  
  with tf.name_scope(scope) as sc:
    boxes, class_ids, scores = \
      tf.py_func(sample.sample_rpn_outputs,
                 [boxes, scores, PHASE],
                 [tf.float32, tf.int32, tf.float32])
    boxes = tf.convert_to_tensor(boxes, name='Boxes')
    class_ids = tf.convert_to_tensor(tf.cast(class_ids, tf.int32), name='Ids')
    scores = tf.convert_to_tensor(scores, name='Scores')
  
  return boxes, class_ids, scores