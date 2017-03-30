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
from libs.boxes.anchor import anchors_plane

def anchor_encoder(gt_boxes, all_anchors, height, width, stride, scope='AnchorEncoder'):
  
  with tf.name_scope(scope) as sc:
    labels, bbox_targets, bbox_inside_weights = \
      tf.py_func(anchor.encode,
                 [gt_boxes, all_anchors, height, width, stride],
                 [tf.float32, tf.float32, tf.float32, tf.float32])
    labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')
    bbox_targets = tf.convert_to_tensor(bbox_targets, name='bbox_targets')
    bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
    labels = tf.reshape(labels, (1, height, width, -1))
    bbox_targets = tf.reshape(bbox_targets, (1, height, width, -1))
    bbox_inside_weights = tf.reshape(bbox_inside_weights, (1, height, width, -1))
  
  return labels, bbox_targets, bbox_inside_weights


def anchor_decoder(boxes, scores, all_anchors, ih, iw, scope='AnchorDecoder'):
  
  with tf.name_scope(scope) as sc:
    final_boxes, classes, scores = \
      tf.py_func(anchor.decode,
                 [boxes, scores, all_anchors, ih, iw],
                 [tf.float32, tf.int32, tf.float32])
    final_boxes = tf.convert_to_tensor(final_boxes, name='boxes')
    classes = tf.convert_to_tensor(tf.cast(classes, tf.int32), name='classes')
    scores = tf.convert_to_tensor(scores, name='scores')
    final_boxes = tf.reshape(final_boxes, (-1, 4))
    classes = tf.reshape(classes, (-1, ))
    scores = tf.reshape(scores, (-1, ))
  
  return final_boxes, classes, scores


def roi_encoder(gt_boxes, rois, num_classes, scope='ROIEncoder'):
  
  with tf.name_scope(scope) as sc:
    labels, bbox_targets, bbox_inside_weights = \
      tf.py_func(roi.encode,
                [gt_boxes, rois, num_classes],
                [tf.float32, tf.float32, tf.float32])
    labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')
    bbox_targets = tf.convert_to_tensor(bbox_targets, name='bbox_targets')
    bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
    labels = tf.reshape(labels, (-1, ))
    bbox_targets = tf.reshape(bbox_targets, (-1, num_classes * 4))
    bbox_inside_weights = tf.reshape(bbox_inside_weights, (-1, num_classes * 4))
  
  return labels, rois, bbox_targets, bbox_inside_weights


def roi_decoder(boxes, scores, rois, ih, iw, scope='ROIDecoder'):
  
  with tf.name_scope(scope) as sc:
    final_boxes, classes, scores = \
      tf.py_func(roi.decode,
                 [boxes, scores, rois, ih, iw],
                 [tf.float32, tf.int32, tf.float32])
    final_boxes = tf.convert_to_tensor(final_boxes, name='boxes')
    classes = tf.convert_to_tensor(tf.cast(classes, tf.int32), name='classes')
    scores = tf.convert_to_tensor(scores, name='scores')
    final_boxes = tf.reshape(final_boxes, (-1, 4))
    
  return final_boxes, classes, scores

def mask_encoder(gt_masks, gt_boxes, rois, num_classes, mask_height, mask_width, scope='MaskEncoder'):
  
  with tf.name_scope(scope) as sc:
    labels, mask_targets, mask_inside_weights = \
      tf.py_func(mask.encode,
                 [gt_masks, gt_boxes, rois, num_classes, mask_height, mask_width],
                 [tf.float32, tf.int32, tf.float32])
    labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='classes')
    mask_targets = tf.convert_to_tensor(mask_targets, name='mask_targets')
    mask_inside_weights = tf.convert_to_tensor(mask_inside_weights, name='mask_inside_weights')
    labels = tf.reshape(labels, (-1,))
    mask_targets = tf.reshape(mask_targets, (-1, mask_height, mask_width, num_classes))
    mask_inside_weights = tf.reshape(mask_inside_weights, (-1, mask_height, mask_width, num_classes))
  
  return labels, mask_targets, mask_inside_weights

def mask_decoder(mask_targets, rois, classes, ih, iw, scope='MaskDecoder'):
  
  with tf.name_scope(scope) as sc:
    Mask = \
      tf.py_func(mask.decode,
                 [mask_targets, rois, classes, ih, iw,],
                 [tf.float32])
    Mask = tf.convert_to_tensor(Mask, name='MaskImage')
    Mask = tf.reshape(Mask, (ih, iw))
  
  return Mask


def sample_wrapper(boxes, scores, is_training=False, scope='SampleBoxes'):
  
  with tf.name_scope(scope) as sc:
    boxes, class_ids, scores = \
      tf.py_func(sample.sample_rpn_outputs,
                 [boxes, scores, is_training],
                 [tf.float32, tf.int32, tf.float32])
    boxes = tf.convert_to_tensor(boxes, name='Boxes')
    class_ids = tf.convert_to_tensor(tf.cast(class_ids, tf.int32), name='Ids')
    scores = tf.convert_to_tensor(scores, name='Scores')
    boxes = tf.reshape(boxes, (-1, 4))
  
  return boxes, class_ids, scores

def gen_all_anchors(height, width, stride, scope='GenAnchors'):
  
  with tf.name_scope(scope) as sc:
    all_anchors = \
      tf.py_func(anchors_plane,
                 [height, width, stride],
                 [tf.float32]
                 )
    all_anchors = tf.convert_to_tensor(all_anchors, name='AllAnchors')
    all_anchors = tf.reshape(all_anchors, (height, width, -1))
    
    return all_anchors
    