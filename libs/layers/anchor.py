from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import libs.boxes.cython_bbox as cython_bbox
import libs.configs.config_v1 as cfg
from libs.boxes.bbox_transform import bbox_transform, bbox_transform_inv
from libs.boxes.anchor import anchors_plane
# FLAGS = tf.app.flags.FLAGS

def encode(gt_boxes, anchors, height, width, stride):
  """Matching and Encoding groundtruth into learning targets
  Sampling
  
  Parameters
  ---------
  gt_boxes: an array of shape (G x 4), [x1, y1, x2, y2]
  anchors: an array of shape (G x 1), each value is in [0, num_classes]
  width: width of feature
  height: height of feature
  stride: downscale factor w.r.t the input size [4, 8, 16, 32]
  Returns
  --------
  labels:   Nx1 array in [0, num_classes]
  anchors:  Sampled anchors
  bbox_targets: N x (4) regression targets
  bbox_inside_weights: N x (4), in {0, 1} indicating to which class is assigned.
  """
  # TODO: speedup this module
  if anchors is None:
    anchors, inds_inside, total_anchors = anchors_plane(height, width, stride=stride, boarder=0)
  
  #
  areas = (gt_boxes[:, 3] - gt_boxes[:, 1] + 1) * (gt_boxes[:, 2] - gt_boxes[:, 0] + 1)
  ks = np.floor(4 + np.log2(np.sqrt(areas) / 224.0))
  K = int(np.log2(stride))
  inds = np.where((K == ks + 4))[0]
  if inds.size > 0:
    gt_boxes = gt_boxes[inds]
  else:
    labels = np.empty((total_anchors), dtype=np.float32)
    bbox_targets = np.zeros((total_anchors, 4), dtype=np.float32)
    bbox_inside_weights = np.zeros((total_anchors, 4), dtype=np.float32)
    return labels, bbox_targets, bbox_inside_weights

  labels = np.empty((anchors.shape[0]), dtype=np.float32)
  overlaps = cython_bbox.bbox_overlaps(
    np.ascontiguousarray(anchors, dtype=np.float),
    np.ascontiguousarray(gt_boxes, dtype=np.float))

  argmax_overlaps = overlaps.argmax(axis=1)  # (A)
  max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
  gt_argmax_overlaps = overlaps.argmax(axis=0)  # G
  gt_max_overlaps = overlaps[gt_argmax_overlaps,
                             np.arange(overlaps.shape[1])]
  gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

  # fg label: for each gt, anchor with highest overlap
  labels[gt_argmax_overlaps] = 1
  # fg label: above threshold IOU
  labels[max_overlaps >= cfg.FLAGS.fg_threshold] = 1

  # subsample positive labels if there are too many
  num_fg = int(cfg.FLAGS.fg_rpn_fraction * cfg.FLAGS.rpn_batch_size)
  fg_inds = np.where(labels == 1)[0]
  if len(fg_inds) > num_fg:
    disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    labels[disable_inds] = -1

  # subsample negative labels if there are too many
  num_bg = cfg.FLAGS.rpn_batch_size - np.sum(labels == 1)
  bg_inds = np.where(labels == 0)[0]
  if len(bg_inds) > num_bg:
    disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    labels[disable_inds] = -1

  bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
  bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
  bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  bbox_inside_weights[labels == 1, :] = 1

  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
  bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)

  labels = labels.reshape((1, height, width, -1))
  bbox_targets = bbox_targets.reshape((1, height, width, -1))
  bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, -1))

  return labels, bbox_targets, bbox_inside_weights

def decode(outputs):
  """Decode outputs into boxes"""
  return

def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret
  
def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 5

  return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

if __name__ == '__main__':
  anchors, _, _ = anchors_plane(200, 250, stride=4, boarder=0)
  import time
  t = time.time()
  
  for i in range(10):
    cfg.FLAGS.fg_threshold = 0.1
    classes = np.random.randint(0, 3, (50, 1))
    boxes = np.random.randint(10, 50, (50, 2))
    s = np.random.randint(0, 50, (50, 2))
    s = boxes + s
    boxes = np.concatenate((boxes, s), axis=1)
    gt_boxes = np.hstack((boxes, classes))
    # gt_boxes = boxes
    rois = np.random.randint(10, 50, (20, 2))
    s = np.random.randint(0, 20, (20, 2))
    s = rois + s
    rois = np.concatenate((rois, s), axis=1)
    labels, bbox_targets, bbox_inside_weights = encode(gt_boxes, anchors=None, height=200, width=250, stride=4)
    # labels, bbox_targets, bbox_inside_weights = encode(gt_boxes, anchors=None, height=100, width=150, stride=8)
    # labels, bbox_targets, bbox_inside_weights = encode(gt_boxes, anchors=None, height=50, width=75, stride=16)
    # labels, bbox_targets, bbox_inside_weights = encode(gt_boxes, anchors=None, height=25, width=37, stride=32)
  print(labels)
  print('average time: %f' % ((time.time() - t)/10.0))
  # print(bbox_inside_weights)
  
  # ls = np.zeros((labels.shape[0], 3))
  # for i in range(labels.shape[0]):
  #   ls[i, labels[i]] = 1
  # final_boxes = decode(bbox_targets, ls, rois)
  # print(np.hstack((final_boxes, np.expand_dims(labels, axis=1))))
  # print(gt_boxes)