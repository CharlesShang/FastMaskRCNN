from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import libs.configs.config_v1 as cfg
import libs.boxes.nms_wrapper as nms_wrapper
from libs.logs.log import LOG


def sample_rpn_outputs(boxes, scores, is_training=False, only_positive=False):
  """Sample boxes according to scores and some learning strategies
  assuming the first class is background
  Params:
  boxes: of shape (..., Ax4), each entry is [x1, y1, x2, y2], the last axis has k*4 dims
  scores: of shape (..., A), probs of fg, in [0, 1]
  """
  min_size = cfg.FLAGS.min_size
  rpn_nms_threshold = cfg.FLAGS.rpn_nms_threshold
  pre_nms_top_n = cfg.FLAGS.pre_nms_top_n
  post_nms_top_n = cfg.FLAGS.post_nms_top_n
  if not is_training:
    pre_nms_top_n = int(pre_nms_top_n / 2)
    post_nms_top_n = int(post_nms_top_n / 2)
    
  boxes = boxes.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  
  # filter backgrounds
  # Hope this will filter most of background anchors, since a argsort is too slow..
  if only_positive:
    keeps = np.where(scores > 0.5)[0]
    boxes = boxes[keeps, :]
    scores = scores[keeps]
  
  # filter minimum size
  keeps = _filter_boxes(boxes, min_size=min_size)
  boxes = boxes[keeps, :]
  scores = scores[keeps]
  
  # filter with scores
  order = scores.ravel().argsort()
  if cfg.FLAGS.pre_nms_top_n > 0:
    order = order[:pre_nms_top_n]
  boxes = boxes[order, :]
  scores = scores[order]

  # filter with nms
  det = np.hstack((boxes, scores)).astype(np.float32)
  keeps = nms_wrapper.nms(det, rpn_nms_threshold)
  
  if cfg.FLAGS.post_nms_top_n > 0:
    keeps = keeps[:post_nms_top_n]
  boxes = boxes[keeps, :]
  scores = scores[keeps]
  LOG('%d rois has been choosen' % len(keeps))
  
  return boxes, scores

def _filter_boxes(boxes, min_size):
  """Remove all boxes with any side smaller than min_size."""
  ws = boxes[:, 2] - boxes[:, 0] + 1
  hs = boxes[:, 3] - boxes[:, 1] + 1
  keep = np.where((ws >= min_size) & (hs >= min_size))[0]
  return keep

def _apply_nms(boxes, scores, threshold = 0.5):
  """After this only positive boxes are left
  Applying this class-wise
  """
  num_class = scores.shape[1]
  assert boxes.shape[0] == scores.shape[0], \
    'Shape dismatch {} vs {}'.format(boxes.shape, scores.shape)
  
  final_boxes = []
  final_scores = []
  for cls in np.arange(1, num_class):
    cls_boxes = boxes[:, 4*cls: 4*cls+4]
    cls_scores = scores[:, cls]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))
    keep = nms_wrapper.nms(dets, thresh=0.3)
    dets = dets[keep, :]
    dets = dets[np.where(dets[:, 4] > threshold)]
    final_boxes.append(dets[:, :4])
    final_scores.append(dets[:, 4])
  
  final_boxes = np.vstack(final_boxes)
  final_scores = np.vstack(final_scores)
  
  return final_boxes, final_scores

if __name__ == '__main__':
  import time
  t = time.time()
  
  for i in range(10):
    N = 200000
    boxes = np.random.randint(0, 50, (N, 2))
    s = np.random.randint(10, 40, (N, 2))
    s = boxes + s
    boxes = np.hstack((boxes, s))
    
    scores = np.random.rand(N, 1)
    # scores_ = 1 - np.random.rand(N, 1)
    # scores = np.hstack((scores, scores_))
  
    boxes, scores = sample_rpn_outputs(boxes, scores, only_positive=False)
  
  print ('average time %f' % ((time.time() - t) / 10))
