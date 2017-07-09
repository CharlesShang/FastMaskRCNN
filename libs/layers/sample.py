from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import libs.configs.config_v1 as cfg
import libs.boxes.nms_wrapper as nms_wrapper
import libs.boxes.cython_bbox as cython_bbox
from libs.boxes.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes
from libs.logs.log import LOG

_DEBUG=False

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

  # training: 12000, 2000
  # testing: 6000, 400
  if not is_training:
    pre_nms_top_n = int(pre_nms_top_n / 2)
    post_nms_top_n = int(post_nms_top_n / 5)
    
  boxes = boxes.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  assert scores.shape[0] == boxes.shape[0], 'scores and boxes dont match'
  
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
  order = scores.ravel().argsort()[::-1]
  if pre_nms_top_n > 0:
    order = order[:pre_nms_top_n]
  boxes = boxes[order, :]
  scores = scores[order]

  # filter with nms
  det = np.hstack((boxes, scores)).astype(np.float32)
  keeps = nms_wrapper.nms(det, rpn_nms_threshold)
  
  if post_nms_top_n > 0:
    keeps = keeps[:post_nms_top_n]
  boxes = boxes[keeps, :]
  scores = scores[keeps]
  batch_inds = np.zeros([boxes.shape[0]], dtype=np.int32)

  # # random sample boxes
  ## try early sample later
  # fg_inds = np.where(scores > 0.5)[0]
  # num_fgs = min(len(fg_inds.size), int(rois_per_image * fg_roi_fraction))

  if _DEBUG:
    LOG('SAMPLE: %d rois has been choosen' % len(scores))
    LOG('SAMPLE: a positive box: %d %d %d %d %.4f' % (boxes[0, 0], boxes[0, 1], boxes[0, 2], boxes[0, 3], scores[0]))
    LOG('SAMPLE: a negative box: %d %d %d %d %.4f' % (boxes[-1, 0], boxes[-1, 1], boxes[-1, 2], boxes[-1, 3], scores[-1]))
    hs = boxes[:, 3] - boxes[:, 1]
    ws = boxes[:, 2] - boxes[:, 0]
    assert min(np.min(hs), np.min(ws)) > 0, 'invalid boxes'
  
  return boxes, scores.astype(np.float32), batch_inds

def sample_rpn_outputs_wrt_gt_boxes(boxes, scores, gt_boxes, is_training=False, only_positive=False):
    """sample boxes for refined output"""
    boxes, scores, batch_inds = sample_rpn_outputs(boxes, scores, is_training, only_positive)

    if gt_boxes.size > 0:
        overlaps = cython_bbox.bbox_overlaps(
                np.ascontiguousarray(boxes[:, 0:4], dtype=np.float),
                np.ascontiguousarray(gt_boxes[:, 0:4], dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1) # B
        max_overlaps = overlaps[np.arange(boxes.shape[0]), gt_assignment] # B
        fg_inds = np.where(max_overlaps >= cfg.FLAGS.fg_threshold)[0]
        if _DEBUG and np.argmax(overlaps[fg_inds],axis=1).size < gt_boxes.size/5.0:
            print("gt_size")
            print(gt_boxes)
            gt_height = (gt_boxes[:,2]-gt_boxes[:,0])
            gt_width = (gt_boxes[:,3]-gt_boxes[:,1])
            gt_dim = np.vstack((gt_height, gt_width))
            print(np.transpose(gt_dim))
            #print(gt_height)
            #print(gt_width)

            print('SAMPLE: %d after overlaps by %s' % (len(fg_inds),cfg.FLAGS.fg_threshold))
            print("detected object no.")
            print(np.argmax(overlaps[fg_inds],axis=1))
            print("total object")
            print(gt_boxes.size/5.0)

        mask_fg_inds = np.where(max_overlaps >= cfg.FLAGS.mask_threshold)[0]
        if mask_fg_inds.size > cfg.FLAGS.masks_per_image:
            mask_fg_inds = np.random.choice(mask_fg_inds, size=cfg.FLAGS.masks_per_image, replace=False)

        if True:
            gt_argmax_overlaps = overlaps.argmax(axis=0) # G
            fg_inds = np.union1d(gt_argmax_overlaps, fg_inds)

	fg_rois = int(min(fg_inds.size, cfg.FLAGS.rois_per_image * cfg.FLAGS.fg_roi_fraction))
      	if fg_inds.size > 0 and fg_rois < fg_inds.size:
       	   fg_inds = np.random.choice(fg_inds, size=fg_rois, replace=False)
      	
	# TODO: sampling strategy
      	bg_inds = np.where((max_overlaps < cfg.FLAGS.bg_threshold))[0]
      	bg_rois = max(min(cfg.FLAGS.rois_per_image - fg_rois, fg_rois * 3), 8)#64
      	if bg_inds.size > 0 and bg_rois < bg_inds.size:
           bg_inds = np.random.choice(bg_inds, size=bg_rois, replace=False)

        keep_inds = np.append(fg_inds, bg_inds)
        #print(gt_boxes[np.argmax(overlaps[fg_inds],axis=1),4])
    else:
        bg_inds = np.arange(boxes.shape[0])
        bg_rois = min(int(cfg.FLAGS.rois_per_image * (1-cfg.FLAGS.fg_roi_fraction)), 8)#64
        if bg_rois < bg_inds.size:
            bg_inds = np.random.choice(bg_inds, size=bg_rois, replace=False)

        keep_inds = bg_inds
        mask_fg_inds = np.arange(0)
    
    return boxes[keep_inds, :], scores[keep_inds], batch_inds[keep_inds],\
           boxes[mask_fg_inds, :], scores[mask_fg_inds], batch_inds[mask_fg_inds]

def _jitter_boxes(boxes, jitter=0.1):
    """ jitter the boxes before appending them into rois
    """
    jittered_boxes = boxes.copy()
    ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
    hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
    width_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * ws
    height_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * hs
    jittered_boxes[:, 0] += width_offset
    jittered_boxes[:, 2] += width_offset
    jittered_boxes[:, 1] += height_offset
    jittered_boxes[:, 3] += height_offset

    return jittered_boxes

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
