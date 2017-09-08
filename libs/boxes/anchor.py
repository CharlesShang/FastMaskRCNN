from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from libs.boxes import cython_anchor
from libs.logs.log import LOG
from libs.boxes import cython_bbox
from libs.boxes.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes

def anchors(scales=[2, 4, 8, 16, 32], ratios=[0.5, 1, 2.0], base=16):
  """Get a set of anchors at one position """
  return generate_anchors(base_size=base, scales=np.asarray(scales, np.int32), ratios=ratios)

def anchors_plane(height, width, stride = 1.0, 
        scales=[2, 4, 8, 16, 32], ratios=[0.5, 1, 2.0], base=16):
  """Get a complete set of anchors in a spatial plane,
  height, width are plane dimensions
  stride is scale ratio of
  """
  # TODO: implement in C, or pre-compute them, or set to a fixed input-shape
  # enum all anchors in a plane
  # scales = kwargs.setdefault('scales', [2, 4, 8, 16, 32])
  # ratios = kwargs.setdefault('ratios', [0.5, 1, 2.0])
  # base = kwargs.setdefault('base', 16)
  anc = anchors(scales, ratios, base)
  all_anchors = cython_anchor.anchors_plane(height, width, stride, anc).astype(np.float32)
  return all_anchors

# Written by Ross Girshick and Sean Bell
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
  """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  """

  base_anchor = np.array([1, 1, base_size, base_size]) - 1
  ratio_anchors = _ratio_enum(base_anchor, ratios)
  anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                       for i in xrange(ratio_anchors.shape[0])])
  return anchors

def _whctrs(anchor):
  """
  Return width, height, x center, and y center for an anchor (window).
  """

  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
  """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """
  
  ws = ws[:, np.newaxis]
  hs = hs[:, np.newaxis]
  anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                       y_ctr - 0.5 * (hs - 1),
                       x_ctr + 0.5 * (ws - 1),
                       y_ctr + 0.5 * (hs - 1)))
  return anchors


def _ratio_enum(anchor, ratios):
  """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  """
  
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  size = w * h
  size_ratios = size / ratios
  ws = (np.sqrt(size_ratios))
  hs = (ws * ratios)#np.round
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


def _scale_enum(anchor, scales):
  """
  Enumerate a set of anchors for each scale wrt an anchor.
  """
  
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  ws = w * scales
  hs = h * scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors

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

def _jitter_gt_boxes(gt_boxes, jitter=0.05):
    """ jitter the gtboxes, before adding them into rois, to be more robust for cls and rgs
    gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
    """
    jittered_boxes = gt_boxes.copy()
    ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
    hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
    width_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * ws
    height_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * hs
    jittered_boxes[:, 0] += width_offset
    jittered_boxes[:, 2] += width_offset
    jittered_boxes[:, 1] += height_offset
    jittered_boxes[:, 3] += height_offset

    return jittered_boxes

if __name__ == '__main__':
  import time
  
  t = time.time()
  total_anchors = 0


  iw = 1134
  ih = 640
  stride = 16
  # all_anchors = anchors_plane(200, 250, stride=4, boarder=0)
  # num_anchors += all_anchors.shape[0]
  # for i in range(10):
  gt_boxes = np.array([ [705.20550537 ,246.37339783,  915.78503418 , 411.53240967]])
  # gt_boxes = np.array([ [476.03378296,  363.47793579,  961.50238037,  559.27886963],
  #              [ 472.08267212,  378.50143433,  814.7980957,   562.92962646], 
  #              [3.15492964,  491.46292114,  957.62628174,  630.52020264]])

  jittered_gt_boxes = _jitter_gt_boxes(gt_boxes[:, :4])
  clipped_gt_boxes = clip_boxes(jittered_gt_boxes, (ih, iw))

  ancs = anchors()
  print("\n%s" % ancs)
  all_anchors = cython_anchor.anchors_plane(40, 71, stride, ancs)
  total_anchors += all_anchors.shape[0] * all_anchors.shape[1] * all_anchors.shape[2]
  print (all_anchors)
  print (all_anchors.shape)
  all_anchors = all_anchors.reshape([-1, 4])
  labels = np.empty((all_anchors.shape[0], ), dtype=np.int32)
  labels.fill(-1)

  overlaps = cython_bbox.bbox_overlaps(
                   np.ascontiguousarray(all_anchors, dtype=np.float),
                   np.ascontiguousarray(clipped_gt_boxes, dtype=np.float))

  gt_assignment = overlaps.argmax(axis=1)  # (A)
  print(gt_assignment)
  max_overlaps = overlaps[np.arange(total_anchors), gt_assignment]
  print(max_overlaps)
  gt_argmax_overlaps = overlaps.argmax(axis=0)  # G
  print(gt_argmax_overlaps)
  gt_max_overlaps = overlaps[gt_argmax_overlaps,
                             np.arange(overlaps.shape[1])]
  print(gt_max_overlaps)

  # bg label: less than threshold IOU
  labels[max_overlaps < 0.3] = 0    
  # fg label: above threshold IOU 
  labels[max_overlaps >= 0.7] = 1

  # ignore cross-boundary anchors
  cb0_inds = np.where(all_anchors[:, 0] <= 0  - (all_anchors[:, 2] - all_anchors[:, 0]) * 0)
  cb1_inds = np.where(all_anchors[:, 1] <= 0  - (all_anchors[:, 3] - all_anchors[:, 1]) * 0)
  cb2_inds = np.where(all_anchors[:, 2] >= iw + (all_anchors[:, 2] - all_anchors[:, 0]) * 0)
  cb3_inds = np.where(all_anchors[:, 3] >= ih + (all_anchors[:, 3] - all_anchors[:, 1]) * 0)
  cb_inds = np.unique(np.concatenate((cb0_inds, cb1_inds, cb2_inds, cb3_inds), axis =1))
  labels[cb_inds] = -2
  #LOG ("stride: %d total anchor: %d\tremained anchor: %d\t ih:%d iw:%d min size %d %d \t max size %d %d" % (stride, total_anchors, total_anchors-len(cb_inds), ih, iw, np.min(all_anchors[:, 0]), np.min(all_anchors[:, 1]), np.max(all_anchors[:, 2]), np.max(all_anchors[:, 3])))
  print ("stride: %d total anchor: %d\tremained anchor: %d\t ih:%d iw:%d min size %d %d \t max size %d %d" % (stride, total_anchors, total_anchors-len(cb_inds), ih, iw, np.min(all_anchors[labels!=-2, 0]), np.min(all_anchors[labels!=-2, 1]), np.max(all_anchors[labels!=-2, 2]), np.max(all_anchors[labels!=-2, 3])))

  labels[gt_argmax_overlaps] = 2

  print ("above threshold: %s closest box: %s"% ((np.where(labels==1)), (np.where(labels==2))))
  print ("all_anchors anchor\n%s" %all_anchors[labels==2, :])
  print ("gt anchor\n%s" %gt_boxes)
          
  # all_anchors = cython_anchor.anchors_plane(20, 30, 16, ancs)
  # num_anchors += all_anchors.shape[0] * all_anchors.shape[1] * all_anchors.shape[2]

  # all_anchors = cython_anchor.anchors_plane(40, 60, 8, ancs)
  # num_anchors += all_anchors.shape[0] * all_anchors.shape[1] * all_anchors.shape[2]

  # all_anchors = cython_anchor.anchors_plane(80, 120, 4, ancs)
  # num_anchors += all_anchors.shape[0] * all_anchors.shape[1] * all_anchors.shape[2]

  # print('average time: %f' % ((time.time() - t) / 10))
  # print('anchors: %d' % (num_anchors / 10))
  # print(a.shape, '\n', a)
  # print (all_anchors.shape)
  # from IPython import embed
  # embed()


