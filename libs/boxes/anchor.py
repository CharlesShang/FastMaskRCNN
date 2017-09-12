from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from libs.boxes import cython_anchor

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
  all_anchors = cython_anchor.anchors_plane(height, width, stride, anc)
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
  ws = np.round(np.sqrt(size_ratios))
  hs = np.round(ws * ratios)
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

if __name__ == '__main__':
  import time
  
  t = time.time()
  a = anchors()
  num_anchors = 0

  # all_anchors = anchors_plane(200, 250, stride=4, boarder=0)
  # num_anchors += all_anchors.shape[0]
  for i in range(10):
    ancs = anchors()
    all_anchors = cython_anchor.anchors_plane(200, 250, 4, ancs)
    num_anchors += all_anchors.shape[0] * all_anchors.shape[1] * all_anchors.shape[2]
    all_anchors = cython_anchor.anchors_plane(100, 125, 8, ancs)
    num_anchors += all_anchors.shape[0] * all_anchors.shape[1] * all_anchors.shape[2]
    all_anchors = cython_anchor.anchors_plane(50, 63, 16, ancs)
    num_anchors += all_anchors.shape[0] * all_anchors.shape[1] * all_anchors.shape[2]
    all_anchors = cython_anchor.anchors_plane(25, 32, 32, ancs)
    num_anchors += all_anchors.shape[0] * all_anchors.shape[1] * all_anchors.shape[2]
  print('average time: %f' % ((time.time() - t) / 10))
  print('anchors: %d' % (num_anchors / 10))
  print(a.shape, '\n', a)
  print (all_anchors.shape)
  # from IPython import embed
  # embed()
