from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def anchors(scales=[2, 4, 8, 16, 32], ratios=[0.5, 1, 2.0], base=16):
  """Get a set of anchors at one position """
  scales = np.asarray(base * scales)
  areas = scales ** 2
  # enum all anchors
  
  return

def get_anchors(height, width, **kwargs):
  """Get a complete set of anchors in a spatial plane,
  """
  anc = anchors(**kwargs)
  # enum all anchors in a plane
  
  return