from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def roi_align(feat, boxes):
  """Given features and boxes, This function crops feature """
  return

def roi_cropping(feat, boxes, clses, anchors, spatial_scale=1.0/16):
  """This function computes final rpn boxes
   And crops areas from the incoming features
  """
  return