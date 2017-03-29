from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def crop(images, boxes, batch_inds = None, stride = 1, pooled_height = 7, pooled_width = 7, scope='ROIAlign'):
  """Cropping areas of features into fixed size"""
  with tf.name_scope(scope):
    boxes = tf.reshape(boxes, [-1, 2]) # to (x, y)
    boxes = tf.reverse(boxes, [-1]) # to (y, x)
    boxes = tf.reshape(boxes, [-1, 4])  # to (y1, x1, y2, x2)
    
    boxes = boxes / (stride + 0.0)
    if batch_inds == None:
      shape = boxes.get_shape()
      batch_inds = tf.constant(0, tf.int32, (shape[0]))
    return  tf.image.crop_and_resize(images, boxes, batch_inds,
                                     [pooled_height, pooled_width],
                                     method='bilinear',
                                     name='Crop')