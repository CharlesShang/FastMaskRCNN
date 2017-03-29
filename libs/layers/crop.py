from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def crop(images, boxes, batch_inds = False, stride = 1, pooled_height = 7, pooled_width = 7, scope='ROIAlign'):
  """Cropping areas of features into fixed size"""
  with tf.name_scope(scope):
    boxes = tf.reshape(boxes, [-1, 2]) # to (x, y)
    boxes = tf.reverse(boxes, [-1]) # to (y, x)
    boxes = tf.reshape(boxes, [-1, 4])  # to (y1, x1, y2, x2)
    
    boxes = boxes / (stride + 0.0)
    if batch_inds is False:
      shape = boxes.get_shape()
      # shape = tf.shape(boxes)
      # bind_shape = tf.stack((shape[0]))
      # batch_inds = tf.zeros(shape[0], dtype=tf.int32, name='batch_inds')
      zeros = tf.zeros_like(boxes, dtype=tf.int32)
      batch_inds = tf.slice(zeros, [0, 0], [-1, 1])
      batch_inds = tf.reshape(batch_inds, [-1])
    return  tf.image.crop_and_resize(images, boxes, batch_inds,
                                     [pooled_height, pooled_width],
                                     method='bilinear',
                                     name='Crop')