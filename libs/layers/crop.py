from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def crop(images, boxes, batch_inds = False, stride = 1, pooled_height = 7, pooled_width = 7, scope='ROIAlign'):
  """Cropping areas of features into fixed size"""
  with tf.name_scope(scope):
    boxes = boxes / (stride + 0.0)
    shape = tf.shape(images)
    boxes = tf.reshape(boxes, [-1, 2]) # to (x, y)
    x = tf.slice(boxes, [0, 0], [-1, 1])
    y = tf.slice(boxes, [0, 1], [-1, 1])
    x = x / tf.cast(shape[2], tf.float32)
    y = y / tf.cast(shape[1], tf.float32)
    boxes = tf.concat([y, x], axis=1)
    boxes = tf.reshape(boxes, [-1, 4])  # to (y1, x1, y2, x2)
    
    if batch_inds is False:
      shape = tf.shape(boxes)
      # batch_inds = tf.zeros((shape[0], ), dtype=tf.int32, name='batch_inds')
      batch_inds = tf.zeros([shape[0]], dtype=tf.int32, name='batch_inds')
    return  tf.image.crop_and_resize(images, boxes, batch_inds,
                                     [pooled_height, pooled_width],
                                     method='bilinear',
                                     name='Crop')