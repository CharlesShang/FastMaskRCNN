from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def crop(images, boxes, batch_inds, image_height, image_width, stride = 1, pooled_height = 7, pooled_width = 7, scope='ROIAlign'):
  """Cropping areas of features into fixed size
  Params:
  --------
  images: a 4-d Tensor of shape (N, H, W, C)
  boxes: rois in the original image, of shape (N, ..., 4), [x1, y1, x2, y2]
  batch_inds: 

  Returns:
  --------
  A Tensor of shape (N, pooled_height, pooled_width, C)
  """
  with tf.name_scope(scope):
    #
    boxes = tf.reshape(boxes, [-1, 4])

    # normalize the boxes and swap x y dimensions
    shape = tf.shape(images)
    boxes = tf.reshape(boxes, [-1, 2]) # to (x, y)

    xs = boxes[:, 0] 
    ys = boxes[:, 1]
    xs = xs / tf.cast(image_width, tf.float32)
    ys = ys / tf.cast(image_height, tf.float32)
    boxes = tf.concat([ys[:, tf.newaxis], xs[:, tf.newaxis]], axis=1)
    boxes = tf.reshape(boxes, [-1, 4])  # to (y1, x1, y2, x2)
    
    # if batch_inds is False:
    #   num_boxes = tf.shape(boxes)[0]
    #   batch_inds = tf.zeros([num_boxes], dtype=tf.int32, name='batch_inds')
    # batch_inds = boxes[:, 0] * 0
    # batch_inds = tf.cast(batch_inds, tf.int32)

    # assert_op = tf.Assert(tf.greater(tf.shape(images)[0], tf.reduce_max(batch_inds)), [images, batch_inds])
    assert_op = tf.Assert(tf.greater(tf.size(images), 0), [images, batch_inds])
    with tf.control_dependencies([assert_op, images, batch_inds]):
        return  [tf.image.crop_and_resize(images, boxes, batch_inds,
                                         [pooled_height, pooled_width],
                                         method='bilinear',
                                         name='Crop')] + [boxes]

