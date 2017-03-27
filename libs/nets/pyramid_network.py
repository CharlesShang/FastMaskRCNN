from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim
from .roi_utils import roi_cropping
# mapping each stage to its' tensor features
networks_map = {
  'resnet50': {'C1':'resnet_v1_50/conv1/Relu:0',
               'C2':'resnet_v1_50/block1/unit_2/bottleneck_v1/Relu:0',
               'C3':'resnet_v1_50/block2/unit_3/bottleneck_v1/Relu:0',
               'C4':'resnet_v1_50/block3/unit_5/bottleneck_v1/Relu:0',
               'C5':'resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0',
               },
  'resnet101': {'C1': '', 'C2': '',
                'C3': '', 'C4': '',
                'C5': '',
               }
}

def extra_conv_arg_scope(weight_decay=0.00005, activation_fn=None, normalizer_fn=None):

  with slim.arg_scope(
      [slim.conv2d, slim.conv2d_transpose],
      padding='SAME',
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=None,
      normalizer_fn=None,) as arg_sc:
        return arg_sc

def build_pyramid(name, end_points, bilinear=True):
  """build pyramid features from a typical network,
  assume each stage is 2 time larger than its top feature
  Returns:
    returns several endpoints
  """
  pyramid = {}
  Cs = networks_map[name]

  arg_scope = extra_conv_arg_scope()
  with tf.name_scope('extra'):
    with slim.arg_scope(arg_scope):
      
      pyramid['P5'] = \
        slim.conv2d(end_points[Cs['C5']], 256, [1, 1], stride=1, scope='C5')
      
      for c in range(4, 1, -1):
        s, s_ = pyramid['P%d'%(c+1)], end_points[Cs['C%d'%(c)]]

        s_ = slim.conv2d(s_, 256, [3, 3], stride=1, scope='C%d'%c)
        
        up_shape = s_.get_shape()
        out_shape = tf.stack(up_shape[1].value, up_shape[2].value)
        # s = slim.conv2d(s, 256, [3, 3], stride=1, scope='C%d'%c)
        s = tf.image.resize_bilinear(s, out_shape, name='C%d/upscale'%c)
        
        s = tf.add(s, s_, name='C%d/addition'%c)
        pyramid['P%d'%(c)] = s
      
      return pyramid
  
def build_head(pyramid, num_classes, num_anchors):
  """Build the 3-way output, class, box and mask
  First, rpn->
  """
  outputs = {}
  arg_scope = extra_conv_arg_scope(activation_fn=None)
  
  with slim.arg_scope(arg_scope):
    # for p in pyramid:
    for i in range(5, 1, -1):
      p = 'P%d'%i
      outputs[p] = {}
      
      rpn = slim.conv2d(pyramid[p], 256, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='%s/rpn'%p)
      box = slim.conv2d(rpn, num_classes * num_anchors * 4, [1, 1], stride=1, scope='%s/rpn/box'%p)
      cls = slim.conv2d(rpn, num_classes * num_anchors * 2, [1, 1], stride=1, scope='%s/rpn/cls'%p)
      # rpn -> conv
      outputs[p]['rpn'] = {'box':box, 'class':cls}
      roi_cropping(pyramid[p], box, cls, anchors, spatial_scale=1.0/(2**i))
      # roi align and convs
      outputs[p]['refine'] = {'box': '', 'class': ''}
      # add a mask, given the predicted boxes and classes
      outputs[p]['mask'] = {'mask':''}
      
  return outputs
