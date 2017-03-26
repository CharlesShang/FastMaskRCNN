from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim

# mapping each stage to its' tensor features
networks_map = {
  'resnet50': {'C1':'', 'C2':'',
               'C3':'', 'C4':'',
               'C5':'',
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

def build_pyramid(name, bilinear=True):
  """build pyramid features from a typical network,
  assume each stage is 2 time larger than its top feature
  Returns:
    returns several endpoints
  """
  end_points = {}
  Cs = networks_map[name]

  arg_scope = extra_conv_arg_scope()
  with slim.arg_scope(arg_scope):
    end_points['P5'] = \
      slim.conv2d(Cs['C5'])
    
    for c in range(5, 2, -1):
      s, s_ = Cs['C%d'%c], Cs['C%d'%(c-1)]
      
      s = slim.conv2d(s)
      s = tf.image.resize_bilinear(s)
      s_ = slim.conv2d(s_)
      s = tf.concat([s, s_])
      end_points['P%d'%(c-1)] = s
    
    return end_points
  
def build_output(end_points):
  """Build the 3-way output, class, box and mask
  """
  outputs = {}
  arg_scope = extra_conv_arg_scope(activation_fn=tf.nn.relu)
  with slim.arg_scope(arg_scope):
    for p in end_points:
      outputs[p] = {}
      end_points[p]
      # rpn -> conv
      outputs[p]['rpn'] = {'box':'', 'class':''}
      # roi align and convs
      outputs[p]['refine'] = {'box': '', 'class': ''}
      # add a mask, given the predicted boxes and classes
      outputs[p]['mask'] = {'mask':''}
      
  return outputs
