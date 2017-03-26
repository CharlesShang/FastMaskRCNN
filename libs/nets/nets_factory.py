from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from .resnet_v1 import resnet_v1_50 as resnet50
from .resnet_utils import resnet_arg_scope
from .resnet_v1 import resnet_v1_101 as resnet101

slim = tf.contrib.slim

networks_map = {'resnet50': resnet50,
                'resnet101': resnet101,
               }

arg_scopes_map = {'resnet50': resnet_arg_scope,
                  'resnet101': resnet_arg_scope,
                 }

def get_network_fn(name, num_classes, weight_decay=0.00005, is_training=False):
  """ Load the pretrained model graph,
   Set is_training to False to disable BN update
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)

  arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
  func = networks_map[name]

  @functools.wraps(func)
  def network_fn(images):
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn