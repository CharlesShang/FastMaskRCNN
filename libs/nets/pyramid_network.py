from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.boxes.roi import roi_cropping
from libs.layers import anchor_encoder
from libs.layers import anchor_decoder
from libs.layers import roi_encoder
from libs.layers import roi_decoder
from libs.layers import mask_encoder
from libs.layers import mask_decoder
from libs.layers import ROIAlign
from libs.layers import sample_rpn_outputs

# mapping each stage to its' tensor features
_networks_map = {
  'resnet50': {'C1':'resnet_v1_50/conv1/Relu:0',
               'C2':'resnet_v1_50/block1/unit_2/bottleneck_v1',
               'C3':'resnet_v1_50/block2/unit_3/bottleneck_v1',
               'C4':'resnet_v1_50/block3/unit_5/bottleneck_v1',
               'C5':'resnet_v1_50/block4/unit_3/bottleneck_v1',
               },
  'resnet101': {'C1': '', 'C2': '',
                'C3': '', 'C4': '',
                'C5': '',
               }
}

def _extra_conv_arg_scope(weight_decay=0.00005, activation_fn=None, normalizer_fn=None):

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
  convs_map = _networks_map[name]

  arg_scope = _extra_conv_arg_scope()
  with tf.name_scope('pyramid'):
    with slim.arg_scope(arg_scope):
      
      pyramid['P5'] = \
        slim.conv2d(end_points[convs_map['C5']], 256, [1, 1], stride=1, scope='C5')
      
      for c in range(4, 1, -1):
        s, s_ = pyramid['P%d'%(c+1)], end_points[convs_map['C%d' % (c)]]

        s_ = slim.conv2d(s_, 256, [3, 3], stride=1, scope='C%d'%c)
        
        up_shape = s_.get_shape()
        out_shape = tf.stack((up_shape[1], up_shape[2]))
        # s = slim.conv2d(s, 256, [3, 3], stride=1, scope='C%d'%c)
        s = tf.image.resize_bilinear(s, out_shape, name='C%d/upscale'%c)
        
        s = tf.add(s, s_, name='C%d/addition'%c)
        pyramid['P%d'%(c)] = s
      
      return pyramid
  
def build_head(pyramid, num_classes, num_anchors):
  """Build the 3-way outputs, i.e., class, box and mask in the pyramid
  Algo
  ----
  For each layer:
    1. Build anchor layer
    2. Process the results of anchor layer
    3. Build roi layer
    4. Process the results of roi layer
    5. Build the mask layer
    6. Build losses
  """
  outputs = {}
  arg_scope = _extra_conv_arg_scope(activation_fn=None)
  
  with slim.arg_scope(arg_scope):
    # for p in pyramid:
    for i in range(5, 1, -1):
      p = 'P%d'%i
      outputs[p] = {}
      
      # rpn head
      rpn = slim.conv2d(pyramid[p], 256, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='%s/rpn'%p)
      box = slim.conv2d(rpn, num_classes * num_anchors * 4, [1, 1], stride=1, scope='%s/rpn/box'%p)
      cls = slim.conv2d(rpn, num_classes * num_anchors * 2, [1, 1], stride=1, scope='%s/rpn/cls'%p)
      outputs[p]['rpn'] = {'box': box, 'class': cls}
      # sample and crop
      rois, classes, scores = \
                anchor_decoder(box, cls, all_anchors=None, ih=224, iw=224)
      rois, class_ids, scores = sample_rpn_outputs(rois, scores)
      cropped = ROIAlign(pyramid[p], rois, None, stride=2**i,
                         pooled_height=7, pooled_width=7,)
      
      # refine head
      
      
      roi_cropping(pyramid[p], box, cls, anchors, spatial_scale=1.0/(2**i))
      # roi align and convs
      outputs[p]['refine'] = {'box': '', 'class': ''}
      # add a mask, given the predicted boxes and classes
      outputs[p]['mask'] = {'mask':''}
      
  return outputs
