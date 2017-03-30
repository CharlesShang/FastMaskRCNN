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
from libs.layers import gen_all_anchors
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
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,) as arg_sc:
    with slim.arg_scope(
      [slim.fully_connected],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          weights_initializer=slim.variance_scaling_initializer(),
          activation_fn=activation_fn,
          normalizer_fn=normalizer_fn) as arg_sc:
        return arg_sc

def _smooth_l1_dist(x, y, sigma2=9.0, name='smooth_l1_dist'):
  """Smooth L1 loss
  Returns
  ------
  dist: element-wise distance, as the same shape of x, y
  """
  deltas = x - y
  with tf.name_scope(name=name) as scope:
    deltas_abs = tf.abs(deltas)
    smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
    return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
           (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

def build_pyramid(net_name, end_points, bilinear=True):
  """build pyramid features from a typical network,
  assume each stage is 2 time larger than its top feature
  Returns:
    returns several endpoints
  """
  pyramid = {}
  convs_map = _networks_map[net_name]
  pyramid['inputs'] = end_points['inputs']
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
  
def build_heads(pyramid, num_classes, base_anchors, is_training=False):
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
  inshape = tf.shape(pyramid['inputs'])
  ih, iw = inshape[1], inshape[2]
  arg_scope = _extra_conv_arg_scope(activation_fn=None)
  with slim.arg_scope(arg_scope):
    # for p in pyramid:
    for i in range(5, 1, -1):
      p = 'P%d'%i
      stride = 2 ** i
      outputs[p] = {}
      
      # rpn head
      shape = tf.shape(pyramid[p])
      height, width = shape[1], shape[2]
      rpn = slim.conv2d(pyramid[p], 256, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='%s/rpn'%p)
      box = slim.conv2d(rpn, num_classes * base_anchors * 4, [1, 1], stride=1, scope='%s/rpn/box' % p)
      cls = slim.conv2d(rpn, num_classes * base_anchors * 2, [1, 1], stride=1, scope='%s/rpn/cls' % p)
      outputs[p]['rpn'] = {'box': box, 'classes': cls}
      
      # decode, sample and crop
      all_anchors = gen_all_anchors(height, width, stride)
      rois, classes, scores = \
                anchor_decoder(box, cls, all_anchors, ih, iw)
      rois, class_ids, scores = sample_rpn_outputs(rois, scores)
      cropped = ROIAlign(pyramid[p], rois, False, stride=2**i,
                         pooled_height=7, pooled_width=7,)
      
      # refine head
      refine = slim.fully_connected(cropped, 1024, activation_fn=tf.nn.relu)
      refine = slim.dropout(refine, keep_prob=0.75, is_training=is_training)
      refine = slim.fully_connected(refine,  1024, activation_fn=tf.nn.relu)
      refine = slim.dropout(refine, keep_prob=0.75, is_training=is_training)
      cls2 = slim.fully_connected(refine, num_classes, activation_fn=None)
      box = slim.fully_connected(refine, num_classes*4, activation_fn=None)
      outputs[p]['refined'] = {'box': box, 'classes': cls2}
      
      # decode refine net outputs
      final_boxes, classes, scores = \
              roi_decoder(box, cls2, rois, ih, iw)
      
      # for testing, maskrcnn takes refined boxes as inputs
      if not is_training:
        rois = final_boxes
      
      # mask head
      # rois, class_ids, scores = sample_rpn_outputs(rois, scores)
      m = ROIAlign(pyramid[p], rois, False, stride=2 ** i,
                   pooled_height=14, pooled_width=14)
      for i in range(4):
        m = slim.conv2d(m, 256, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu)
      m = slim.conv2d_transpose(m, 256, [2, 2], stride=2, padding='VALID', activation_fn=tf.nn.relu)
      m = slim.conv2d(m, 81, [1, 1], stride=1, padding='VALID', activation_fn=None)
      
      # add a mask, given the predicted boxes and classes
      outputs[p]['mask'] = {'mask':m, 'classes': classes, 'scores': scores}
      
  return outputs

def build_losses(pyramid, outputs, gt_boxes, gt_masks,
                 num_classes,
                 rpn_box_lw =1.0, rpn_cls_lw = 1.0,
                 refined_box_lw=1.0, refined_cls_lw=1.0,
                 mask_lw=1.0):
  """Building 3-way output losses, totally 5 losses
  Params:
  ------
  outputs: output of build_heads
  gt_boxes: A tensor of shape (G, 5), [x1, y1, x2, y2, class]
  gt_masks: A tensor of shape (G, ih, iw),  {0, 1}
  *_lw: loss weight of rpn, refined and mask losses
  
  Returns:
  -------
  l: a loss tensor
  """
  for i in range(5, 1, -1):
    p = 'P%d' % i
    stride = 2 ** i
    shape = tf.shape(pyramid[p])
    height, width = shape[1], shape[2]
    
    ### rpn losses
    # 1. encode ground truth
    # 2. compute distances
    all_anchors = gen_all_anchors(height, width, stride)
    labels, bbox_targets, bbox_inside_weights = \
      anchor_encoder(gt_boxes, all_anchors, height, width, stride, scope='AnchorEncoder')
    boxes = outputs[p]['rpn']['box']
    classes = outputs[p]['rpn']['cls']
    rpn_box_loss = bbox_inside_weights * _smooth_l1_dist(boxes, bbox_targets)
    rpn_box_loss = tf.reshape(rpn_box_loss, [-1, 4])
    rpn_box_loss = tf.reduce_sum(rpn_box_loss, axis=1)
    rpn_box_loss = rpn_box_lw * tf.reduce_mean(rpn_box_loss)

    labels = slim.one_hot_encoding(labels, 2, on_value=1.0, off_value=0.0)
    rpn_cls_loss = rpn_cls_lw * tf.losses.softmax_cross_entropy(classes, labels)
    
    ### refined loss
    # 1. encode ground truth
    # 2. compute distances
    boxes = outputs[p]['refined']['box']
    classes = outputs[p]['refined']['cls']
    labels, rois, bbox_targets, bbox_inside_weights = \
      roi_encoder(gt_boxes, boxes, num_classes, scope='ROIEncoder')
    refined_box_loss = bbox_inside_weights * _smooth_l1_dist(boxes, bbox_targets)
    refined_box_loss = tf.reshape(refined_box_loss, [-1, 4])
    refined_box_loss = tf.reduce_sum(refined_box_loss, axis=1)
    refined_box_loss = refined_box_lw * tf.reduce_mean(refined_box_loss)

    labels = slim.one_hot_encoding(labels, num_classes, on_value=1.0, off_value=0.0)
    refined_cls_loss = refined_cls_lw * tf.losses.softmax_cross_entropy(classes, labels)

    ### mask loss
    # {'mask': m, 'classes': classes, 'scores': scores}
    # TODO: re-sample in encoder? or outside?
    masks = outputs[p]['mask']['mask']
    rois, labels, mask_targets, mask_inside_weights = \
      mask_encoder(gt_masks, gt_boxes, rois, num_classes, 28, 28, scope='MaskEncoder')
    
    
  return
