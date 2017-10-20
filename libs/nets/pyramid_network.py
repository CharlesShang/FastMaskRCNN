# coding=utf-8
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
from libs.layers import sample_rpn_outputs_with_gt
from libs.layers import sample_rcnn_outputs
from libs.layers import assign_boxes
from libs.visualization.summary_utils import visualize_bb, visualize_final_predictions, visualize_input

_BN = True

# mapping each stage to its' tensor features
_networks_map = {
  'resnet50': {'C1':'resnet_v1_50/conv1/Relu:0',
               'C2':'resnet_v1_50/block1/unit_2/bottleneck_v1',
               'C3':'resnet_v1_50/block2/unit_3/bottleneck_v1',
               'C4':'resnet_v1_50/block3/unit_5/bottleneck_v1',
               'C5':'resnet_v1_50/block4/unit_3/bottleneck_v1',
               },
}

def _extra_conv_arg_scope_with_bn(weight_decay=0.00001,
                     activation_fn=None,
                     batch_norm_decay=0.9,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     is_training=True):

  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'is_training': is_training
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc

def _extra_conv_arg_scope(weight_decay=0.00001, activation_fn=None, normalizer_fn=None):

  with slim.arg_scope(
      [slim.conv2d, slim.conv2d_transpose],
      padding='SAME',
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),#tf.truncated_normal_initializer(stddev=0.001),
      activation_fn=tf.nn.relu,
      normalizer_fn=normalizer_fn,):
    with slim.arg_scope(
      [slim.fully_connected],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
          activation_fn=activation_fn,
          normalizer_fn=normalizer_fn):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc: 
          return arg_sc

def my_sigmoid(x):
    """add an active function for the box output layer, which is linear around 0"""
    return (tf.nn.sigmoid(x) - tf.cast(0.5, tf.float32)) * 6.0

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

def _get_valid_sample_fraction(labels, p=0):
    """return fraction of non-negative examples, the ignored examples have been marked as negative"""
    num_valid = tf.reduce_sum(tf.cast(tf.greater_equal(labels, p), tf.float32))
    num_example = tf.cast(tf.size(labels), tf.float32)
    frac = tf.cond(tf.greater(num_example, 0), lambda:num_valid / num_example,  
            lambda: tf.cast(0, tf.float32))
    frac_ = tf.cond(tf.greater(num_valid, 0), lambda:num_example / num_valid, 
            lambda: tf.cast(0, tf.float32))
    return frac, frac_


def _filter_negative_samples(labels, tensors):
    """keeps only samples with none-negative labels 
    Params:
    -----
    labels: of shape (N,)
    tensors: a list of tensors, each of shape (N, .., ..) the first axis is sample number

    Returns:
    -----
    tensors: filtered tensors
    """
    # return tensors
    keeps = tf.where(tf.greater_equal(labels, 0))
    keeps = tf.reshape(keeps, [-1])

    filtered = []
    for t in tensors:
        # tf.assert_equal(tf.shape(t)[0], tf.shape(labels)[0]) - I removed this assertion because it was never used.
        # assertion is not automatically checked you should execute it in graph as any other operation

        f = tf.gather(t, keeps)
        filtered.append(f)

    return filtered
        
def _add_jittered_boxes(rois, scores, batch_inds, gt_boxes, jitter=0.1):
    ws = gt_boxes[:, 2] - gt_boxes[:, 0]
    hs = gt_boxes[:, 3] - gt_boxes[:, 1]
    shape = tf.shape(gt_boxes)[0]
    jitter = tf.random_uniform([shape, 1], minval = -jitter, maxval = jitter)
    jitter = tf.reshape(jitter, [-1])
    ws_offset = ws * jitter
    hs_offset = hs * jitter
    x1s = gt_boxes[:, 0] + ws_offset
    x2s = gt_boxes[:, 2] + ws_offset
    y1s = gt_boxes[:, 1] + hs_offset
    y2s = gt_boxes[:, 3] + hs_offset
    boxes = tf.concat(
            values=[
                x1s[:, tf.newaxis],
                y1s[:, tf.newaxis],
                x2s[:, tf.newaxis],
                y2s[:, tf.newaxis]],
            axis=1)
    new_scores = tf.ones([shape], tf.float32)
    new_batch_inds = tf.zeros([shape], tf.int32)

    return tf.concat(values=[rois, boxes], axis=0), \
           tf.concat(values=[scores, new_scores], axis=0), \
           tf.concat(values=[batch_inds, new_batch_inds], axis=0)

def build_pyramid(net_name, end_points, bilinear=True, is_training=True):
  """Build pyramid (P2-P5) from typical network (convolutional layer C2-C5 of Resnet),
  assume each stage is 2 time larger than its top feature
  Returns:
    returns several endpoints
  """

  if _BN is True:
    if is_training is True:
      arg_scope = _extra_conv_arg_scope_with_bn()
    else:
      arg_scope = _extra_conv_arg_scope_with_bn(batch_norm_decay=0.0, weight_decay=0.0)
      #arg_scope = _extra_conv_arg_scope_with_bn(batch_norm_decay=0.0, weight_decay=0.0, is_training=is_training)
  else:
    arg_scope = _extra_conv_arg_scope(activation_fn=tf.nn.relu)
  #
  with tf.name_scope('pyramid') as py_scope:
    with slim.arg_scope(arg_scope) as slim_scope:
      pyramid = {}
      if isinstance(net_name, str):
        pyramid_map = _networks_map[net_name]
      else:
        pyramid_map = net_name
      
      pyramid['P5'] = \
        slim.conv2d(end_points[pyramid_map['C5']], 256, [1, 1], stride=1, scope='pyramid/C5')
      
      for c in range(4, 1, -1):
        s, s_ = pyramid['P%d'%(c+1)], end_points[pyramid_map['C%d' % (c)]]

        up_shape = tf.shape(s_)

        s = tf.image.resize_bilinear(s, [up_shape[1], up_shape[2]], name='pyramid/C%d/upscale'%c)
        s_ = slim.conv2d(s_, 256, [1,1], stride=1, scope='pyramid/C%d'%c)
        
        s = tf.add(s, s_, name='pyramid/C%d/addition'%c)
        s = slim.conv2d(s, 256, [3,3], stride=1, scope='pyramid/C%d/fusion'%c)
        
        pyramid['P%d'%(c)] = s
      
      return pyramid, py_scope, slim_scope
  
def build_heads(pyramid, py_scope, slim_scope, image_height, image_width, num_classes, base_anchors, is_training=False, gt_boxes=None):
  """Build the 3-way outputs, i.e., class, box and mask in the pyramid
  Algo
  ----
  For each layer:
    1. Build anchor layer
    2. Process the results of anchor layer, decode the output into rois 
    3. Sample rois 
    4. Build roi layer
    5. Process the results of roi layer, decode the output into boxes
    6. Build the mask layer
    7. Build losses
  """
  outputs = {}
  # if _BN is True:
  #   if is_training is True:
  #     arg_scope = _extra_conv_arg_scope_with_bn()
  #   else:
  #     arg_scope = _extra_conv_arg_scope_with_bn(batch_norm_decay=0.0)
  #   # arg_scope = _extra_conv_arg_scope_with_bn(is_training=is_training)
  # else:
  #   arg_scope = _extra_conv_arg_scope(activation_fn=tf.nn.relu)
  with tf.name_scope(py_scope) as py_scope:
    with slim.arg_scope(slim_scope) as slim_scope:
        ### for p in pyramid
        outputs['rpn'] = {}
        for i in range(5, 1, -1):
          p = 'P%d'%i
          stride = 2 ** i
          
          """Build RPN head
          RPN takes features from each layer of pyramid network. 
          strides are respectively set to [4, 8, 16, 32] for pyramid feature layer P2,P3,P4,P5 
          anchor_scales are set to [2 **(i-2), 2 ** (i-1), 2 **(i)] in all pyramid layers (*This is probably inconsistent with original paper where the only scale is 8)
          It generates 2 outputs.
          box: an array of shape (1, pyramid_height, pyramid_width, num_anchorx4). box regression values [shift_x, shift_y, scale_width, scale_height] are stored in the last dimension of the array.
          cls: an array of shape (1, pyramid_height, pyramid_width, num_anchorx2). Note that this value is before softmax   
          """
          shape = tf.shape(pyramid[p])
          height, width = shape[1], shape[2]
          rpn = slim.conv2d(pyramid[p], 256, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='pyramid/%s/rpn'%p)
          box = slim.conv2d(rpn, base_anchors * 4, [1, 1], stride=1, scope='pyramid/%s/rpn/box' % p, \
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.001), activation_fn=None, normalizer_fn=None)
          cls = slim.conv2d(rpn, base_anchors * 2, [1, 1], stride=1, scope='pyramid/%s/rpn/cls' % p, \
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01), activation_fn=None, normalizer_fn=None)

          anchor_scales = [8]#[2 **(i-2), 2 ** (i-1), 2 **(i)]
          print("anchor_scales = " , anchor_scales)
          all_anchors = gen_all_anchors(height, width, stride, anchor_scales)
          outputs['rpn'][p]={'box':box, 'cls':cls, 'anchor':all_anchors, 'shape':shape}

        ### gather boxes, clses, anchors from all pyramid layers
        rpn_boxes = [tf.reshape(outputs['rpn']['P%d'%p]['box'], [-1, 4]) for p in range(5, 1, -1)]  
        rpn_clses = [tf.reshape(outputs['rpn']['P%d'%p]['cls'], [-1, 1]) for p in range(5, 1, -1)]  
        rpn_anchors = [tf.reshape(outputs['rpn']['P%d'%p]['anchor'], [-1, 4]) for p in range(5, 1, -1)]  
        rpn_boxes = tf.concat(values=rpn_boxes, axis=0)
        rpn_clses = tf.concat(values=rpn_clses, axis=0)
        rpn_anchors = tf.concat(values=rpn_anchors, axis=0)
        
        ### softmax to get probability
        rpn_probs = tf.nn.softmax(tf.reshape(rpn_clses, [-1, 2])) 
        ### decode anchors and box regression values into proposed bounding boxes
        rpn_final_boxes, rpn_final_clses, rpn_final_scores = anchor_decoder(rpn_boxes, rpn_probs, rpn_anchors, image_height, image_width) 
        
        outputs['rpn_boxes'] = rpn_boxes
        outputs['rpn_clses'] = rpn_clses
        outputs['rpn_anchor'] = rpn_anchors
        outputs['rpn_final_boxes'] = rpn_final_boxes
        outputs['rpn_final_clses'] = rpn_final_clses
        outputs['rpn_final_scores'] = rpn_final_scores

        if is_training is True:
          ### for training, rcnn and maskrcnn take rpn proposed bounding boxes as inputs
          rpn_rois_to_rcnn, rpn_scores_to_rcnn, rpn_batch_inds_to_rcnn, rpn_rois_to_mask, rpn_scores_to_mask, rpn_batch_inds_to_mask = \
                sample_rpn_outputs_with_gt(rpn_final_boxes, rpn_final_scores, gt_boxes, is_training=is_training, only_positive=False)#True
        else:
          ### for testing, only rcnn takes rpn boxes as inputs. maskrcnn takes rcnn boxes as inputs
          rpn_rois_to_rcnn, rpn_scores_to_rcnn, rpn_batch_inds_to_rcnn = sample_rpn_outputs(rpn_final_boxes, rpn_final_scores, only_positive=False)
        
        ### assign pyramid layer indexs to rcnn network's ROIs.   
        [rcnn_assigned_rois, rcnn_assigned_batch_inds, rcnn_assigned_layer_inds] = \
              assign_boxes(rpn_rois_to_rcnn, [rpn_rois_to_rcnn, rpn_batch_inds_to_rcnn], [2, 3, 4, 5])
     
        ### crop features from pyramid using ROIs. Note that this will change order of the ROIs, so ROIs are also reordered.
        rcnn_cropped_features = []
        rcnn_ordered_rois = []
        for i in range(5, 1, -1):
            p = 'P%d'%i
            rcnn_splitted_roi = rcnn_assigned_rois[i-2]
            rcnn_batch_ind = rcnn_assigned_batch_inds[i-2]
            rcnn_cropped_feature, rcnn_rois_to_crop_and_resize = ROIAlign(pyramid[p], rcnn_splitted_roi, rcnn_batch_ind, image_height, image_width, stride=2**i,
                               pooled_height=14, pooled_width=14)
            rcnn_cropped_features.append(rcnn_cropped_feature)
            rcnn_ordered_rois.append(rcnn_splitted_roi)
            
        rcnn_cropped_features = tf.concat(values=rcnn_cropped_features, axis=0)
        rcnn_ordered_rois = tf.concat(values=rcnn_ordered_rois, axis=0)

        """Build rcnn head
        rcnn takes cropped features and generates 2 outputs. 
        rcnn_boxes: an array of shape (num_ROIs, num_classes x 4). Box regression values of each classes [shift_x, shift_y, scale_width, scale_height] are stored in the last dimension of the array.
        rcnn_clses: an array of shape (num_ROIs, num_classes). Class prediction values (before softmax) are stored
        """
        rcnn = slim.max_pool2d(rcnn_cropped_features, [3, 3], stride=2, padding='SAME')
        rcnn = slim.flatten(rcnn)
        rcnn = slim.fully_connected(rcnn, 1024, activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.001), scope="pyramid/fully_connected")
        rcnn = slim.dropout(rcnn, keep_prob=0.75, is_training=is_training)#is_training
        rcnn = slim.fully_connected(rcnn,  1024, activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.001), scope="pyramid/fully_connected_1")
        rcnn = slim.dropout(rcnn, keep_prob=0.75, is_training=is_training)#is_training
        rcnn_clses = slim.fully_connected(rcnn, num_classes, activation_fn=None, normalizer_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.001), scope="pyramid/fully_connected_2")
        rcnn_boxes = slim.fully_connected(rcnn, num_classes*4, activation_fn=None, normalizer_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.001), scope="pyramid/fully_connected_3")
        
        ### softmax to get probability
        rcnn_scores = tf.nn.softmax(rcnn_clses)

        ### decode ROIs and box regression values into bounding boxes
        rcnn_final_boxes, rcnn_final_classes, rcnn_final_scores = roi_decoder(rcnn_boxes, rcnn_scores, rcnn_ordered_rois, image_height, image_width)
  
        outputs['rcnn_ordered_rois'] = rcnn_ordered_rois
        outputs['rcnn_cropped_features'] = rcnn_cropped_features
        tf.add_to_collection('__CROPPED__', rcnn_cropped_features)
        outputs['rcnn_boxes'] = rcnn_boxes
        outputs['rcnn_clses'] = rcnn_clses
        outputs['rcnn_scores'] = rcnn_scores
        outputs['rcnn_final_boxes'] = rcnn_final_boxes
        outputs['rcnn_final_clses'] = rcnn_final_classes
        outputs['rcnn_final_scores'] = rcnn_final_scores
        
        if is_training:
          ### assign pyramid layer indexs to mask network's ROIs
          [mask_assigned_rois, mask_assigned_batch_inds, mask_assigned_layer_inds] = \
               assign_boxes(rpn_rois_to_mask, [rpn_rois_to_mask, rpn_batch_inds_to_mask], [2, 3, 4, 5])

          ### crop features from pyramid using ROIs. Again, this will change order of the ROIs, so ROIs are reordered.
          mask_cropped_features = []
          mask_ordered_rois = []

          ### crop features from pyramid for mask network
          for i in range(5, 1, -1):
              p = 'P%d'%i
              mask_splitted_roi = mask_assigned_rois[i-2]
              mask_batch_ind = mask_assigned_batch_inds[i-2]
              mask_cropped_feature, mask_rois_to_crop_and_resize = ROIAlign(pyramid[p], mask_splitted_roi, mask_batch_ind, image_height, image_width, stride=2**i,
                                 pooled_height=14, pooled_width=14)
              mask_cropped_features.append(mask_cropped_feature)
              mask_ordered_rois.append(mask_splitted_roi)
              
          mask_cropped_features = tf.concat(values=mask_cropped_features, axis=0)
          mask_ordered_rois = tf.concat(values=mask_ordered_rois, axis=0)
          
        else:
          ### for testing, mask network takes rcnn boxes as inputs
          rcnn_rois_to_mask, rcnn_clses_to_mask, rcnn_scores_to_mask, rcnn_batch_inds_to_mask = sample_rcnn_outputs(rcnn_final_boxes, rcnn_final_classes, rcnn_scores, class_agnostic=False) 
          [mask_assigned_rois, mask_assigned_clses, mask_assigned_scores, mask_assigned_batch_inds, mask_assigned_layer_inds] =\
               assign_boxes(rcnn_rois_to_mask, [rcnn_rois_to_mask, rcnn_clses_to_mask, rcnn_scores_to_mask, rcnn_batch_inds_to_mask], [2, 3, 4, 5])
          
          mask_cropped_features = []
          mask_ordered_rois = []
          mask_ordered_clses = []
          mask_ordered_scores = []
          for i in range(5, 1, -1):
            p = 'P%d'%i
            mask_splitted_roi = mask_assigned_rois[i-2]
            mask_splitted_cls = mask_assigned_clses[i-2]
            mask_splitted_score = mask_assigned_scores[i-2]
            mask_batch_ind = mask_assigned_batch_inds[i-2]
            mask_cropped_feature, mask_rois_to_crop_and_resize = ROIAlign(pyramid[p], mask_splitted_roi, mask_batch_ind, image_height, image_width, stride=2**i,
                               pooled_height=14, pooled_width=14)
            mask_cropped_features.append(mask_cropped_feature)
            mask_ordered_rois.append(mask_splitted_roi)
            mask_ordered_clses.append(mask_splitted_cls)
            mask_ordered_scores.append(mask_splitted_score)

          mask_cropped_features = tf.concat(values=mask_cropped_features, axis=0)
          mask_ordered_rois = tf.concat(values=mask_ordered_rois, axis=0)
          mask_ordered_clses = tf.concat(values=mask_ordered_clses, axis=0)
          mask_ordered_scores = tf.concat(values=mask_ordered_scores, axis=0)

          outputs['mask_final_clses'] = mask_ordered_clses
          outputs['mask_final_scores'] = mask_ordered_scores

        """Build mask rcnn head
        mask rcnn takes cropped features and generates masks for each classes. 
        m: an array of shape (28, 28, num_classes). Note that this value is before sigmoid.
        """
        m = mask_cropped_features
        m = slim.conv2d(m, 256, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope="pyramid/Conv")
        m = slim.conv2d(m, 256, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope="pyramid/Conv_1")
        m = slim.conv2d(m, 256, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope="pyramid/Conv_2")
        m = slim.conv2d(m, 256, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope="pyramid/Conv_3")
        m = slim.conv2d_transpose(m, 256, 2, stride=2, padding='VALID', activation_fn=tf.nn.relu, scope="pyramid/Conv2d_transpose")
        tf.add_to_collection('__TRANSPOSED__', m)
        m = slim.conv2d(m, num_classes, [1, 1], stride=1, padding='VALID', activation_fn=None, normalizer_fn=None, scope="pyramid/Conv_4")

        outputs['mask_ordered_rois'] = mask_ordered_rois
        outputs['mask_cropped_features'] = mask_cropped_features 
        outputs['mask_mask'] = m
        outputs['mask_final_mask'] = tf.nn.sigmoid(m)
          
        return outputs, py_scope, slim_scope

def build_losses(pyramid, py_scope, slim_scope, image_height, image_width, outputs, gt_boxes, gt_masks,
                 num_classes, base_anchors,
                 rpn_box_lw =0.1, rpn_cls_lw = 0.1,
                 rcnn_box_lw=1.0, rcnn_cls_lw=0.1,
                 mask_lw=1.0):
  """Building 3-way output losses, totally 5 losses
  Params:
  ------
  outputs: output of build_heads
  gt_boxes: A tensor of shape (G, 5), [x1, y1, x2, y2, class]
  gt_masks: A tensor of shape (G, ih, iw),  {0, 1}Ì[MaÌ[MaÌ]]
  *_lw: loss weight of rpn, rcnn and mask losses
  
  Returns:
  -------
  l: a loss tensor
  """

  # losses for pyramid
  losses = []
  rpn_box_losses, rpn_cls_losses = [], []
  rcnn_box_losses, rcnn_cls_losses = [], []
  mask_losses = []
  
  # watch some info during training
  rpn_batch = []
  rcnn_batch = []
  mask_batch = []
  rpn_batch_pos = []
  rcnn_batch_pos = []
  mask_batch_pos = []

  # if _BN is True:
  #     arg_scope = _extra_conv_arg_scope_with_bn()
  #   # arg_scope = _extra_conv_arg_scope_with_bn(is_training=True)
  # else:
  #   arg_scope = _extra_conv_arg_scope(activation_fn=tf.nn.relu)
  with tf.name_scope(py_scope) as py_scope:
      with slim.arg_scope(slim_scope) as slim_scope:
        ## assigning gt_boxes
        [assigned_gt_boxes, assigned_layer_inds] = assign_boxes(gt_boxes, [gt_boxes], [2, 3, 4, 5])

        ## build losses for PFN
        for i in range(5, 1, -1):
            p = 'P%d' % i
            stride = 2 ** i
            shape = tf.shape(pyramid[p])
            height, width = shape[1], shape[2]

            splitted_gt_boxes = assigned_gt_boxes[i-2]
            
            ### rpn losses
            # 1. encode ground truth
            # 2. compute distances
            all_anchors = outputs['rpn'][p]['anchor']
            rpn_boxes = outputs['rpn'][p]['box']
            rpn_clses = tf.reshape(outputs['rpn'][p]['cls'], (1, height, width, base_anchors, 2))

            rpn_clses_target, rpn_boxes_target, rpn_boxes_inside_weight = \
                    anchor_encoder(splitted_gt_boxes, all_anchors, height, width, stride, image_height, image_width, scope='AnchorEncoder')

            rpn_clses_target, rpn_clses, rpn_boxes, rpn_boxes_target, rpn_boxes_inside_weight = \
                    _filter_negative_samples(tf.reshape(rpn_clses_target, [-1]), [
                        tf.reshape(rpn_clses_target, [-1]),
                        tf.reshape(rpn_clses, [-1, 2]),
                        tf.reshape(rpn_boxes, [-1, 4]),
                        tf.reshape(rpn_boxes_target, [-1, 4]),
                        tf.reshape(rpn_boxes_inside_weight, [-1, 4])
                        ])

            rpn_batch.append(
                    tf.reduce_sum(tf.cast(
                        tf.greater_equal(rpn_clses_target, 0), tf.float32
                        )))
            rpn_batch_pos.append(
                    tf.reduce_sum(tf.cast(
                        tf.greater_equal(rpn_clses_target, 1), tf.float32
                        )))

            rpn_box_loss = rpn_boxes_inside_weight * _smooth_l1_dist(rpn_boxes, rpn_boxes_target)
            rpn_box_loss = tf.reshape(rpn_box_loss, [-1, 4])
            rpn_box_loss = tf.reduce_sum(rpn_box_loss, axis=1)
            rpn_box_loss = rpn_box_lw * tf.reduce_mean(rpn_box_loss) 
            tf.add_to_collection(tf.GraphKeys.LOSSES, rpn_box_loss)
            rpn_box_losses.append(rpn_box_loss)

            ### NOTE: examples with negative labels are ignore when compute one_hot_encoding and entropy losses 
            # BUT these examples still count when computing the average of softmax_cross_entropy, 
            # the loss become smaller by a factor (None_negtive_labels / all_labels)
            # the BEST practise still should be gathering all none-negative examples
            rpn_clses_target = slim.one_hot_encoding(rpn_clses_target, 2, on_value=1.0, off_value=0.0) # this will set -1 label to all zeros
            rpn_cls_loss = rpn_cls_lw * tf.nn.softmax_cross_entropy_with_logits(labels=rpn_clses_target, logits=rpn_clses) 
            rpn_cls_loss = tf.reduce_mean(rpn_cls_loss) 
            tf.add_to_collection(tf.GraphKeys.LOSSES, rpn_cls_loss)
            rpn_cls_losses.append(rpn_cls_loss)

        ### rcnn losses
        # 1. encode ground truth
        # 2. compute distances
        rcnn_ordered_rois = outputs['rcnn_ordered_rois']
        rcnn_boxes = outputs['rcnn_boxes']
        rcnn_clses = outputs['rcnn_clses']
        rcnn_scores = outputs['rcnn_scores']

        rcnn_clses_target, rcnn_boxes_target, rcnn_boxes_inside_weight = \
                roi_encoder(gt_boxes, rcnn_ordered_rois, num_classes, scope='ROIEncoder')

        rcnn_clses_target, rcnn_ordered_rois, rcnn_clses, rcnn_scores, rcnn_boxes, rcnn_boxes_target, rcnn_boxes_inside_weight = \
                _filter_negative_samples(tf.reshape(rcnn_clses_target, [-1]),[
                    tf.reshape(rcnn_clses_target, [-1]),
                    tf.reshape(rcnn_ordered_rois, [-1, 4]),
                    tf.reshape(rcnn_clses, [-1, num_classes]),
                    tf.reshape(rcnn_scores, [-1, num_classes]),
                    tf.reshape(rcnn_boxes, [-1, num_classes * 4]),
                    tf.reshape(rcnn_boxes_target, [-1, num_classes * 4]),
                    tf.reshape(rcnn_boxes_inside_weight, [-1, num_classes * 4])
                    ] )

        rcnn_batch.append(
                tf.reduce_sum(tf.cast(
                    tf.greater_equal(rcnn_clses_target, 0), tf.float32
                    )))
        rcnn_batch_pos.append(
                tf.reduce_sum(tf.cast(
                    tf.greater_equal(rcnn_clses_target, 1), tf.float32
                    )))

        rcnn_box_loss = rcnn_boxes_inside_weight * _smooth_l1_dist(rcnn_boxes, rcnn_boxes_target)
        rcnn_box_loss = tf.reshape(rcnn_box_loss, [-1, 4])
        rcnn_box_loss = tf.reduce_sum(rcnn_box_loss, axis=1)
        rcnn_box_loss = rcnn_box_lw * tf.reduce_mean(rcnn_box_loss) # * frac_
        tf.add_to_collection(tf.GraphKeys.LOSSES, rcnn_box_loss)
        rcnn_box_losses.append(rcnn_box_loss)

        rcnn_clses_target = slim.one_hot_encoding(rcnn_clses_target, num_classes, on_value=1.0, off_value=0.0)
        rcnn_cls_loss = rcnn_cls_lw * tf.nn.softmax_cross_entropy_with_logits(labels=rcnn_clses_target, logits=rcnn_clses) 
        rcnn_cls_loss = tf.reduce_mean(rcnn_cls_loss) # * frac_
        tf.add_to_collection(tf.GraphKeys.LOSSES, rcnn_cls_loss)
        rcnn_cls_losses.append(rcnn_cls_loss)

        outputs['training_rcnn_rois'] = rcnn_ordered_rois
        outputs['training_rcnn_clses_target'] = rcnn_clses_target
        outputs['training_rcnn_clses'] = rcnn_clses
        outputs['training_rcnn_scores'] = rcnn_scores

        ### mask loss
        # mask of shape (N, h, w, num_classes)
        mask_ordered_rois = outputs['mask_ordered_rois']
        masks = outputs['mask_mask']

        mask_clses_target, mask_targets, mask_inside_weights, mask_rois = \
                mask_encoder(gt_masks, gt_boxes, mask_ordered_rois, num_classes, 28, 28,scope='MaskEncoder')

        mask_clses_target, mask_targets, mask_inside_weights, mask_rois, masks = \
                _filter_negative_samples(tf.reshape(mask_clses_target, [-1]), [
                    tf.reshape(mask_clses_target, [-1]),
                    tf.reshape(mask_targets, [-1, 28, 28, num_classes]),
                    tf.reshape(mask_inside_weights, [-1, 28, 28, num_classes]),
                    tf.reshape(mask_rois, [-1, 4]),
                    tf.reshape(masks, [-1, 28, 28, num_classes]),
                    ])

        mask_batch.append(
                tf.reduce_sum(tf.cast(
                    tf.greater_equal(mask_clses_target, 0), tf.float32
                    )))
        mask_batch_pos.append(
                tf.reduce_sum(tf.cast(
                    tf.greater_equal(mask_clses_target, 1), tf.float32
                    )))
        ### NOTE: w/o competition between classes. 
        mask_loss = mask_inside_weights * tf.nn.sigmoid_cross_entropy_with_logits(labels=mask_targets, logits=masks) 
        mask_loss = mask_lw * mask_loss
        mask_loss = tf.reduce_mean(mask_loss) 
        mask_loss = tf.cond(tf.greater(tf.size(mask_clses_target), 0), lambda: mask_loss, lambda: tf.constant(0.0))
        tf.add_to_collection(tf.GraphKeys.LOSSES, mask_loss)
        mask_losses.append(mask_loss)

        outputs['training_mask_rois'] = mask_rois
        outputs['training_mask_clses_target'] = mask_clses_target
        outputs['training_mask_final_mask'] = tf.nn.sigmoid(masks)
        outputs['training_mask_final_mask_target'] = mask_targets

        rpn_box_losses = tf.add_n(rpn_box_losses)
        rpn_cls_losses = tf.add_n(rpn_cls_losses)
        rcnn_box_losses = tf.add_n(rcnn_box_losses)
        rcnn_cls_losses = tf.add_n(rcnn_cls_losses)
        mask_losses = tf.add_n(mask_losses)
        losses = [rpn_box_losses, rpn_cls_losses, rcnn_box_losses, rcnn_cls_losses, mask_losses]
        total_loss = tf.add_n(losses)

        rpn_batch = tf.cast(tf.add_n(rpn_batch), tf.float32)
        rcnn_batch = tf.cast(tf.add_n(rcnn_batch), tf.float32)
        mask_batch = tf.cast(tf.add_n(mask_batch), tf.float32)
        rpn_batch_pos = tf.cast(tf.add_n(rpn_batch_pos), tf.float32)
        rcnn_batch_pos = tf.cast(tf.add_n(rcnn_batch_pos), tf.float32)
        mask_batch_pos = tf.cast(tf.add_n(mask_batch_pos), tf.float32)
         
        return total_loss, losses, [rpn_batch_pos, rpn_batch, \
                                    rcnn_batch_pos, rcnn_batch, \
                                    mask_batch_pos, mask_batch]

def decode_output(outputs):
    """decode outputs into boxes and masks"""
    return [], [], []

def build(end_points, image_height, image_width, pyramid_map, 
        num_classes,
        base_anchors,
        is_training,
        gt_boxes=None,
        gt_masks=None, 
        loss_weights=[0.1, 0.1, 1.0, 0.1, 1.0]):
    
    pyramid, py_scope, slim_scope = build_pyramid(pyramid_map, end_points, is_training=is_training)

    if is_training: 
      outputs, py_scope, slim_scope = \
          build_heads(pyramid, py_scope, slim_scope, image_height, image_width, num_classes, base_anchors, 
                      is_training=is_training, gt_boxes=gt_boxes)
      loss, losses, batch_info = build_losses(pyramid, py_scope, slim_scope, image_height, image_width, outputs, 
                      gt_boxes, gt_masks,
                      num_classes=num_classes, base_anchors=base_anchors,
                      rpn_box_lw=loss_weights[0], rpn_cls_lw=loss_weights[1],
                      rcnn_box_lw=loss_weights[2], rcnn_cls_lw=loss_weights[3],
                      mask_lw=loss_weights[4])

      outputs['losses'] = losses
      outputs['total_loss'] = loss
      outputs['batch_info'] = batch_info
    else:
      outputs, py_scope, slim_scope = \
          build_heads(pyramid, py_scope, slim_scope, image_height, image_width, num_classes, base_anchors, 
                      is_training=is_training)

    ### just decode outputs into readable prediction
    # pred_boxes, pred_classes, pred_masks = decode_output(outputs)
    # outputs['pred_boxes'] = pred_boxes
    # outputs['pred_classes'] = pred_classes
    # outputs['pred_masks'] = pred_masks


    # ### image and gt visualization
    # visualize_input(gt_boxes, end_points["input"], tf.expand_dims(gt_masks, axis=3))

    # ### rpn visualization
    # visualize_bb(end_points["input"], outputs['rpn_final_boxes'], name="rpn_bb_visualization")

    # ### mask network visualization
    # first_mask = outputs['training_mask_final_mask'][:1]
    # first_mask = tf.transpose(first_mask, [3, 1, 2, 0])

    # visualize_final_predictions(outputs['rcnn_final_boxes'], end_points["input"], first_mask)

    return outputs
