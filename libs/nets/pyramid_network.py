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
from libs.layers import ROIAlign_
from libs.layers import sample_rpn_outputs
from libs.layers import sample_rpn_outputs_with_gt
from libs.layers import assign_boxes
from libs.visualization.summary_utils import visualize_bb, visualize_final_predictions, visualize_input

_TRAIN_MASK = True

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

def _extra_conv_arg_scope_with_bn(weight_decay=0.00001,
                     activation_fn=None,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):

  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
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
      weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,) as arg_sc:
    with slim.arg_scope(
      [slim.fully_connected],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
          activation_fn=activation_fn,
          normalizer_fn=normalizer_fn) as arg_sc:
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
        tf.assert_equal(tf.shape(t)[0], tf.shape(labels)[0])
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

def build_pyramid(net_name, end_points, bilinear=True):
  """build pyramid features from a typical network,
  assume each stage is 2 time larger than its top feature
  Returns:
    returns several endpoints
  """
  pyramid = {}
  if isinstance(net_name, str):
    pyramid_map = _networks_map[net_name]
  else:
    pyramid_map = net_name
  # pyramid['inputs'] = end_points['inputs']
  #arg_scope = _extra_conv_arg_scope()
  arg_scope = _extra_conv_arg_scope_with_bn()
  with tf.variable_scope('pyramid'):
    with slim.arg_scope(arg_scope):
      
      pyramid['P5'] = \
        slim.conv2d(end_points[pyramid_map['C5']], 256, [1, 1], stride=1, scope='C5')
      
      for c in range(4, 1, -1):
        s, s_ = pyramid['P%d'%(c+1)], end_points[pyramid_map['C%d' % (c)]]

        # s_ = slim.conv2d(s_, 256, [3, 3], stride=1, scope='C%d'%c)
        
        up_shape = tf.shape(s_)
        # out_shape = tf.stack((up_shape[1], up_shape[2]))
        # s = slim.conv2d(s, 256, [3, 3], stride=1, scope='C%d'%c)
        s = tf.image.resize_bilinear(s, [up_shape[1], up_shape[2]], name='C%d/upscale'%c)
        s_ = slim.conv2d(s_, 256, [1,1], stride=1, scope='C%d'%c)
        
        s = tf.add(s, s_, name='C%d/addition'%c)
        s = slim.conv2d(s, 256, [3,3], stride=1, scope='C%d/fusion'%c)
        
        pyramid['P%d'%(c)] = s
      
      return pyramid
  
def build_heads(pyramid, ih, iw, num_classes, base_anchors, is_training=False, gt_boxes=None):
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
  #arg_scope = _extra_conv_arg_scope(activation_fn=None)
  arg_scope = _extra_conv_arg_scope_with_bn(activation_fn=None)
  my_sigmoid = None
  with slim.arg_scope(arg_scope):
    with tf.variable_scope('pyramid'):
        # for p in pyramid:
        outputs['rpn'] = {}
        for i in range(5, 1, -1):
          p = 'P%d'%i
          stride = 2 ** i
          
          ## rpn head
          shape = tf.shape(pyramid[p])
          height, width = shape[1], shape[2]
          rpn = slim.conv2d(pyramid[p], 256, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='%s/rpn'%p)
          box = slim.conv2d(rpn, base_anchors * 4, [1, 1], stride=1, scope='%s/rpn/box' % p, \
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.001), activation_fn=my_sigmoid)
          cls = slim.conv2d(rpn, base_anchors * 2, [1, 1], stride=1, scope='%s/rpn/cls' % p, \
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01))

          anchor_scales = [2 **(i-2), 2 ** (i-1), 2 **(i)]
          print("anchor_scales = " , anchor_scales)
          all_anchors = gen_all_anchors(height, width, stride, anchor_scales)
          outputs['rpn'][p]={'box':box, 'cls':cls, 'anchor':all_anchors}

        ## gather all rois
        # print (outputs['rpn'])
        rpn_boxes = [tf.reshape(outputs['rpn']['P%d'%p]['box'], [-1, 4]) for p in range(5, 1, -1)]  
        rpn_clses = [tf.reshape(outputs['rpn']['P%d'%p]['cls'], [-1, 1]) for p in range(5, 1, -1)]  
        rpn_anchors = [tf.reshape(outputs['rpn']['P%d'%p]['anchor'], [-1, 4]) for p in range(5, 1, -1)]  
        rpn_boxes = tf.concat(values=rpn_boxes, axis=0)
        rpn_clses = tf.concat(values=rpn_clses, axis=0)
        rpn_anchors = tf.concat(values=rpn_anchors, axis=0)

        outputs['rpn']['box'] = rpn_boxes
        outputs['rpn']['cls'] = rpn_clses
        outputs['rpn']['anchor'] = rpn_anchors
        # outputs['rpn'] = {'box': rpn_boxes, 'cls': rpn_clses, 'anchor': rpn_anchors}
        
        rpn_probs = tf.nn.softmax(tf.reshape(rpn_clses, [-1, 2]))
        rois, roi_clses, scores, = anchor_decoder(rpn_boxes, rpn_probs, rpn_anchors, ih, iw)
        # rois, scores, batch_inds = sample_rpn_outputs(rois, rpn_probs[:, 1])
        rois, scores, batch_inds, mask_rois, mask_scores, mask_batch_inds = \
                sample_rpn_outputs_with_gt(rois, rpn_probs[:, 1], gt_boxes, is_training=is_training)

        # if is_training:
        #     # rois, scores, batch_inds = _add_jittered_boxes(rois, scores, batch_inds, gt_boxes)
        #     rois, scores, batch_inds = _add_jittered_boxes(rois, scores, batch_inds, gt_boxes, jitter=0.2)
        
        outputs['roi'] = {'box': rois, 'score': scores}

        ## cropping regions
        [assigned_rois, assigned_batch_inds, assigned_layer_inds] = \
                assign_boxes(rois, [rois, batch_inds], [2, 3, 4, 5])

        outputs['assigned_rois'] = assigned_rois
        outputs['assigned_layer_inds'] = assigned_layer_inds

        cropped_rois = []
        ordered_rois = []
        pyramid_feature = []
        for i in range(5, 1, -1):
            print(i)
            p = 'P%d'%i
            splitted_rois = assigned_rois[i-2]
            batch_inds = assigned_batch_inds[i-2]
            cropped, boxes_in_crop = ROIAlign_(pyramid[p], splitted_rois, batch_inds, ih, iw, stride=2**i,
                               pooled_height=14, pooled_width=14)
            # cropped = ROIAlign(pyramid[p], splitted_rois, batch_inds, stride=2**i,
            #                    pooled_height=14, pooled_width=14)
            cropped_rois.append(cropped)
            ordered_rois.append(splitted_rois)
            pyramid_feature.append(tf.transpose(pyramid[p],[0,3,1,2]))
            # if i is 5:
            #     outputs['tmp_0'] = tf.transpose(pyramid[p],[0,3,1,2])
            #     outputs['tmp_1'] = splitted_rois
            #     outputs['tmp_2'] = tf.transpose(cropped,[0,3,1,2])
            #     outputs['tmp_3'] = boxes_in_crop
            #     outputs['tmp_4'] = [ih, iw]
            
        cropped_rois = tf.concat(values=cropped_rois, axis=0)
        ordered_rois = tf.concat(values=ordered_rois, axis=0)


        outputs['ordered_rois'] = ordered_rois
        outputs['pyramid_feature'] = pyramid_feature

        outputs['roi']['cropped_rois'] = cropped_rois
        tf.add_to_collection('__CROPPED__', cropped_rois)

        ## refine head
        # to 7 x 7
        cropped_regions = slim.max_pool2d(cropped_rois, [3, 3], stride=2, padding='SAME')
        refine = slim.flatten(cropped_regions)
        refine = slim.fully_connected(refine, 1024, activation_fn=tf.nn.relu)
        refine = slim.dropout(refine, keep_prob=0.75, is_training=is_training)
        refine = slim.fully_connected(refine,  1024, activation_fn=tf.nn.relu)
        refine = slim.dropout(refine, keep_prob=0.75, is_training=is_training)
        cls2 = slim.fully_connected(refine, num_classes, activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.05))
        box = slim.fully_connected(refine, num_classes*4, activation_fn=my_sigmoid, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.05))

        outputs['refined'] = {'box': box, 'cls': cls2}
        
        ## decode refine net outputs
        cls2_prob = tf.nn.softmax(cls2)
        final_boxes, classes, scores = \
                roi_decoder(box, cls2_prob, ordered_rois, ih, iw)

        #outputs['tmp_0'] = ordered_rois
        #outputs['tmp_1'] = assigned_rois
        #outputs['tmp_2'] = box
        #outputs['tmp_3'] = final_boxes
        #outputs['tmp_4'] = cls2_prob

        #outputs['final_boxes'] = {'box': final_boxes, 'cls': classes}
        outputs['final_boxes'] = {'box': final_boxes, 'cls': classes, 'prob': cls2_prob}
        ## for testing, maskrcnn takes refined boxes as inputs
        if not is_training:
          rois = final_boxes
          # [assigned_rois, assigned_batch_inds, assigned_layer_inds] = \
          #       assign_boxes(rois, [rois, batch_inds], [2, 3, 4, 5])
          for i in range(5, 1, -1):
            p = 'P%d'%i
            splitted_rois = assigned_rois[i-2]
            batch_inds = assigned_batch_inds[i-2]
            cropped = ROIAlign(pyramid[p], splitted_rois, batch_inds, stride=2**i,
                               pooled_height=14, pooled_width=14)
            cropped_rois.append(cropped)
            ordered_rois.append(splitted_rois)
          cropped_rois = tf.concat(values=cropped_rois, axis=0)
          ordered_rois = tf.concat(values=ordered_rois, axis=0)
          
        ## mask head
        m = cropped_rois
        for _ in range(4):
            m = slim.conv2d(m, 256, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu)
        # to 28 x 28
        m = slim.conv2d_transpose(m, 256, 2, stride=2, padding='VALID', activation_fn=tf.nn.relu)
        tf.add_to_collection('__TRANSPOSED__', m)
        m = slim.conv2d(m, num_classes, [1, 1], stride=1, padding='VALID', activation_fn=None)
          
        # add a mask, given the predicted boxes and classes
        outputs['mask'] = {'mask':m, 'cls': classes, 'score': scores}
          
  return outputs

def build_losses(pyramid, outputs, gt_boxes, gt_masks,
                 num_classes, base_anchors,
                 rpn_box_lw =1.0, rpn_cls_lw = 1.0,
                 refined_box_lw=1.0, refined_cls_lw=1.0,
                 mask_lw=1.0):
  """Building 3-way output losses, totally 5 losses
  Params:
  ------
  outputs: output of build_heads
  gt_boxes: A tensor of shape (G, 5), [x1, y1, x2, y2, class]
  gt_masks: A tensor of shape (G, ih, iw),  {0, 1}Ì[MaÌ[MaÌ]]
  *_lw: loss weight of rpn, refined and mask losses
  
  Returns:
  -------
  l: a loss tensor
  """

  # losses for pyramid
  losses = []
  rpn_box_losses, rpn_cls_losses = [], []
  refined_box_losses, refined_cls_losses = [], []
  mask_losses = []
  
  # watch some info during training
  rpn_batch = []
  refine_batch = []
  mask_batch = []
  rpn_batch_pos = []
  refine_batch_pos = []
  mask_batch_pos = []

  #arg_scope = _extra_conv_arg_scope(activation_fn=None)
  arg_scope = _extra_conv_arg_scope_with_bn(activation_fn=None)
  with slim.arg_scope(arg_scope):
      with tf.variable_scope('pyramid'):

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
            # anchor_scales = [2 **(i-2), 2 ** (i-1), 2 **(i)]
            # all_anchors = gen_all_anchors(height, width, stride, anchor_scales)
            all_anchors = outputs['rpn'][p]['anchor']
            labels, bbox_targets, bbox_inside_weights = \
              anchor_encoder(splitted_gt_boxes, all_anchors, height, width, stride, scope='AnchorEncoder')
            boxes = outputs['rpn'][p]['box']
            classes = tf.reshape(outputs['rpn'][p]['cls'], (1, height, width, base_anchors, 2))

            labels, classes, boxes, bbox_targets, bbox_inside_weights = \
                    _filter_negative_samples(tf.reshape(labels, [-1]), [
                        tf.reshape(labels, [-1]),
                        tf.reshape(classes, [-1, 2]),
                        tf.reshape(boxes, [-1, 4]),
                        tf.reshape(bbox_targets, [-1, 4]),
                        tf.reshape(bbox_inside_weights, [-1, 4])
                        ])
            # _, frac_ = _get_valid_sample_fraction(labels)
            rpn_batch.append(
                    tf.reduce_sum(tf.cast(
                        tf.greater_equal(labels, 0), tf.float32
                        )))
            rpn_batch_pos.append(
                    tf.reduce_sum(tf.cast(
                        tf.greater_equal(labels, 1), tf.float32
                        )))
            rpn_box_loss = bbox_inside_weights * _smooth_l1_dist(boxes, bbox_targets)
            rpn_box_loss = tf.reshape(rpn_box_loss, [-1, 4])
            rpn_box_loss = tf.reduce_sum(rpn_box_loss, axis=1)
            rpn_box_loss = rpn_box_lw * tf.reduce_mean(rpn_box_loss) 
            tf.add_to_collection(tf.GraphKeys.LOSSES, rpn_box_loss)
            rpn_box_losses.append(rpn_box_loss)

            # NOTE: examples with negative labels are ignore when compute one_hot_encoding and entropy losses 
            # BUT these examples still count when computing the average of softmax_cross_entropy, 
            # the loss become smaller by a factor (None_negtive_labels / all_labels)
            # the BEST practise still should be gathering all none-negative examples
            labels = slim.one_hot_encoding(labels, 2, on_value=1.0, off_value=0.0) # this will set -1 label to all zeros
            rpn_cls_loss = rpn_cls_lw * tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=classes) 
            rpn_cls_loss = tf.reduce_mean(rpn_cls_loss) 
            tf.add_to_collection(tf.GraphKeys.LOSSES, rpn_cls_loss)
            rpn_cls_losses.append(rpn_cls_loss)
            

        ### refined loss
        # 1. encode ground truth
        # 2. compute distances
        ordered_rois = outputs['ordered_rois']
        #rois = outputs['roi']['box']
        
        boxes = outputs['refined']['box']
        classes = outputs['refined']['cls']

        labels, bbox_targets, bbox_inside_weights = \
          roi_encoder(gt_boxes, ordered_rois, num_classes, scope='ROIEncoder')

        outputs['final_boxes']['gt_cls'] = slim.one_hot_encoding(labels, num_classes, on_value=1.0, off_value=0.0)
        outputs['gt'] = gt_boxes
        labels, classes, boxes, bbox_targets, bbox_inside_weights = \
                _filter_negative_samples(tf.reshape(labels, [-1]),[
                    tf.reshape(labels, [-1]),
                    tf.reshape(classes, [-1, num_classes]),
                    tf.reshape(boxes, [-1, num_classes * 4]),
                    tf.reshape(bbox_targets, [-1, num_classes * 4]),
                    tf.reshape(bbox_inside_weights, [-1, num_classes * 4])
                    ] )
        # frac, frac_ = _get_valid_sample_fraction(labels, 1)
        refine_batch.append(
                tf.reduce_sum(tf.cast(
                    tf.greater_equal(labels, 0), tf.float32
                    )))
        refine_batch_pos.append(
                tf.reduce_sum(tf.cast(
                    tf.greater_equal(labels, 1), tf.float32
                    )))

        refined_box_loss = bbox_inside_weights * _smooth_l1_dist(boxes, bbox_targets)
        refined_box_loss = tf.reshape(refined_box_loss, [-1, 4])
        refined_box_loss = tf.reduce_sum(refined_box_loss, axis=1)
        refined_box_loss = refined_box_lw * tf.reduce_mean(refined_box_loss) # * frac_
        tf.add_to_collection(tf.GraphKeys.LOSSES, refined_box_loss)
        refined_box_losses.append(refined_box_loss)

        labels = slim.one_hot_encoding(labels, num_classes, on_value=1.0, off_value=0.0)
        refined_cls_loss = refined_cls_lw * tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=classes) 
        refined_cls_loss = tf.reduce_mean(refined_cls_loss) # * frac_
        tf.add_to_collection(tf.GraphKeys.LOSSES, refined_cls_loss)
        refined_cls_losses.append(refined_cls_loss)

        outputs['tmp_3'] = labels
        outputs['tmp_4'] = classes

        # outputs['tmp_0'] = outputs['ordered_rois']
        # outputs['tmp_1'] = outputs['pyramid_feature']
        # outputs['tmp_2'] = tf.transpose(outputs['roi']['cropped_rois'],[0,3,1,2])
        # outputs['tmp_3'] = outputs['assigned_rois']
        

        ### mask loss
        # mask of shape (N, h, w, num_classes)
        masks = outputs['mask']['mask']
        # mask_shape = tf.shape(masks)
        # masks = tf.reshape(masks, (mask_shape[0], mask_shape[1],
        #                            mask_shape[2], tf.cast(mask_shape[3]/2, tf.int32), 2))
        labels, mask_targets, mask_inside_weights = \
          mask_encoder(gt_masks, gt_boxes, ordered_rois, num_classes, 28, 28, scope='MaskEncoder')
        labels, masks, mask_targets, mask_inside_weights = \
                _filter_negative_samples(tf.reshape(labels, [-1]), [
                    tf.reshape(labels, [-1]),
                    masks,
                    mask_targets, 
                    mask_inside_weights, 
                    ])
        # _, frac_ = _get_valid_sample_fraction(labels)
        mask_batch.append(
                tf.reduce_sum(tf.cast(
                    tf.greater_equal(labels, 0), tf.float32
                    )))
        mask_batch_pos.append(
                tf.reduce_sum(tf.cast(
                    tf.greater_equal(labels, 1), tf.float32
                    )))
        # mask_targets = slim.one_hot_encoding(mask_targets, 2, on_value=1.0, off_value=0.0)
        # mask_binary_loss = mask_lw * tf.losses.softmax_cross_entropy(mask_targets, masks)
        # NOTE: w/o competition between classes. 
        mask_targets = tf.cast(mask_targets, tf.float32)
        mask_loss = mask_lw * tf.nn.sigmoid_cross_entropy_with_logits(labels=mask_targets, logits=masks) 
        mask_loss = tf.reduce_mean(mask_loss) 
        mask_loss = tf.cond(tf.greater(tf.size(labels), 0), lambda: mask_loss, lambda: tf.constant(0.0))
        tf.add_to_collection(tf.GraphKeys.LOSSES, mask_loss)
        mask_losses.append(mask_loss)

  rpn_box_losses = tf.add_n(rpn_box_losses)
  rpn_cls_losses = tf.add_n(rpn_cls_losses)
  refined_box_losses = tf.add_n(refined_box_losses)
  refined_cls_losses = tf.add_n(refined_cls_losses)
  mask_losses = tf.add_n(mask_losses)
  losses = [rpn_box_losses, rpn_cls_losses, refined_box_losses, refined_cls_losses, mask_losses]
  total_loss = tf.add_n(losses)

  rpn_batch = tf.cast(tf.add_n(rpn_batch), tf.float32)
  refine_batch = tf.cast(tf.add_n(refine_batch), tf.float32)
  mask_batch = tf.cast(tf.add_n(mask_batch), tf.float32)
  rpn_batch_pos = tf.cast(tf.add_n(rpn_batch_pos), tf.float32)
  refine_batch_pos = tf.cast(tf.add_n(refine_batch_pos), tf.float32)
  mask_batch_pos = tf.cast(tf.add_n(mask_batch_pos), tf.float32)
    
  return total_loss, losses, [rpn_batch_pos, rpn_batch, \
                              refine_batch_pos, refine_batch, \
                              mask_batch_pos, mask_batch]

def decode_output(outputs):
    """decode outputs into boxes and masks"""
    return [], [], []

def build(end_points, image_height, image_width, pyramid_map, 
        num_classes,
        base_anchors,
        is_training,
        gt_boxes,
        gt_masks, 
        loss_weights=[0.5, 0.5, 1.0, 0.5, 0.1]):
    
    pyramid = build_pyramid(pyramid_map, end_points)

    for p in pyramid:
        print (p)

    outputs = \
        build_heads(pyramid, image_height, image_width, num_classes, base_anchors, 
                    is_training=is_training, gt_boxes=gt_boxes)

    if is_training:
        loss, losses, batch_info = build_losses(pyramid, outputs, 
                        gt_boxes, gt_masks,
                        num_classes=num_classes, base_anchors=base_anchors,
                        rpn_box_lw=loss_weights[0], rpn_cls_lw=loss_weights[1],
                        refined_box_lw=loss_weights[2], refined_cls_lw=loss_weights[3],
                        mask_lw=loss_weights[4])

        outputs['losses'] = losses
        outputs['total_loss'] = loss
        outputs['batch_info'] = batch_info

    ## just decode outputs into readable prediction
    pred_boxes, pred_classes, pred_masks = decode_output(outputs)
    outputs['pred_boxes'] = pred_boxes
    outputs['pred_classes'] = pred_classes
    outputs['pred_masks'] = pred_masks

    # image and gt visualization
    visualize_input(gt_boxes, end_points["input"], tf.expand_dims(gt_masks, axis=3))

    # rpn visualization
    visualize_bb(end_points["input"], outputs['roi']["box"], name="rpn_bb_visualization")

    # final network visualization
    first_mask = outputs['mask']['mask'][:1]
    first_mask = tf.transpose(first_mask, [3, 1, 2, 0])

    visualize_final_predictions(outputs['final_boxes']["box"], end_points["input"], first_mask)

    return outputs
