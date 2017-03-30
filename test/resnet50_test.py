#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.logs.log import LOG
import libs.configs.config_v1 as cfg
import libs.nets.resnet_v1 as resnet_v1
import libs.datasets.dataset_factory as dataset_factory
resnet50 = resnet_v1.resnet_v1_50
FLAGS = tf.app.flags.FLAGS

with tf.Graph().as_default():

  images = tf.placeholder(tf.float32, [1, 224, 224, 3], name='image')
  logits, end_points = resnet50(images, 1000, is_training=False)
  end_points['inputs'] = images
  
  for x in sorted(end_points.keys()):
    print (x, end_points[x].name, end_points[x].shape)
    
  import libs.nets.pyramid_network as pyramid_network
  pyramid = pyramid_network.build_pyramid('resnet50', end_points)
  for p in pyramid:
    print (p, pyramid[p])
  
  outputs = pyramid_network.build_heads(pyramid, num_classes=81, base_anchors=15, is_training=True)
  
  gt_boxes = np.random.randint(0, 200, (32, 2))
  shape = np.random.randint(10, 30, (32, 2))
  classes = np.random.randint(0, 81, (32,))
  gt_boxes = np.hstack((gt_boxes, gt_boxes + shape, classes[:, np.newaxis]))
  gt_masks = np.zeros((32, 224, 224), np.int32)
  gt_masks[:, 100:150, 100:150 ] = 1
  
  outputs = pyramid_network.build_losses(pyramid, outputs,
                                         gt_boxes.astype(np.float32), gt_masks.astype(np.float32),
                                         num_classes=81, base_anchors=15)
