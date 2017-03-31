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
import libs.datasets.coco as coco
resnet50 = resnet_v1.resnet_v1_50
FLAGS = tf.app.flags.FLAGS

with tf.Graph().as_default():

  image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
    coco.read('./data/coco/records-back/coco_train2014_00000-of-00040.tfrecord')

  sess = tf.Session()
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  tf.train.start_queue_runners(sess=sess)
  sess.run(init_op)
  with sess.as_default():
    npimage = image.eval()
    npih = ih.eval()
    npiw = iw.eval()
    npnum_instances = num_instances.eval()
    npgt_masks = gt_masks.eval()
    npgt_boxes = gt_boxes.eval()
    
    print (img_id.eval())

    # print(npimage)
    print(npgt_boxes)
    print(npih, npiw, npnum_instances)