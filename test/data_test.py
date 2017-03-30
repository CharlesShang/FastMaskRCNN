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
  dataset = dataset_factory.get_dataset(
    FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
  
  num_classes = dataset.num_classes
  
  # provider = slim.dataset_data_provider.DatasetDataProvider(
  #   dataset,
  #   num_readers=FLAGS.num_readers,
  #   common_queue_capacity=4,
  #   common_queue_min=2)
  # [image, label, gt_masks, gt_boxes, ih, iw] = provider.get(['image', 'label',
  #                                                            'gt_masks', 'gt_boxes',
  #                                                            'height', 'width'])

  image, ih, iw, gt_boxes, gt_masks, num_instances = \
    coco.read('./data/coco/records/coco_train2014_00000-of-00040.tfrecord')
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  
  with tf.Session()  as sess:
  
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    image = image.eval()
    ih = ih.eval()
    iw = iw.eval()
    num_instances = num_instances.eval()
    # print (image)
    print (ih, iw, num_instances)
    gt_masks = gt_masks.eval()
    gt_boxes = gt_boxes.eval()
    print(gt_boxes)
    
    