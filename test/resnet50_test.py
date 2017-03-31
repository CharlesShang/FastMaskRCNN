#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import libs.configs.config_v1 as cfg
import libs.datasets.coco as coco
import libs.nets.resnet_v1 as resnet_v1
from libs.nets.train_utils import _configure_learning_rate, _configure_optimizer, \
  _get_variables_to_train, _get_init_fn

resnet50 = resnet_v1.resnet_v1_50
FLAGS = tf.app.flags.FLAGS

DEBUG = False

with tf.Graph().as_default():
  global_step = slim.create_global_step()
  
  ## data
  image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
    coco.read('./data/coco/records/coco_train2014_00000-of-00040.tfrecord')
  image = tf.cast(image[tf.newaxis, :, :, :], tf.float32)
  
  ##  network
  logits, end_points = resnet50(image, 1000, is_training=False)
  end_points['inputs'] = image
  
  for x in sorted(end_points.keys()):
    print (x, end_points[x].name, end_points[x].shape)
    
  import libs.nets.pyramid_network as pyramid_network
  pyramid = pyramid_network.build_pyramid('resnet50', end_points)
  for p in pyramid:
    print (p, pyramid[p])

  summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
  for p in pyramid:
    summaries.add(tf.summary.histogram('pyramid/hist/' + p, pyramid[p]))
    summaries.add(tf.summary.scalar('pyramid/means/'+ p, tf.reduce_mean(tf.abs(pyramid[p]))))
    
  outputs = pyramid_network.build_heads(pyramid, ih, iw, num_classes=81, base_anchors=15, is_training=True)
  
  if DEBUG:
    gt_boxes = np.random.randint(0, 200, (32, 2))
    shape = np.random.randint(10, 30, (32, 2))
    classes = np.random.randint(0, 81, (32,))
    gt_boxes = np.hstack((gt_boxes, gt_boxes + shape, classes[:, np.newaxis]))
    gt_boxes = gt_boxes.astype(np.float32)
    gt_masks = np.zeros((32, 224, 224), np.int32)
    gt_masks[:, 100:150, 100:150 ] = 1
  
  ## losses
  outputs = pyramid_network.build_losses(pyramid, outputs,
                                         gt_boxes, gt_masks,
                                         num_classes=81, base_anchors=15)

  ## optimization
  learning_rate = _configure_learning_rate(82783, global_step)
  optimizer = _configure_optimizer(learning_rate)
  summaries.add(tf.summary.scalar('learning_rate', learning_rate))
  for loss in tf.get_collection(tf.GraphKeys.LOSSES):
    summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
  learning_rate = _configure_learning_rate(82783, global_step)
  optimizer = _configure_optimizer(learning_rate)

  loss = tf.get_collection(tf.GraphKeys.LOSSES)
  regular_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n(loss + regular_loss)
  summaries.add(tf.summary.scalar('total_loss', total_loss))
  
  variables_to_train = _get_variables_to_train()
  gradients = optimizer.compute_gradients(total_loss, var_list=variables_to_train)
  grad_updates = optimizer.apply_gradients(gradients,
                                           global_step=global_step)

  update_op = tf.group(grad_updates)
  
  summary_op = tf.summary.merge(list(summaries), name='summary_op')
  
  
  
  sess = tf.Session()
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  
  sess.run(init_op)
  tf.train.start_queue_runners(sess=sess)
  ## training loop
  # with sess.as_default():
  #   npimage = image.eval()
  #   npih = ih.eval()
  #   npiw = iw.eval()
  #   npnum_instances = num_instances.eval()
  #   npgt_masks = gt_masks.eval()
  #   npgt_boxes = gt_boxes.eval()
  #
  #   print(img_id.eval())
  #
  #   # print(npimage)
  #   print(npgt_boxes)
  #   print(npih, npiw, npnum_instances)
    
  slim.learning.train(
    update_op,
    logdir=FLAGS.train_dir,
    init_fn=_get_init_fn(),
    summary_op=summary_op,
    number_of_steps=FLAGS.max_number_of_steps,
    log_every_n_steps=FLAGS.log_every_n_steps,
    save_summaries_secs=FLAGS.save_summaries_secs,
    save_interval_secs=FLAGS.save_interval_secs)