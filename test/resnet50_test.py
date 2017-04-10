#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import os, sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from time import gmtime, strftime
import tensorflow as tf
import tensorflow.contrib.slim as slim
import libs.configs.config_v1 as cfg
import libs.datasets.coco as coco
import libs.preprocessings.coco_v1 as coco_preprocess
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1
from libs.nets.train_utils import _configure_learning_rate, _configure_optimizer, \
  _get_variables_to_train, _get_init_fn, get_var_list_to_restore

resnet50 = resnet_v1.resnet_v1_50
FLAGS = tf.app.flags.FLAGS

DEBUG = False

with tf.Graph().as_default():
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, 
                              allow_growth=True,
                              )
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                        allow_soft_placement=True)) as sess:
      global_step = slim.create_global_step()
      
      ## data
      image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
        coco.read('./data/coco/records/coco_train2014_00000-of-00040.tfrecord')
      image, gt_boxes, gt_masks = coco_preprocess.preprocess_image(image, gt_boxes, gt_masks, is_training=True)
      
      ##  network
      with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, end_points = resnet50(image, 1000, is_training=False)
      end_points['inputs'] = image
      
      for x in sorted(end_points.keys()):
        print (x, end_points[x].name, end_points[x].shape)
        
      pyramid = pyramid_network.build_pyramid('resnet50', end_points)
      # for p in pyramid:
      #   print (p, pyramid[p])

      summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
      for p in pyramid:
        summaries.add(tf.summary.histogram('pyramid/hist/' + p, pyramid[p]))
        summaries.add(tf.summary.scalar('pyramid/means/'+ p, tf.reduce_mean(tf.abs(pyramid[p]))))
        
      outputs = pyramid_network.build_heads(pyramid, ih, iw, num_classes=81, base_anchors=15, is_training=True)
      
      
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

      loss = tf.get_collection(tf.GraphKeys.LOSSES)
      regular_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      total_loss = tf.add_n(loss + regular_loss)
      summaries.add(tf.summary.scalar('total_loss', total_loss))
      
      variables_to_train = _get_variables_to_train()
      update_op = optimizer.minimize(total_loss)
      # gradients = optimizer.compute_gradients(total_loss, var_list=variables_to_train)
      # grad_updates = optimizer.apply_gradients(gradients,
      #                                          global_step=global_step)
      # update_op = tf.group(grad_updates)
      
      # summary_op = tf.summary.merge(list(summaries), name='summary_op')
      summary_op = tf.summary.merge_all()
      logdir = os.path.join(FLAGS.train_dir, strftime('%Y%m%d%H%M%S', gmtime()))
      if not os.path.exists(logdir):
        os.makedirs(logdir)
      summary_writer = tf.summary.FileWriter(
            logdir,
            graph=sess.graph)
      
      
      init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
      
      sess.run(init_op)
      tf.train.start_queue_runners(sess=sess)

      ## restore pretrained model
      # FLAGS.pretrained_model = None
      if FLAGS.pretrained_model:
          if tf.gfile.IsDirectory(FLAGS.pretrained_model):
              checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_model)
          else:
              checkpoint_path = FLAGS.pretrained_model
          FLAGS.checkpoint_exclude_scopes='pyramid'
          FLAGS.checkpoint_include_scopes='resnet_v1_50'
          vars_to_restore = get_var_list_to_restore()
          for var in vars_to_restore:
              print ('restoring ', var.name)
          
          try:
              restorer = tf.train.Saver(vars_to_restore)
              restorer.restore(sess, checkpoint_path)
              print ('Restored %d(%d) vars from %s' %(
                  len(vars_to_restore), len(tf.global_variables()),
                  checkpoint_path ))
          except:
              print ('Checking your params %s' %(checkpoint_path))
              raise
      
      # import libs.memory_util as memory_util
      # memory_util.vlog(1)
      # with memory_util.capture_stderr() as stderr:
      #     sess.run([update_op, total_loss])
      # memory_util.print_memory_timeline(stderr, ignore_less_than_bytes=1000)
      
      ## training loop
      print ('Starting training...')
      # with sess.as_default():  
      for step in range(FLAGS.max_iters):
        start_time = time.time()
        # summary_str, _, tot_loss = \
        #             sess.run([summary_op, update_op, total_loss])
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run([update_op, total_loss], 
                            options=run_options, 
                            run_metadata=run_metadata)
        # tot_loss = sess.run(update_op)
        duration_time = time.time() - start_time
        if step % 10 == 0: 
          print ('iter %d: image-id:%07d, time:%.3f(sec)' %(step, 0, duration_time))
        # if step % 100 == 0:
        #   summary_str = sess.run([summary_op], options=run_options, run_metadata=run_metadata)
        #   summary_writer.add_summary(summary_str, step)

        if (step % 10000 == 0 or step + 1 == FLAGS.max_iters) and step != 0:
          checkpoint_path = os.path.join(FLAGS.train_dir, 
                                         FLAGS.dataset_name + '_model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
