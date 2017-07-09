#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import functools
import os, sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import gmtime, strftime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libs.configs.config_v1 as cfg
import libs.datasets.dataset_factory as datasets
import libs.nets.nets_factory as network 

import libs.preprocessings.coco_v1 as coco_preprocess
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1

from train.train_utils import _configure_learning_rate, _configure_optimizer, \
  _get_variables_to_train, _get_init_fn, get_var_list_to_restore

from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from libs.datasets import download_and_convert_coco
#from libs.datasets.download_and_convert_coco import _cat_id_to_cls_name
from libs.visualization.pil_utils import cat_id_to_cls_name, draw_img, draw_bbox

FLAGS = tf.app.flags.FLAGS
resnet50 = resnet_v1.resnet_v1_50

def solve(global_step):
    """add solver to losses"""
    # learning reate
    lr = _configure_learning_rate(82783, global_step)
    optimizer = _configure_optimizer(lr)
    tf.summary.scalar('learning_rate', lr)

    # compute and apply gradient
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    regular_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regular_loss = tf.add_n(regular_losses)
    out_loss = tf.add_n(losses)
    total_loss = tf.add_n(losses + regular_losses)

    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('out_loss', out_loss)
    tf.summary.scalar('regular_loss', regular_loss)

    update_ops = []
    variables_to_train = _get_variables_to_train()
    # update_op = optimizer.minimize(total_loss)
    gradients = optimizer.compute_gradients(total_loss, var_list=variables_to_train)
    grad_updates = optimizer.apply_gradients(gradients, 
            global_step=global_step)
    update_ops.append(grad_updates)
    
    # update moving mean and variance
    if FLAGS.update_bn:
        update_bns = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_bn = tf.group(*update_bns)
        update_ops.append(update_bn)

    return tf.group(*update_ops)

def restore(sess):
     """choose which param to restore"""
     if FLAGS.restore_previous_if_exists:
        try:
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            ###########
            restorer = tf.train.Saver()
            ###########

            ###########
            # not_restore = [ 'pyramid/fully_connected/weights:0', 
            #                 'pyramid/fully_connected/biases:0',
            #                 'pyramid/fully_connected/weights:0', 
            #                 'pyramid/fully_connected_1/biases:0',
            #                 'pyramid/fully_connected_1/weights:0', 
            #                 'pyramid/fully_connected_2/weights:0', 
            #                 'pyramid/fully_connected_2/biases:0',
            #                 'pyramid/fully_connected_3/weights:0', 
            #                 'pyramid/fully_connected_3/biases:0',
            #                 'pyramid/Conv/weights:0', 
            #                 'pyramid/Conv/biases:0',
            #                 'pyramid/Conv_1/weights:0', 
            #                 'pyramid/Conv_1/biases:0', 
            #                 'pyramid/Conv_2/weights:0', 
            #                 'pyramid/Conv_2/biases:0', 
            #                 'pyramid/Conv_3/weights:0', 
            #                 'pyramid/Conv_3/biases:0',
            #                 'pyramid/Conv2d_transpose/weights:0', 
            #                 'pyramid/Conv2d_transpose/biases:0', 
            #                 'pyramid/Conv_4/weights:0',
            #                 'pyramid/Conv_4/biases:0',
            #                 'pyramid/fully_connected/weights/Momentum:0', 
            #                 'pyramid/fully_connected/biases/Momentum:0',
            #                 'pyramid/fully_connected/weights/Momentum:0', 
            #                 'pyramid/fully_connected_1/biases/Momentum:0',
            #                 'pyramid/fully_connected_1/weights/Momentum:0', 
            #                 'pyramid/fully_connected_2/weights/Momentum:0', 
            #                 'pyramid/fully_connected_2/biases/Momentum:0',
            #                 'pyramid/fully_connected_3/weights/Momentum:0', 
            #                 'pyramid/fully_connected_3/biases/Momentum:0',
            #                 'pyramid/Conv/weights/Momentum:0', 
            #                 'pyramid/Conv/biases/Momentum:0',
            #                 'pyramid/Conv_1/weights/Momentum:0', 
            #                 'pyramid/Conv_1/biases/Momentum:0', 
            #                 'pyramid/Conv_2/weights/Momentum:0', 
            #                 'pyramid/Conv_2/biases/Momentum:0', 
            #                 'pyramid/Conv_3/weights/Momentum:0', 
            #                 'pyramid/Conv_3/biases/Momentum:0',
            #                 'pyramid/Conv2d_transpose/weights/Momentum:0', 
            #                 'pyramid/Conv2d_transpose/biases/Momentum:0', 
            #                 'pyramid/Conv_4/weights/Momentum:0',
            #                 'pyramid/Conv_4/biases/Momentum:0',]
            # vars_to_restore = [v for v in  tf.all_variables()if v.name not in not_restore]
            # restorer = tf.train.Saver(vars_to_restore)
            # for var in vars_to_restore:
            #     print ('restoring ', var.name)
            ############

            restorer.restore(sess, checkpoint_path)
            print ('restored previous model %s from %s'\
                    %(checkpoint_path, FLAGS.train_dir))
            time.sleep(2)
            return
        except:
            print ('--restore_previous_if_exists is set, but failed to restore in %s %s'\
                    % (FLAGS.train_dir, checkpoint_path))
            time.sleep(2)

     if FLAGS.pretrained_model:
        if tf.gfile.IsDirectory(FLAGS.pretrained_model):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_model)
        else:
            checkpoint_path = FLAGS.pretrained_model

        if FLAGS.checkpoint_exclude_scopes is None:
            FLAGS.checkpoint_exclude_scopes='pyramid'
        if FLAGS.checkpoint_include_scopes is None:
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
    
def train():
    """The main function that runs training"""

    ## data
    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
        datasets.get_dataset(FLAGS.dataset_name, 
                             FLAGS.dataset_split_name, 
                             FLAGS.dataset_dir, 
                             FLAGS.im_batch,
                             is_training=True)

    data_queue = tf.RandomShuffleQueue(capacity=32, min_after_dequeue=16,
            dtypes=(
                image.dtype, ih.dtype, iw.dtype, 
                gt_boxes.dtype, gt_masks.dtype, 
                num_instances.dtype, img_id.dtype)) 
    enqueue_op = data_queue.enqueue((image, ih, iw, gt_boxes, gt_masks, num_instances, img_id))
    data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
    (image, ih, iw, gt_boxes, gt_masks, num_instances, img_id) =  data_queue.dequeue()
    im_shape = tf.shape(image)
    image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], 3))

    ## network
    logits, end_points, pyramid_map = network.get_network(FLAGS.network, image,
            weight_decay=FLAGS.weight_decay, is_training=True)
    outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map,
            num_classes=81,
            base_anchors=9,
            is_training=True,
            gt_boxes=gt_boxes, gt_masks=gt_masks,
            loss_weights=[0.2, 0.2, 1.0, 0.2, 1.0])


    total_loss = outputs['total_loss']
    losses  = outputs['losses']
    batch_info = outputs['batch_info']
    regular_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    
    input_image = end_points['input']
    final_box = outputs['final_boxes']['box']
    final_cls = outputs['final_boxes']['cls']
    final_prob = outputs['final_boxes']['prob']
    final_gt_cls = outputs['final_boxes']['gt_cls']
    gt = outputs['gt']

    #############################
    tmp_0 = outputs['losses']
    tmp_1 = outputs['losses']
    tmp_2 = outputs['losses']
    tmp_3 = outputs['losses']
    tmp_4 = outputs['losses']

    # tmp_0 = outputs['tmp_0']
    # tmp_1 = outputs['tmp_1']
    # tmp_2 = outputs['tmp_2']
    tmp_3 = outputs['tmp_3']
    tmp_4 = outputs['tmp_4']
    ############################


    ## solvers
    global_step = slim.create_global_step()
    update_op = solve(global_step)

    cropped_rois = tf.get_collection('__CROPPED__')[0]
    transposed = tf.get_collection('__TRANSPOSED__')[0]
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
            )
    sess.run(init_op)

    summary_op = tf.summary.merge_all()
    logdir = os.path.join(FLAGS.train_dir, strftime('%Y%m%d%H%M%S', gmtime()))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

    ## restore
    restore(sess)

    ## main loop
    coord = tf.train.Coordinator()
    threads = []
    # print (tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

    tf.train.start_queue_runners(sess=sess, coord=coord)
    saver = tf.train.Saver(max_to_keep=20)

    for step in range(FLAGS.max_iters):
        
        start_time = time.time()

        s_, tot_loss, reg_lossnp, img_id_str, \
        rpn_box_loss, rpn_cls_loss, refined_box_loss, refined_cls_loss, mask_loss, \
        gt_boxesnp, \
        rpn_batch_pos, rpn_batch, refine_batch_pos, refine_batch, mask_batch_pos, mask_batch, \
        input_imagenp, final_boxnp, final_clsnp, final_probnp, final_gt_clsnp, gtnp, tmp_0np, tmp_1np, tmp_2np, tmp_3np, tmp_4np= \
                     sess.run([update_op, total_loss, regular_loss, img_id] + 
                              losses + 
                              [gt_boxes] + 
                              batch_info + 
                              [input_image] + [final_box] + [final_cls] + [final_prob] + [final_gt_cls] + [gt] + [tmp_0] + [tmp_1] + [tmp_2] + [tmp_3] + [tmp_4])

        duration_time = time.time() - start_time
        if step % 1 == 0: 
            print ( """iter %d: image-id:%07d, time:%.3f(sec), regular_loss: %.6f, """
                    """total-loss %.4f(%.4f, %.4f, %.6f, %.4f, %.4f), """
                    """instances: %d, """
                    """batch:(%d|%d, %d|%d, %d|%d)""" 
                   % (step, img_id_str, duration_time, reg_lossnp, 
                      tot_loss, rpn_box_loss, rpn_cls_loss, refined_box_loss, refined_cls_loss, mask_loss,
                      gt_boxesnp.shape[0], 
                      rpn_batch_pos, rpn_batch, refine_batch_pos, refine_batch, mask_batch_pos, mask_batch))

            # draw_bbox(step, 
            #           np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
            #           name='est', 
            #           bbox=final_boxnp, 
            #           label=final_clsnp, 
            #           prob=final_probnp,
            #           gt_label=np.argmax(np.asarray(final_gt_clsnp),axis=1),
            #           )

            # draw_bbox(step, 
            #           np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
            #           name='gt', 
            #           bbox=gtnp[:,0:4], 
            #           label=np.asarray(gtnp[:,4], dtype=np.uint8),
            #           )
            
            print ("labels")
            # print (cat_id_to_cls_name(np.unique(np.argmax(np.asarray(final_gt_clsnp),axis=1)))[1:])
            # print (cat_id_to_cls_name(np.unique(np.asarray(gt_boxesnp, dtype=np.uint8)[:,4])))
            print (cat_id_to_cls_name(np.unique(np.argmax(np.asarray(tmp_3np),axis=1)))[1:])
            #print (cat_id_to_cls_name(np.unique(np.argmax(np.asarray(gt_boxesnp)[:,4],axis=1))))
            print ("classes")
            print (cat_id_to_cls_name(np.unique(np.argmax(np.array(tmp_4np),axis=1))))
            # print (np.asanyarray(tmp_3np))

            #print ("ordered rois")
            #print (np.asarray(tmp_0np)[0])
            #print ("pyramid_feature")
            #print ()
             #print(np.unique(np.argmax(np.array(final_probnp),axis=1)))
            #for var, val in zip(tmp_2, tmp_2np):
            #    print(var.name)  
            #print(np.argmax(np.array(tmp_0np),axis=1))
            
            
            if np.isnan(tot_loss) or np.isinf(tot_loss):
                print (gt_boxesnp)
                raise
          
        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        if (step % 10000 == 0 or step + 1 == FLAGS.max_iters) and step != 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 
                                           FLAGS.dataset_name + '_' + FLAGS.network + '_model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

        if coord.should_stop():
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train()
