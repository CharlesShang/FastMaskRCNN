# #!/usr/bin/env python
# # coding=utf-8
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
 
# import functools
# import os, sys
# import time
# import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
# from time import gmtime, strftime

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# import libs.configs.config_v1 as cfg
# import libs.datasets.dataset_factory as datasets
# import libs.nets.nets_factory as network 

# import libs.preprocessings.coco_v1 as coco_preprocess
# import libs.nets.pyramid_network as pyramid_network
# import libs.nets.resnet_v1 as resnet_v1

# from train.train_utils import _configure_learning_rate, _configure_optimizer, \
#   _get_variables_to_train, _get_init_fn, get_var_list_to_restore

# from PIL import Image, ImageFont, ImageDraw, ImageEnhance
# from libs.datasets import download_and_convert_coco
# from libs.visualization.pil_utils import cat_id_to_cls_name, draw_img, draw_bbox

# FLAGS = tf.app.flags.FLAGS
# resnet50 = resnet_v1.resnet_v1_50

# def solve(global_step):
#     """add solver to losses"""
#     # learning reate
#     lr = _configure_learning_rate(82783, global_step)
#     optimizer = _configure_optimizer(lr)
#     tf.summary.scalar('learning_rate', 0.0)

#     # compute and apply gradient
#     losses = tf.get_collection(tf.GraphKeys.LOSSES)
#     regular_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#     regular_loss = tf.add_n(regular_losses)
#     out_loss = tf.add_n(losses)
#     total_loss = tf.add_n(losses + regular_losses)

#     tf.summary.scalar('total_loss', total_loss)
#     tf.summary.scalar('out_loss', out_loss)
#     tf.summary.scalar('regular_loss', regular_loss)

#     update_ops = []
#     variables_to_train = _get_variables_to_train()
#     # update_op = optimizer.minimize(total_loss)
#     gradients = optimizer.compute_gradients(total_loss, var_list=variables_to_train)
#     grad_updates = optimizer.apply_gradients(gradients, 
#             global_step=global_step)
#     update_ops.append(grad_updates)
    
#     # update moving mean and variance
#     if FLAGS.update_bn:
#         update_bns = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         update_bn = tf.group(*update_bns)
#         update_ops.append(update_bn)

#     return tf.group(*update_ops)

# def restore(sess):
#     """choose which param to restore"""
#     if FLAGS.restore_previous_if_exists:
#         try:
#             checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
#             ###########
#             restorer = tf.train.Saver()
#             ###########

#             ###########
#             # not_restore = [ 'pyramid/fully_connected/weights:0', 
#             #                 'pyramid/fully_connected/biases:0',
#             #                 'pyramid/fully_connected/weights:0', 
#             #                 'pyramid/fully_connected_1/biases:0',
#             #                 'pyramid/fully_connected_1/weights:0', 
#             #                 'pyramid/fully_connected_2/weights:0', 
#             #                 'pyramid/fully_connected_2/biases:0',
#             #                 'pyramid/fully_connected_3/weights:0', 
#             #                 'pyramid/fully_connected_3/biases:0',
#             #                 'pyramid/Conv/weights:0', 
#             #                 'pyramid/Conv/biases:0',
#             #                 'pyramid/Conv_1/weights:0', 
#             #                 'pyramid/Conv_1/biases:0', 
#             #                 'pyramid/Conv_2/weights:0', 
#             #                 'pyramid/Conv_2/biases:0', 
#             #                 'pyramid/Conv_3/weights:0', 
#             #                 'pyramid/Conv_3/biases:0',
#             #                 'pyramid/Conv2d_transpose/weights:0', 
#             #                 'pyramid/Conv2d_transpose/biases:0', 
#             #                 'pyramid/Conv_4/weights:0',
#             #                 'pyramid/Conv_4/biases:0',
#             #                 'pyramid/fully_connected/weights/Momentum:0', 
#             #                 'pyramid/fully_connected/biases/Momentum:0',
#             #                 'pyramid/fully_connected/weights/Momentum:0', 
#             #                 'pyramid/fully_connected_1/biases/Momentum:0',
#             #                 'pyramid/fully_connected_1/weights/Momentum:0', 
#             #                 'pyramid/fully_connected_2/weights/Momentum:0', 
#             #                 'pyramid/fully_connected_2/biases/Momentum:0',
#             #                 'pyramid/fully_connected_3/weights/Momentum:0', 
#             #                 'pyramid/fully_connected_3/biases/Momentum:0',
#             #                 'pyramid/Conv/weights/Momentum:0', 
#             #                 'pyramid/Conv/biases/Momentum:0',
#             #                 'pyramid/Conv_1/weights/Momentum:0', 
#             #                 'pyramid/Conv_1/biases/Momentum:0', 
#             #                 'pyramid/Conv_2/weights/Momentum:0', 
#             #                 'pyramid/Conv_2/biases/Momentum:0', 
#             #                 'pyramid/Conv_3/weights/Momentum:0', 
#             #                 'pyramid/Conv_3/biases/Momentum:0',
#             #                 'pyramid/Conv2d_transpose/weights/Momentum:0', 
#             #                 'pyramid/Conv2d_transpose/biases/Momentum:0', 
#             #                 'pyramid/Conv_4/weights/Momentum:0',
#             #                 'pyramid/Conv_4/biases/Momentum:0',]
#             # vars_to_restore = [v for v in  tf.all_variables()if v.name not in not_restore]
#             # restorer = tf.train.Saver(vars_to_restore)
#             # for var in vars_to_restore:
#             #     print ('restoring ', var.name)
#             ############

#             restorer.restore(sess, checkpoint_path)
#             print ('restored previous model %s from %s'\
#                     %(checkpoint_path, FLAGS.train_dir))
#             time.sleep(2)
#             return
#         except:
#             print ('--restore_previous_if_exists is set, but failed to restore in %s %s'\
#                     % (FLAGS.train_dir, checkpoint_path))
#             time.sleep(2)

#     if FLAGS.pretrained_model:
#         if tf.gfile.IsDirectory(FLAGS.pretrained_model):
#             checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_model)
#         else:
#             checkpoint_path = FLAGS.pretrained_model

#         if FLAGS.checkpoint_exclude_scopes is None:
#             FLAGS.checkpoint_exclude_scopes='pyramid'
#         if FLAGS.checkpoint_include_scopes is None:
#             FLAGS.checkpoint_include_scopes='resnet_v1_50'

#         vars_to_restore = get_var_list_to_restore()
#         for var in vars_to_restore:
#             print ('restoring ', var.name)
      
#         try:
#            restorer = tf.train.Saver(vars_to_restore)
#            restorer.restore(sess, checkpoint_path)
#            print ('Restored %d(%d) vars from %s' %(
#                len(vars_to_restore), len(tf.global_variables()),
#                checkpoint_path ))
#         except:
#            print ('Checking your params %s' %(checkpoint_path))
#            raise
    
# def train():
#     """The main function that runs training"""

#     ## data
#     image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
#         datasets.get_dataset(FLAGS.dataset_name, 
#                              FLAGS.dataset_split_name, 
#                              FLAGS.dataset_dir, 
#                              FLAGS.im_batch,
#                              is_training=False)

#     data_queue = tf.RandomShuffleQueue(capacity=32, min_after_dequeue=16,
#             dtypes=(
#                 image.dtype, ih.dtype, iw.dtype, 
#                 gt_boxes.dtype, gt_masks.dtype, 
#                 num_instances.dtype, img_id.dtype)) 
#     enqueue_op = data_queue.enqueue((image, ih, iw, gt_boxes, gt_masks, num_instances, img_id))
#     data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
#     tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
#     (image, ih, iw, gt_boxes, gt_masks, num_instances, img_id) =  data_queue.dequeue()
#     im_shape = tf.shape(image)
#     image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], 3))

#     ## network
#     logits, end_points, pyramid_map = network.get_network(FLAGS.network, image,
#             weight_decay=FLAGS.weight_decay, is_training=False)
#     outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map,
#             num_classes=81,
#             base_anchors=9,
#             is_training=False,
#             gt_boxes=gt_boxes, gt_masks=gt_masks,
#             )

#     input_image = end_points['input']
#     final_box = outputs['final_boxes']['box']
#     final_cls = outputs['final_boxes']['cls']
#     final_prob = outputs['final_boxes']['prob']
#     final_rpn_box = outputs['final_boxes']['rpn_box']
#     final_mask = outputs['mask']['final_mask']

#     #############################
#     tmp_0 = outputs['mask']['final_mask']
#     tmp_1 = outputs['mask']['final_mask']
#     tmp_2 = outputs['mask']['final_mask']
#     tmp_3 = outputs['mask']['final_mask']
#     tmp_4 = outputs['mask']['final_mask']

#     # tmp_0 = outputs['tmp_0']
#     # tmp_1 = outputs['tmp_1']
#     # tmp_2 = outputs['tmp_2']
#     # tmp_3 = outputs['tmp_3']
#     # tmp_4 = outputs['tmp_4']
#     ############################


#     ## solvers
#     global_step = slim.create_global_step()

#     cropped_rois = tf.get_collection('__CROPPED__')[0]
#     transposed = tf.get_collection('__TRANSPOSED__')[0]
    
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#     init_op = tf.group(
#             tf.global_variables_initializer(),
#             tf.local_variables_initializer()
#             )
#     sess.run(init_op)

#     summary_op = tf.summary.merge_all()
#     logdir = os.path.join(FLAGS.train_dir, strftime('%Y%m%d%H%M%S', gmtime()))
#     if not os.path.exists(logdir):
#         os.makedirs(logdir)
#     summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

#     ## restore
#     restore(sess)

#     ## main loop
#     coord = tf.train.Coordinator()
#     threads = []
#     # print (tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
#     for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
#         threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
#                                          start=True))

#     tf.train.start_queue_runners(sess=sess, coord=coord)
#     saver = tf.train.Saver(max_to_keep=20)

#     for step in range(FLAGS.max_iters):
        
#         start_time = time.time()

#         img_id_str, \
#         gt_boxesnp, \
#         input_imagenp, final_boxnp, final_clsnp, final_probnp, final_rpn_boxnp, final_masknp, tmp_0np, tmp_1np, tmp_2np, tmp_3np, tmp_4np= \
#                      sess.run([img_id] + 
#                               [gt_boxes] + 
#                               [input_image] + [final_box] + [final_cls] + [final_prob] + [final_rpn_box] + [final_mask] + [tmp_0] + [tmp_1] + [tmp_2] + [tmp_3] + [tmp_4])

#         duration_time = time.time() - start_time

#         if step % 1 == 0: 
#             print ( """iter %d: image-id:%07d, time:%.3f(sec), """
#                     """instances: %d, """
                    
#                    % (step, img_id_str, duration_time, 
#                       gt_boxesnp.shape[0]))

#             draw_bbox(step, 
#                       np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
#                       name='test_est', 
#                       bbox=final_boxnp, 
#                       label=final_clsnp, 
#                       prob=final_probnp,
#                       mask=final_masknp,
#                       vis_all=True
#                       )

#             # draw_bbox(step, 
#             #           np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
#             #           name='train_roi', 
#             #           bbox=final_rpn_boxnp, 
#             #           label=final_clsnp, 
#             #           prob=final_probnp,
#             #           gt_label=np.argmax(np.asarray(final_gt_clsnp),axis=1),
#             #           iou=final_max_overlapsnp
#             #           )

#             # draw_bbox(step, 
#             #           np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
#             #           name='train_msk', 
#             #           bbox=tmp_0np, 
#             #           label=tmp_2np, 
#             #           prob=np.zeros((tmp_2np.shape[0],81), dtype=np.float32)+1.0,
#             #           mask=tmp_1np,
#             #           vis_all=True
#             #           )

#             # draw_bbox(step, 
#             #           np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
#             #           name='train_gt', 
#             #           bbox=gtnp[:,0:4], 
#             #           label=np.asarray(gtnp[:,4], dtype=np.uint8),
#             #           )
            
#             # print ("labels")
#             # print (cat_id_to_cls_name(np.unique(np.argmax(np.asarray(tmp_3np),axis=1)))[1:])
#             # print ("classes")
#             # print (cat_id_to_cls_name(np.unique(np.argmax(np.array(tmp_4np),axis=1))))
            

#         if coord.should_stop():
#             coord.request_stop()
#             coord.join(threads)


# if __name__ == '__main__':
#     train()







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
    
def test():
    """The main function that runs training"""

    ## data
    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
        datasets.get_dataset(FLAGS.dataset_name, 
                             FLAGS.dataset_split_name, 
                             FLAGS.dataset_dir, 
                             FLAGS.im_batch,
                             is_training=False)

    # data_queue = tf.RandomShuffleQueue(capacity=32, min_after_dequeue=16,
    #         dtypes=(
    #             image.dtype, ih.dtype, iw.dtype, 
    #             gt_boxes.dtype, gt_masks.dtype, 
    #             num_instances.dtype, img_id.dtype)) 
    # enqueue_op = data_queue.enqueue((image, ih, iw, gt_boxes, gt_masks, num_instances, img_id))
    # data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
    # tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
    # (image, ih, iw, gt_boxes, gt_masks, num_instances, img_id) =  data_queue.dequeue()
    im_shape = tf.shape(image)
    image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], 3))

    ## network
    logits, end_points, pyramid_map = network.get_network(FLAGS.network, image,
            weight_decay=FLAGS.weight_decay, is_training=True)
    outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map,
            num_classes=81,
            base_anchors=9,
            is_training=True,
            gt_boxes=gt_boxes, gt_masks=gt_masks,)

    input_image = end_points['input']
    final_box = outputs['final_boxes']['box']
    final_cls = outputs['final_boxes']['cls']
    final_prob = outputs['final_boxes']['prob']
    final_rpn_box = outputs['final_boxes']['rpn_box']
    final_mask = outputs['mask']['mask']

    #############################
    tmp_0 = outputs['mask']['mask']
    tmp_1 = outputs['mask']['mask']
    tmp_2 = outputs['mask']['mask']
    tmp_3 = outputs['mask']['mask']
    tmp_4 = outputs['mask']['mask']

    # tmp_0 = outputs['tmp_0']
    # tmp_1 = outputs['tmp_1']
    # tmp_2 = outputs['tmp_2']
    # tmp_3 = outputs['tmp_3']
    # tmp_4 = outputs['tmp_4']
    ############################


    ## solvers
    global_step = slim.create_global_step()
    #update_op = solve(global_step)

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

        img_id_str, \
        gt_boxesnp, \
        input_imagenp, final_boxnp, final_clsnp, final_probnp, final_rpn_boxnp, final_masknp, tmp_0np, tmp_1np, tmp_2np, tmp_3np, tmp_4np= \
                     sess.run([img_id] + 
                              [gt_boxes] + 
                              [input_image] + [final_box] + [final_cls] + [final_prob] + [final_rpn_box] + [final_mask] + [tmp_0] + [tmp_1] + [tmp_2] + [tmp_3] + [tmp_4])

        duration_time = time.time() - start_time
        if step % 1 == 0: 
            print ( """iter %d: image-id:%07d, time:%.3f(sec), """
                    """instances: %d, """
                    
                   % (step, img_id_str, duration_time, 
                      gt_boxesnp.shape[0]))

            # print("tmp")
            # print(np.asarray(tmp_0np))
            # print(np.asarray(tmp_1np))
            # print(np.asarray(tmp_2np))
            # print(np.asarray(tmp_3np))

            # print ("labels")    
            # print (cat_id_to_cls_name(np.unique(np.argmax(np.asarray(tmp_3np),axis=1)))[1:])
            # print ("classes")
            # print (cat_id_to_cls_name(np.unique(np.argmax(np.array(tmp_4np),axis=1))))

            #print ("iw", np.asanyarray(tmp_4np))
            #if np.asarray(tmp_3np[3]).shape[0]>=1:
                #print ("ordered_rois")
                #print (np.asarray(tmp_0np)[0])
                #print ("pyramid_feature")
                #print ("p5",np.asarray(tmp_1np[0]).shape)
                #print (np.asarray(tmp_1np[0][0][0]))

                #print ("real_pyramid")
                #print (np.asarray(tmp_4np).shape)
                #print (np.asarray(tmp_4np)[0][0])
                #print ("p4",np.asanyarray(tmp_1np[1]).shape)
                #print ("p3",np.asanyarray(tmp_1np[2]).shape)
                #print ("p2",np.asanyarray(tmp_1np[3]).shape)

                #print ("cropped_rois")
                #print (np.asarray(tmp_2np).shape)
                #print (np.asarray(tmp_2np)[0][0])
                # print ("assigned_layer_num")
                # print ("p5:",np.asarray(tmp_3np[3]).shape[0])
                # print ("p4:",np.asarray(tmp_3np[2]).shape[0])
                # print ("p3:",np.asarray(tmp_3np[1]).shape[0])
                # print ("p2:",np.asarray(tmp_3np[0]).shape[0])
        if step % 1 == 0: 
            draw_bbox(step, 
                      np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
                      name='test_est', 
                      bbox=final_boxnp, 
                      label=final_clsnp, 
                      prob=final_probnp,
                      mask=final_masknp,)

            draw_bbox(step, 
                      np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
                      name='test_roi', 
                      bbox=final_boxnp, 
                      label=final_clsnp, 
                      prob=final_probnp,
                      )
            # print ("boxes")
            # print (np.asarray(final_boxnp).shape)
            # print ("classes")
            # print (cat_id_to_cls_name(np.unique(np.asarray(final_clsnp))))
            #print (cat_id_to_cls_name(np.unique(np.argmax(np.array(final_clsnp),axis=1))))
            


if __name__ == '__main__':
    test()
