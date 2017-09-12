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
import json
from time import gmtime, strftime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libs.configs.config_v1 as cfg
import libs.datasets.dataset_factory as datasets
import libs.nets.nets_factory as network 

import libs.preprocessings.coco_v1 as coco_preprocess
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1
import libs.boxes.cython_bbox as cython_bbox

from train.train_utils import _configure_learning_rate, _configure_optimizer, \
  _get_variables_to_train, _get_init_fn, get_var_list_to_restore

from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from libs.datasets import download_and_convert_coco
from libs.visualization.pil_utils import cat_id_to_cls_name, draw_img, draw_bbox

FLAGS = tf.app.flags.FLAGS
resnet50 = resnet_v1.resnet_v1_50

def _cat_id_to_real_id(readId):
  """Note coco has 80 classes, but the catId ranges from 1 to 90!"""
  cat_id_to_real_id = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                                10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 
                                21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 
                                34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 
                                44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 
                                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 
                                65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 
                                79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 
                                90,])
  return cat_id_to_real_id[readId]

def _writeJSON(_dict): 
    with open(FLAGS.train_dir + 'results.json', 'a+') as f:
        f.seek(0,2)              #Go to the end of file    
        if f.tell() == 0 :       #Check if file is empty
            json.dump([_dict], f)  #If empty, write an array
        else :
            f.seek(-1,2)           
            f.truncate()           #Remove the last character, open the array
            f.write(' , ')         #Write the separator
            json.dump(_dict,f)     #Dump the dictionary
            f.write(']')           #Close the array
        f.close()
        return

def _convertBoxes(boxes, img_h, img_w, new_img_h, new_img_w):
    new_boxes = boxes
    new_boxes[:,2] = (boxes[:,2]*img_h/new_img_h - boxes[:,0]*img_h/new_img_h).astype(np.float32)
    new_boxes[:,3] = (boxes[:,3]*img_w/new_img_w - boxes[:,1]*img_w/new_img_w).astype(np.float32)
    new_boxes[:,0] = (boxes[:,0]*img_h/new_img_h).astype(np.float32)
    new_boxes[:,1] = (boxes[:,1]*img_w/new_img_w).astype(np.float32)
    return new_boxes

def _collectData(image_id, classes, boxes, probs, img_h, img_w, new_img_h, new_img_w):
    instance_num = probs.shape[0]
    boxes = _convertBoxes(boxes, img_h, img_w, new_img_h, new_img_w)

    image_ids = [image_id] * instance_num
    real_category_id = _cat_id_to_real_id(classes).tolist()
    bbox = boxes.tolist()#change format
    score = probs.tolist()

    for instance_index in range(instance_num):
        instance = {}
        instance['image_id'] = int(image_ids[instance_index])
        instance['category_id'] = real_category_id[instance_index]
        instance['bbox'] = bbox[instance_index]
        instance['score'] = score[instance_index][classes[instance_index]]
        _writeJSON(instance)



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
# import libs.boxes.cython_bbox as cython_bbox

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
#     tf.summary.scalar('learning_rate', lr)

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
            print (' failed to restore in %s %s' % (FLAGS.train_dir, checkpoint_path))
            raise

def evaluate(ap_threshold, gt_boxes, gt_masks, boxes, classes, probs, masks):
    num_instances = gt_boxes.shape[0]
    num_prediction = boxes.shape[0]
    recall = []
    precision = []
    if num_instances is not 0 and num_prediction is not 0:
        m = np.array(masks)
        m = np.transpose(m,(0,3,1,2))

        overlaps = cython_bbox.bbox_overlaps(
        np.ascontiguousarray(boxes[:, 0:4], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, 0:4], dtype=np.float))
    

        overlaps_recall = np.max(overlaps, axis=0)
        overlaps_precision = np.max(overlaps, axis=1)
        for i, threshold in enumerate(ap_threshold):
            recall.append(np.sum(overlaps_recall > threshold))
            precision.append(np.sum(overlaps_precision > threshold))
    else:
        for i, threshold in enumerate(ap_threshold):
            recall.append(0)
            precision.append(0)
    return np.array(recall), np.array(precision), num_instances, num_prediction
    
                
def test():
    """The main function that runs training"""

    ## data
    image, ih, iw, new_ih, new_iw, gt_boxes, gt_masks, num_instances, img_id = \
        datasets.get_dataset(FLAGS.dataset_name, 
                             FLAGS.dataset_split_name, 
                             FLAGS.dataset_dir, 
                             FLAGS.im_batch,
                             is_training=False)

    im_shape = tf.shape(image)
    image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], 3))

    ## network
    logits, end_points, pyramid_map = network.get_network(FLAGS.network, image,
            weight_decay=FLAGS.weight_decay, is_training=True)
    outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map,
            num_classes=81,
            base_anchors=9,
            is_training=False,
            gt_boxes=None, gt_masks=None, loss_weights=[0.0, 0.0, 0.0, 0.0, 0.0])

    input_image = end_points['input']

    testing_mask_rois = outputs['mask_ordered_rois']
    testing_mask_final_mask = outputs['mask_final_mask']
    testing_mask_final_clses = outputs['mask_final_clses']
    testing_mask_final_scores = outputs['mask_final_scores']

    ## solvers
    global_step = slim.create_global_step()

    cropped_rois = tf.get_collection('__CROPPED__')[0]
    transposed = tf.get_collection('__TRANSPOSED__')[0]
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
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
        threads.extend(qr.create_threads(sess, coord=coord, start=True))

    tf.train.start_queue_runners(sess=sess, coord=coord)

    # for step in range(FLAGS.max_iters):
    for step in range(10000):#range(40503):
        
        start_time = time.time()

        img_id_str, img_h, img_w, new_img_h, new_img_w, \
        gt_boxesnp, gt_masksnp,\
        input_imagenp,\
        testing_mask_roisnp, testing_mask_final_masknp, testing_mask_final_clsesnp, testing_mask_final_scoresnp = \
                     sess.run([img_id] + [ih] + [iw] + [new_ih] + [new_iw] +\
                              [gt_boxes] + [gt_masks] +\
                              [input_image] + \
                              [testing_mask_rois] + [testing_mask_final_mask] + [testing_mask_final_clses] + [testing_mask_final_scores])

        duration_time = time.time() - start_time
        if step % 1 == 0: 
            print ( """iter %d: image-id:%07d, time:%.3f(sec), """
                    """instances: %d, """
                    
                   % (step, img_id_str, duration_time, 
                      gt_boxesnp.shape[0]))

        if step % 1 == 0: 
            draw_bbox(step, 
                      np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
                      name='test_est', 
                      bbox=testing_mask_roisnp, 
                      label=testing_mask_final_clsesnp, 
                      prob=testing_mask_final_scoresnp,
                      mask=testing_mask_final_masknp,
                      vis_th=0.2)

            draw_bbox(step, 
                      np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
                      name='test_gt', 
                      bbox=gt_boxesnp[:,0:4], 
                      label=gt_boxesnp[:,4].astype(np.int32), 
                      prob=np.ones((gt_boxesnp.shape[0],81), dtype=np.float32),)


            print ("predict")
            # LOG (cat_id_to_cls_name(np.unique(np.argmax(np.array(training_rcnn_clsesnp),axis=1))))
            print (cat_id_to_cls_name(testing_mask_final_clsesnp))
            print (np.max(np.array(testing_mask_final_scoresnp),axis=1))

        _collectData(img_id_str, testing_mask_final_clsesnp, testing_mask_roisnp, testing_mask_final_scoresnp, img_h, img_w, new_img_h, new_img_w)

    # ## solvers
    # global_step = slim.create_global_step()

    # cropped_rois = tf.get_collection('__CROPPED__')[0]
    # transposed = tf.get_collection('__TRANSPOSED__')[0]
    
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # init_op = tf.group(
    #         tf.global_variables_initializer(),
    #         tf.local_variables_initializer()
    #         )
    # sess.run(init_op)

    # summary_op = tf.summary.merge_all()
    # logdir = os.path.join(FLAGS.train_dir, strftime('%Y%m%d%H%M%S', gmtime()))
    # if not os.path.exists(logdir):
    #     os.makedirs(logdir)
    # summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

    # ## restore
    # restore(sess)

    # ## main loop
    # coord = tf.train.Coordinator()
    # threads = []
    # # print (tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
    # for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
    #     threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
    #                                      start=True))

    # tf.train.start_queue_runners(sess=sess, coord=coord)

    # ap_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # total_recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # total_precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # total_instance = 0
    # total_prediction = 0


    # # for step in range(FLAGS.max_iters):
    # for step in range(2500):
        
    #     start_time = time.time()

    #     img_id_str, \
    #     gt_boxesnp, gt_masksnp,\
    #     input_imagenp, tmp_0np, tmp_1np, tmp_2np, tmp_3np, tmp_4np, tmp_5np, \
    #     testing_mask_roisnp, testing_mask_final_masknp, testing_mask_final_clsesnp, testing_mask_final_scoresnp = \
    #                  sess.run([img_id] + \
    #                           [gt_boxes] + [gt_masks] +\
    #                           [input_image] + [tmp_0] + [tmp_1] + [tmp_2] + [tmp_3] + [tmp_4] + [tmp_5] + \
    #                           [testing_mask_rois] + [testing_mask_final_mask] + [testing_mask_final_clses] + [testing_mask_final_scores])

    #     duration_time = time.time() - start_time
    #     if step % 1 == 0: 
    #         print ( """iter %d: image-id:%07d, time:%.3f(sec), """
    #                 """instances: %d, """
                    
    #                % (step, img_id_str, duration_time, 
    #                   gt_boxesnp.shape[0]))

    #     if step % 1 == 0: 
    #         draw_bbox(step, 
    #                   np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
    #                   name='test_est', 
    #                   bbox=testing_mask_roisnp, 
    #                   label=testing_mask_final_clsesnp, 
    #                   prob=testing_mask_final_scoresnp,
    #                   mask=testing_mask_final_masknp,)

    #     if step % 1 == 0:   
    #         draw_bbox(step, 
    #                   np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
    #                   name='test_gt', 
    #                   bbox=gt_boxesnp[:,0:4], 
    #                   label=gt_boxesnp[:,4].astype(np.int32), 
    #                   prob=np.ones((gt_boxesnp.shape[0],81), dtype=np.float32),)

    #     recall, precision, num_instances, num_prediction = evaluate(ap_threshold, gt_boxesnp, gt_masksnp, testing_mask_roisnp, testing_mask_final_clsesnp, testing_mask_final_scoresnp, testing_mask_final_masknp)
    #     total_recall += recall
    #     total_precision += precision
    #     total_instance += num_instances
    #     total_prediction += num_prediction

    #     print("recall       = {}".format([x / float(total_instance) for x in total_recall]))
    #     print("precision    = {}".format([x / float(total_prediction) for x in total_precision]))
    # print("recall       = {}".format(total_recall / float(total_instance)))
    # print("precision    = {}".format(total_precision / float(total_prediction)))



if __name__ == '__main__':
    test()
