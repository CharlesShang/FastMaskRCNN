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
import cv2
from time import gmtime, strftime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libs.configs.config_v1 as cfg
import libs.datasets.dataset_factory as datasets
import libs.nets.nets_factory as network 
import libs.datasets.pycocotools.mask as pycoco_mask

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

def _convertBoxes(image_id, boxes, original_image_height, original_image_width, image_height, image_width):
    original_image_boxes = boxes
    height_ratio = original_image_height / image_height
    width_ratio = original_image_width / image_width
    original_image_boxes[:,2] = (boxes[:,2] * width_ratio  - boxes[:,0] * width_ratio ).astype(np.float32)
    original_image_boxes[:,3] = (boxes[:,3] * height_ratio - boxes[:,1] * height_ratio).astype(np.float32)
    original_image_boxes[:,0] = (boxes[:,0] * width_ratio ).astype(np.float32)
    original_image_boxes[:,1] = (boxes[:,1] * height_ratio).astype(np.float32)
    return original_image_boxes

def _convertMasks(image_id, masks, classes, boxes, image_height, image_width):
    assert masks.shape[0] == classes.shape[0] == boxes.shape[0], \
      'convertMasks error %d vs %d ' % (masks.shape[0], classes.shape[0], boxes.shape[0])
    original_image_masks = []
    for instance_index, (mask, cls, box) in enumerate(zip(masks, classes, boxes)):
        mask = np.transpose(mask, [2, 0, 1])
        box = np.round(box)
        box_offset_x = box[0]
        box_offset_y = box[1]
        box_width    = box[2]
        box_height   = box[3]
        #create blank image
        size = (image_height, image_width)
        original_image_mask = np.zeros(size, np.uint8)
        #fit mask to box
        mask = cv2.resize(mask[cls], (box_width, box_height))
        #place box on blank image
        y1 = int(box_offset_y)
        y2 = int(box_offset_y + mask.shape[0])
        x1 = int(box_offset_x) 
        x2 = int(box_offset_x + mask.shape[1])
        original_image_mask[y1:y2, x1:x2] = mask*255
        #threshold by 0.5
        original_image_mask = (original_image_mask >= 127) * 255
        original_image_masks.append(original_image_mask)

    return original_image_masks

def _collectData(image_id, classes, boxes, probs, original_image_height, original_image_width, image_height, image_width, masks=None):
    instance_num = probs.shape[0]
    original_image_boxes = _convertBoxes(image_id, boxes, original_image_height, original_image_width, image_height, image_width)
    if masks is not None:
        original_image_masks = _convertMasks(image_id, masks, classes, original_image_boxes, original_image_height, original_image_width)

    image_ids = [image_id] * instance_num
    real_category_id = _cat_id_to_real_id(classes).tolist()
    original_image_boxes = original_image_boxes.tolist()#change format
    score = probs.tolist()

    for instance_index in range(instance_num):
        instance = {}
        instance['image_id'] = int(image_ids[instance_index])
        instance['category_id'] = real_category_id[instance_index]
        instance['bbox'] = original_image_boxes[instance_index]
        if masks is not None:
            RLE = np.array(original_image_masks[instance_index], order='F', dtype= np.uint8)
            RLE = pycoco_mask.encode(RLE)
            instance['segmentation'] = RLE
        instance['score'] = score[instance_index][classes[instance_index]]
        _writeJSON(instance)

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
                
def test():
    """The main function that runs training"""

    ## data
    image, original_image_height, original_image_width, image_height, image_width, gt_boxes, gt_masks, num_instances, image_id = \
        datasets.get_dataset(FLAGS.dataset_name, 
                             FLAGS.dataset_split_name_test, 
                             FLAGS.dataset_dir, 
                             FLAGS.im_batch,
                             is_training=False)

    im_shape = tf.shape(image)
    image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], 3))

    ## network
    logits, end_points, pyramid_map = network.get_network(FLAGS.network, image,
            weight_decay=0.0, batch_norm_decay=0.0, is_training=True)
    outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map,
            num_classes=81,
            base_anchors=3,
            is_training=False,
            gt_boxes=None, gt_masks=None, loss_weights=[0.0, 0.0, 0.0, 0.0, 0.0])

    input_image = end_points['input']

    testing_mask_rois = outputs['mask_ordered_rois']
    testing_mask_final_mask = outputs['mask_final_mask']
    testing_mask_final_clses = outputs['mask_final_clses']
    testing_mask_final_scores = outputs['mask_final_scores']

    ## solvers
    global_step = slim.create_global_step()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # init_op = tf.group(
    #         tf.global_variables_initializer(),
    #         tf.local_variables_initializer()
    #         )
    # sess.run(init_op)

    # summary_op = tf.summary.merge_all()
    logdir = os.path.join(FLAGS.train_dir, strftime('%Y%m%d%H%M%S', gmtime()))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

    ## restore
    restore(sess)
    tf.train.start_queue_runners(sess=sess)

    ## main loop
    # for step in range(FLAGS.max_iters):
    for step in range(82783):#range(40503):
        
        start_time = time.time()

        image_id_str, original_image_heightnp, original_image_widthnp, image_heightnp, image_widthnp, \
        gt_boxesnp, gt_masksnp,\
        input_imagenp,\
        testing_mask_roisnp, testing_mask_final_masknp, testing_mask_final_clsesnp, testing_mask_final_scoresnp = \
                     sess.run([image_id] + [original_image_height] + [original_image_width] + [image_height] + [image_width] +\
                              [gt_boxes] + [gt_masks] +\
                              [input_image] + \
                              [testing_mask_rois] + [testing_mask_final_mask] + [testing_mask_final_clses] + [testing_mask_final_scores])

        duration_time = time.time() - start_time
        if step % 1 == 0: 
            print ( """iter %d: image-id:%07d, time:%.3f(sec), """
                    """instances: %d, """
                    
                   % (step, image_id_str, duration_time, 
                      gt_boxesnp.shape[0]))

        if step % 1 == 0: 
            draw_bbox(step, 
                      np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0), 
                      name='test_est', 
                      bbox=testing_mask_roisnp, 
                      label=testing_mask_final_clsesnp, 
                      prob=testing_mask_final_scoresnp,
                      mask=testing_mask_final_masknp,
                      vis_th=0.5)

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

        _collectData(image_id_str, testing_mask_final_clsesnp, testing_mask_roisnp, testing_mask_final_scoresnp, original_image_heightnp, original_image_widthnp, image_heightnp, image_widthnp, testing_mask_final_masknp)

if __name__ == '__main__':
    test()
