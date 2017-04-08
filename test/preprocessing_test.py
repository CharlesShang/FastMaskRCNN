#!/usr/bin/env python
# coding=utf-8

import numpy as np
import sys
import os
import tensorflow as tf 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import libs.preprocessings.coco_v1 as coco_preprocess
import  libs.configs.config_v1 as cfg

ih, iw, ic = 400,500, 3
N = 3
image = np.random.randint(0, 255, (ih, iw, ic)).astype(np.uint8)
gt_masks = np.zeros((N, ih, iw)).astype(np.int32)
xy = np.random.randint(0, min(iw, ih)-100, (N, 2)).astype(np.float32)
wh = np.random.randint(20, 40, (N, 2)).astype(np.float32)
cls = np.random.randint(1, 6, (N, 1)).astype(np.float32)
gt_boxes = np.hstack((xy, xy + wh, cls)).astype(np.float32)
gt_boxes_np = gt_boxes 
image_np = image 
gt_masks_np = gt_masks 

for i in range(N):
    box = gt_boxes[i, 0:4]
    gt_masks[i, int(box[1]):int(box[3]),
                int(box[0]):int(box[2])] = 1
image = tf.constant(image)
gt_boxes = tf.constant(gt_boxes)
gt_masks = tf.constant(gt_masks)

image, gt_boxes, gt_masks = \
        coco_preprocess.preprocess_image(image, gt_boxes, gt_masks, is_training=True)

with tf.Session() as sess:
    # print(image.eval())
    image_tf, gt_boxes_tf, gt_masks_tf = \
            sess.run([image, gt_boxes, gt_masks])
    print ('#######################')
    print ('DATA PREPROCESSING TEST')
    print ('#######################')
    print ('gt_boxes shape:', gt_boxes_tf.shape)
    print('mask shape:', gt_masks_tf.shape)
    print(gt_boxes_tf)
    for i in range(N):
        box = np.round(gt_boxes_tf[i, 0:4])
        box = box.astype(np.int32)
        m = gt_masks_tf[i, box[1]:box[3], box[0]:box[2]]
        print ('after:', box)
        print (np.sum(m)/ (0.0 + m.size))
        print (m)
        box = np.round(gt_boxes_np[i, 0:4])
        box = box.astype(np.int32)
        m = gt_masks_np[i, box[1]:box[3], box[0]:box[2]]
        print ('ori box:', box)
        print (np.sum(m)/ (0.0 + m.size))
