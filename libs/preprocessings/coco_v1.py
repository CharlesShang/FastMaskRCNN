#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import libs.configs.config_v1 as cfg
from . import utils as preprocess_utils

FLAGS = tf.app.flags.FLAGS 

def preprocess_image(image, gt_boxes, gt_masks, is_training=False):
    """preprocess image for coco
    1. random flipping
    2. min size resizing
    3. zero mean 
    4. ... 
    """
    if is_training:
        return preprocess_for_training(image, gt_boxes, gt_masks)
    else:
        return preprocess_for_test(image, gt_boxes, gt_masks)


def preprocess_for_training(image, gt_boxes, gt_masks):
    
    ih, iw = tf.shape(image)[0], tf.shape(image)[1]
    ## random flipping
    coin = tf.to_float(tf.random_uniform([1]))[0]
    image, gt_boxes, gt_masks =\
            tf.cond(tf.greater_equal(coin, 0.5), 
                    lambda: (preprocess_utils.flip_image(image),
                            preprocess_utils.flip_gt_boxes(gt_boxes, ih, iw),
                            preprocess_utils.flip_gt_masks(gt_masks)),
                    lambda: (image, gt_boxes, gt_masks))

    ## min size resizing
    new_ih, new_iw = preprocess_utils._smallest_size_at_least(ih, iw, cfg.FLAGS.image_min_size)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [new_ih, new_iw], align_corners=False)
    image = tf.squeeze(image, axis=[0])

    gt_masks = tf.expand_dims(gt_masks, -1)
    gt_masks = tf.cast(gt_masks, tf.float32)
    gt_masks = tf.image.resize_nearest_neighbor(gt_masks, [new_ih, new_iw], align_corners=False)
    gt_masks = tf.cast(gt_masks, tf.int32)
    gt_masks = tf.squeeze(gt_masks, axis=[-1])

    scale_ratio = tf.to_float(new_ih) / tf.to_float(ih)
    gt_boxes = preprocess_utils.resize_gt_boxes(gt_boxes, scale_ratio)

    ## random flip image
    # val_lr = tf.to_float(tf.random_uniform([1]))[0]
    # image = tf.cond(val_lr > 0.5, lambda: preprocess_utils.flip_image(image), lambda: image)
    # gt_masks = tf.cond(val_lr > 0.5, lambda: preprocess_utils.flip_gt_masks(gt_masks), lambda: gt_masks)
    # gt_boxes = tf.cond(val_lr > 0.5, lambda: preprocess_utils.flip_gt_boxes(gt_boxes, new_ih, new_iw), lambda: gt_boxes)

    ## zero mean image
    image = tf.cast(image, tf.float32)
    image = image / 256.0
    image = (image - 0.5) * 2.0
    image = tf.expand_dims(image, axis=0)

    ## rgb to bgr
    image = tf.reverse(image, axis=[-1])

    return image, gt_boxes, gt_masks 

def preprocess_for_test(image, gt_boxes, gt_masks):


    ih, iw = tf.shape(image)[0], tf.shape(image)[1]

    ## min size resizing
    new_ih, new_iw = preprocess_utils._smallest_size_at_least(ih, iw, cfg.FLAGS.image_min_size)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [new_ih, new_iw], align_corners=False)
    image = tf.squeeze(image, axis=[0])

    gt_masks = tf.expand_dims(gt_masks, -1)
    gt_masks = tf.cast(gt_masks, tf.float32)
    gt_masks = tf.image.resize_nearest_neighbor(gt_masks, [new_ih, new_iw], align_corners=False)
    gt_masks = tf.cast(gt_masks, tf.int32)
    gt_masks = tf.squeeze(gt_masks, axis=[-1])

    scale_ratio = tf.to_float(new_ih) / tf.to_float(ih)
    gt_boxes = preprocess_utils.resize_gt_boxes(gt_boxes, scale_ratio)
    
    ## zero mean image
    image = tf.cast(image, tf.float32)
    image = image / 256.0
    image = (image - 0.5) * 2.0
    image = tf.expand_dims(image, axis=0)

    ## rgb to bgr
    image = tf.reverse(image, axis=[-1])

    return image, gt_boxes, gt_masks 
