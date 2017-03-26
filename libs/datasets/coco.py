from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import tensorflow.contrib.slim as slim

_FILE_PATTERN = 'coco_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train2014': 82783, 'val2014': 40504}

_NUM_CLASSES = 80

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'An annotation image of varying size. (pixel-level masks)',
    'masks': 'masks of instances in this image. (instance-level masks)',
    'classes': 'classes of instances in this image.',
    'bboxes': 'bounding boxes of instances in this image.',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)
  
  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, 'records', file_pattern % split_name)
  
  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader
  
  keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
    'label/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'label/format': tf.FixedLenFeature((), tf.string, default_value='png'),
    'image/height': tf.FixedLenFeature((), tf.int64),
    'image/width': tf.FixedLenFeature((), tf.int64),
    
    'label/instances': tf.FixedLenFeature((), tf.int64),
    'label/classes': tf.FixedLenFeature((), tf.int64),
    'label/bboxes': tf.FixedLenFeature((), tf.string),
    'label/masks': tf.FixedLenFeature((), tf.string),
  }
  
  def masks_decoder(keys_to_tensors):
    masks = tf.decode_raw(keys_to_tensors['label/masks'], tf.uint8)
    width = tf.cast(keys_to_tensors['image/width'], tf.int32)
    height = tf.cast(keys_to_tensors['image/height'], tf.int32)
    instances = tf.cast(keys_to_tensors['label/instances'], tf.int32)
    mask_shape = tf.stack([instances, height, width])
    return tf.reshape(masks, mask_shape)
  
  def bboxes_decoder(keys_to_tensors):
    bboxes = tf.decode_raw(keys_to_tensors['label/bboxes'], tf.float32)
    instances = tf.cast(keys_to_tensors['label/instances'], tf.int32)
    bboxes_shape = tf.stack([instances, 4])
    return tf.reshape(bboxes, bboxes_shape)
  
  def classes_decoder(keys_to_tensors):
    classes = keys_to_tensors['label/classes']
    instances = tf.cast(keys_to_tensors['label/instances'], tf.int32)
    class_shape = tf.stack([instances, 1])
    return tf.reshape(classes, class_shape)
  
  def width_decoder(keys_to_tensors):
    width = keys_to_tensors['image/width']
    return tf.cast(width, tf.int32)
  
  def height_decoder(keys_to_tensors):
    height = keys_to_tensors['image/height']
    return tf.cast(height, tf.int32)
  
  items_to_handlers = {
    'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'label': slim.tfexample_decoder.Image('label/encoded', 'label/format', channels=1),
    'masks': slim.tfexample_decoder.ItemHandlerCallback(
      ['label/masks', 'label/instances', 'image/width', 'image/height'], masks_decoder),
    'bboxes': slim.tfexample_decoder.ItemHandlerCallback(['label/bboxes', 'label/instances'], bboxes_decoder),
    'classes': slim.tfexample_decoder.ItemHandlerCallback(['label/classes', 'label/instances'], classes_decoder),
    'width': slim.tfexample_decoder.ItemHandlerCallback(['image/width'], width_decoder),
    'height': slim.tfexample_decoder.ItemHandlerCallback(['image/height'], height_decoder),
  }
  
  decoder = slim.tfexample_decoder.TFExampleDecoder(
    keys_to_features, items_to_handlers)
  
  return slim.dataset.Dataset(
    data_sources=file_pattern,
    reader=reader,
    decoder=decoder,
    num_samples=SPLITS_TO_SIZES[split_name],
    items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
    num_classes=_NUM_CLASSES)

