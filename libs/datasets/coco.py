from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

_FILE_PATTERN = 'coco_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train2014': 82783, 'val2014': 40504}

_NUM_CLASSES = 81

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'An annotation image of varying size. (pixel-level masks)',
    'gt_masks': 'masks of instances in this image. (instance-level masks), of shape (N, image_height, image_width)',
    'gt_boxes': 'bounding boxes and classes of instances in this image, of shape (N, 5), each entry is (x1, y1, x2, y2)',
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
    
    'label/num_instances': tf.FixedLenFeature((), tf.int64),
    'label/gt_boxes': tf.FixedLenFeature((), tf.string),
    'label/gt_masks': tf.FixedLenFeature((), tf.string),
  }
  
  def _masks_decoder(keys_to_tensors):
    masks = tf.decode_raw(keys_to_tensors['label/gt_masks'], tf.uint8)
    width = tf.cast(keys_to_tensors['image/width'], tf.int32)
    height = tf.cast(keys_to_tensors['image/height'], tf.int32)
    instances = tf.cast(keys_to_tensors['label/num_instances'], tf.int32)
    mask_shape = tf.stack([instances, height, width])
    return tf.reshape(masks, mask_shape)
  
  def _gt_boxes_decoder(keys_to_tensors):
    bboxes = tf.decode_raw(keys_to_tensors['label/gt_boxes'], tf.float32)
    instances = tf.cast(keys_to_tensors['label/num_instances'], tf.int32)
    bboxes_shape = tf.stack([instances, 5])
    return tf.reshape(bboxes, bboxes_shape)
  
  def _width_decoder(keys_to_tensors):
    width = keys_to_tensors['image/width']
    return tf.cast(width, tf.int32)
  
  def _height_decoder(keys_to_tensors):
    height = keys_to_tensors['image/height']
    return tf.cast(height, tf.int32)
  
  items_to_handlers = {
    'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'label': slim.tfexample_decoder.Image('label/encoded', 'label/format', channels=1),
    'gt_masks': slim.tfexample_decoder.ItemHandlerCallback(
                ['label/gt_masks', 'label/num_instances', 'image/width', 'image/height'], _masks_decoder),
    'gt_boxes': slim.tfexample_decoder.ItemHandlerCallback(['label/gt_boxes', 'label/num_instances'], _gt_boxes_decoder),
    'width': slim.tfexample_decoder.ItemHandlerCallback(['image/width'], _width_decoder),
    'height': slim.tfexample_decoder.ItemHandlerCallback(['image/height'], _height_decoder),
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

def read(tfrecords_filename):

  if not isinstance(tfrecords_filename, list):
    tfrecords_filename = [tfrecords_filename]
  filename_queue = tf.train.string_input_producer(
    tfrecords_filename, num_epochs=100)

  options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
  reader = tf.TFRecordReader(options=options)
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image/img_id': tf.FixedLenFeature([], tf.int64),
      'image/encoded': tf.FixedLenFeature([], tf.string),
      'image/height': tf.FixedLenFeature([], tf.int64),
      'image/width': tf.FixedLenFeature([], tf.int64),
      'label/num_instances': tf.FixedLenFeature([], tf.int64),
      'label/gt_masks': tf.FixedLenFeature([], tf.string),
      'label/gt_boxes': tf.FixedLenFeature([], tf.string),
      'label/encoded': tf.FixedLenFeature([], tf.string),
      })
  # image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
  img_id = tf.cast(features['image/img_id'], tf.int32)
  ih = tf.cast(features['image/height'], tf.int32)
  iw = tf.cast(features['image/width'], tf.int32)
  num_instances = tf.cast(features['label/num_instances'], tf.int32)
  image = tf.decode_raw(features['image/encoded'], tf.uint8)
  imsize = tf.size(image)
  image = tf.cond(tf.equal(imsize, ih * iw), \
          lambda: tf.image.grayscale_to_rgb(tf.reshape(image, (ih, iw, 1))), \
          lambda: tf.reshape(image, (ih, iw, 3)))

  gt_boxes = tf.decode_raw(features['label/gt_boxes'], tf.float32)
  gt_boxes = tf.reshape(gt_boxes, [num_instances, 5])
  gt_masks = tf.decode_raw(features['label/gt_masks'], tf.uint8)
  gt_masks = tf.cast(gt_masks, tf.int32)
  gt_masks = tf.reshape(gt_masks, [num_instances, ih, iw])
  
  return image, ih, iw, gt_boxes, gt_masks, num_instances, img_id

