# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import zipfile
import time
import numpy as np
import tensorflow as tf
from six.moves import urllib
from PIL import Image
from matplotlib import pyplot as plt

from libs.datasets.pycocotools.coco import COCO
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
from libs.logs.log import LOG

# The URL where the coco data can be downloaded.

_TRAIN_DATA_URL="https://msvocds.blob.core.windows.net/coco2014/train2014.zip"
_VAL_DATA_URL="https://msvocds.blob.core.windows.net/coco2014/val2014.zip"
_INS_LABEL_URL="https://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip"
_KPT_LABEL_URL="https://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip"
_CPT_LABEL_URL="https://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip"
_DATA_URLS=[
  _TRAIN_DATA_URL, _VAL_DATA_URL,
  _INS_LABEL_URL, _KPT_LABEL_URL, _CPT_LABEL_URL,
]

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('vis',  False,
                          'Show some visual masks')

def download_and_uncompress_zip(zip_url, dataset_dir):
  """Downloads the `zip_url` and uncompresses it locally.
     From: https://github.com/tensorflow/models/blob/master/slim/datasets/dataset_utils.py

  Args:
    zip_url: The URL of a zip file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = zip_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()

  if tf.gfile.Exists(filepath):
    print('Zip file already exist. Skip download..', filepath)
  else:
    filepath, _ = urllib.request.urlretrieve(zip_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  with zipfile.ZipFile(filepath) as f:
    print('Extracting ', filepath)
    f.extractall(dataset_dir)
    print('Successfully extracted')

def _real_id_to_cat_id(catId):
  """Note coco has 80 classes, but the catId ranges from 1 to 90!"""
  real_id_to_cat_id = \
    {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17,
     17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28, 27: 31, 28: 32, 29: 33, 30: 34,
     31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46, 42: 47, 43: 48, 44: 49,
     45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55, 51: 56, 52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62, 58: 63,
     59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75, 67: 76, 68: 77, 69: 78, 70: 79, 71: 80, 72: 81,
     73: 82, 74: 84, 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90}
  return real_id_to_cat_id[catId]

def _cat_id_to_real_id(readId):
  """Note coco has 80 classes, but the catId ranges from 1 to 90!"""
  cat_id_to_real_id = \
    {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
     18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
     35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44,
     50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58,
     64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
     82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
  return cat_id_to_real_id[readId]
  

class ImageReader(object):
  def __init__(self):
    self._decode_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_data, channels=3)
    self._decode_png = tf.image.decode_png(self._decode_data)

  def read_jpeg_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape

  def read_png_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 1
    return image


def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'coco_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _get_image_filenames(image_dir):
  return sorted(os.listdir(image_dir))


def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _to_tfexample(image_data, image_format, label_data, label_format, height, width):
  """Encode only masks """
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_feature(image_data),
      'image/format': _bytes_feature(image_format),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'label/encoded': _bytes_feature(label_data),
      'label/format': _bytes_feature(label_format),
      'label/height': _int64_feature(height),
      'label/width': _int64_feature(width),
  }))

def _to_tfexample_coco(image_data, image_format, label_data, label_format,
                       height, width,
                       num_instances, gt_boxes, masks):
  
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_feature(image_data),
      'image/format': _bytes_feature(image_format),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
  
      'label/num_instances': _int64_feature(num_instances), # N
      'label/gt_boxes': _bytes_feature(gt_boxes), # of shape (N, 5), (x1, y1, x2, y2, classid)
      'label/gt_masks': _bytes_feature(masks),       # of shape (N, height, width)
    
      'label/encoded': _bytes_feature(label_data),  # deprecated, this is used for pixel-level segmentation
      'label/format': _bytes_feature(label_format),
  }))


def _get_coco_masks(coco, img_id, height, width, img_name):
  """ get the masks for all the instances
  Note: some images are not annotated
  Return:
    masks, mxhxw numpy array
    classes, mx1
    bboxes, mx4
  """
  annIds = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
  # assert  annIds is not None and annIds > 0, 'No annotaion for %s' % str(img_id)
  anns = coco.loadAnns(annIds)
  # assert len(anns) > 0, 'No annotaion for %s' % str(img_id)
  masks = []
  classes = []
  bboxes = []
  mask = np.zeros((height, width), dtype=np.float32)
  segmentations = []
  for ann in anns:
    m = coco.annToMask(ann) # zero one mask
    masks.append(m)
    cat_id = _cat_id_to_real_id(ann['category_id'])
    classes.append(cat_id)
    bboxes.append(ann['bbox'])
    m = m.astype(np.float32) * cat_id
    mask[m > 0] = m[m > 0]

  masks = np.asarray(masks)
  classes = np.asarray(classes)
  bboxes = np.asarray(bboxes)
  # to x1, y1, x2, y2
  if bboxes.shape[0] <= 0:
    bboxes = np.zeros([0, 4], dtype=np.float32)
    classes = np.zeros([0], dtype=np.float32)
    print ('None Annotations %s' % img_name)
    LOG('None Annotations %s' % img_name)
  bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
  bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
  gt_boxes = np.hstack((bboxes, classes[:, np.newaxis]))
  gt_boxes = gt_boxes.astype(np.float32)
  masks = masks.astype(np.int32)
  mask = mask.astype(np.uint8)
  assert masks.shape[0] == gt_boxes.shape[0], 'Shape Error'
  
  return gt_boxes, masks, mask
  


def _add_to_tfrecord(record_dir, image_dir, annotation_dir, split_name):
  """Loads image files and writes files to a TFRecord.
  Note: masks and bboxes will lose shape info after converting to string.
  """

  assert split_name in ['train2014', 'val2014']
  annFile = os.path.join(annotation_dir, 'instances_%s.json' % (split_name))
  
  coco = COCO(annFile)

  cats = coco.loadCats(coco.getCatIds())
  print ('%s has %d images' %(split_name, len(coco.imgs)))
  imgs = [(img_id, coco.imgs[img_id]) for img_id in coco.imgs]
  
  num_shards = 40 if split_name == 'train2014' else 20
  num_per_shard = int(math.ceil(len(imgs) / float(num_shards)))
  
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    image_reader = ImageReader()
    
    # encode mask to png_string
    mask_placeholder = tf.placeholder(dtype=tf.uint8)
    encoded_image = tf.image.encode_png(mask_placeholder)
    
    with tf.Session('') as sess:
      for shard_id in range(num_shards):
        record_filename = _get_dataset_filename(record_dir, split_name, shard_id, num_shards)
        options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
        with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id + 1) * num_per_shard, len(imgs))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d\n' % (
              i + 1, len(imgs), shard_id))
            sys.stdout.flush()
            
            # image id and path
            img_id = imgs[i][0]
            img_name = imgs[i][1]['file_name']
            img_name = os.path.join(image_dir, split_name, img_name)
            
            if FLAGS.vis:
              im = Image.open(img_name)
              im.save('img.png')
              plt.figure(0)
              plt.axis('off')
              plt.imshow(im)
              # plt.show()
              # plt.close()
            
            # jump over the damaged images
            if split_name == 'val2014' and str(img_id) == '320612':
              continue
            
            # process anns
            h, w = imgs[i][1]['height'], imgs[i][1]['width']
            gt_boxes, masks, mask = _get_coco_masks(coco, img_id, h, w, img_name)
            # this encode matrix to png format string buff
            label_data = sess.run(encoded_image,
                                  feed_dict={mask_placeholder: np.expand_dims(mask, axis=2)})
            
            # read image
            assert os.path.exists(img_name), '%s dont exists'% img_name
            image_data = tf.gfile.FastGFile(img_name, 'r').read()
            height, width, depth = image_reader.read_jpeg_dims(sess, image_data)
            
            # to tf-record
            example = _to_tfexample_coco(
              image_data, 'jpg',
              label_data, 'png',
              height, width, gt_boxes.shape[0],
              gt_boxes.tostring(), masks.tostring())
            tfrecord_writer.write(example.SerializeToString())
  sys.stdout.write('\n')
  sys.stdout.flush()

def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  # for url in _DATA_URLS:
  #   download_and_uncompress_zip(url, dataset_dir)

  record_dir     = os.path.join(dataset_dir, 'records')
  annotation_dir = os.path.join(dataset_dir, 'annotations')

  if not tf.gfile.Exists(record_dir):
    tf.gfile.MakeDirs(record_dir)

  # process the training, validation data:
  _add_to_tfrecord(record_dir,
                   dataset_dir,
                   annotation_dir,
                   'train2014')
  _add_to_tfrecord(record_dir,
                   dataset_dir,
                   annotation_dir,
                   'val2014')

  print('\nFinished converting the coco dataset!')
