#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from libs.datasets import download_and_convert_coco
from libs.configs import config_v1

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string(
#     'dataset_name', 'coco',
#     'The name of the dataset to convert, one of "coco", "cifar10", "flowers", "mnist".')

# tf.app.flags.DEFINE_string(
#     'dataset_dir', 'data/coco',
#     'The directory where the output TFRecords and temporary files are saved.')


def main(_):
  if not os.path.isdir('./output/mask_rcnn'):
    os.makedirs('./output/mask_rcnn')
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  elif FLAGS.dataset_name == 'coco':
    download_and_convert_coco.run(FLAGS.dataset_dir, FLAGS.dataset_split_name)
  else:
    raise ValueError(
        'dataset_name [%s] was not recognized.' % FLAGS.dataset_dir)

if __name__ == '__main__':
  tf.app.run()

