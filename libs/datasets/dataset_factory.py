from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
from libs.datasets import coco
import libs.preprocessings.coco_v1 as coco_preprocess

def get_dataset(dataset_name, split_name, dataset_dir, 
        im_batch=1, is_training=False, file_pattern=None, reader=None):
    """"""
    if file_pattern is None:
        file_pattern = dataset_name + '_' + split_name + '*.tfrecord' 

    tfrecords = glob.glob(dataset_dir + '/records/' + file_pattern)
    image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = coco.read(tfrecords)

    image, gt_boxes, gt_masks = coco_preprocess.preprocess_image(image, gt_boxes, gt_masks, is_training)

    return image, ih, iw, gt_boxes, gt_masks, num_instances, img_id

