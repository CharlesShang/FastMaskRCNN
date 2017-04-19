# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import libs.boxes.cython_bbox as cython_bbox
import libs.configs.config_v1 as cfg
from libs.logs.log import LOG
from libs.boxes.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes

_DEBUG = False 
def encode(gt_masks, gt_boxes, rois, num_classes, mask_height, mask_width):
  """Encode masks groundtruth into learnable targets
  Sample some exmaples
  
  Params
  ------
  gt_masks: image_height x image_width {0, 1} matrix, of shape (G, imh, imw)
  gt_boxes: ground-truth boxes of shape (G, 5), each raw is [x1, y1, x2, y2, class]
  rois:     the bounding boxes of shape (N, 4),
  ## scores:   scores of shape (N, 1)
  num_classes; K
  mask_height, mask_width: height and width of output masks
  
  Returns
  -------
  # rois: boxes sampled for cropping masks, of shape (M, 4)
  labels: class-ids of shape (M, 1)
  mask_targets: learning targets of shape (M, pooled_height, pooled_width, K) in {0, 1} values
  mask_inside_weights: of shape (M, pooled_height, pooled_width, K) in {0, 1}Í indicating which mask is sampled
  """
  total_masks = rois.shape[0]
  if gt_boxes.size > 0: 
      # B x G
      overlaps = cython_bbox.bbox_overlaps(
          np.ascontiguousarray(rois[:, 0:4], dtype=np.float),
          np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
      gt_assignment = overlaps.argmax(axis=1)  # shape is N
      max_overlaps = overlaps[np.arange(len(gt_assignment)), gt_assignment] # N
      # note: this will assign every rois with a positive label 
      # labels = gt_boxes[gt_assignment, 4] # N
      labels = np.zeros((total_masks, ), np.float32)
      labels[:] = -1

      # sample positive rois which intersection is more than 0.5
      keep_inds = np.where(max_overlaps >= cfg.FLAGS.mask_threshold)[0]
      num_masks = int(min(keep_inds.size, cfg.FLAGS.masks_per_image))
      if keep_inds.size > 0 and num_masks < keep_inds.size:
        keep_inds = np.random.choice(keep_inds, size=num_masks, replace=False)
        LOG('Masks: %d of %d rois are considered positive mask. Number of masks %d'\
                     %(num_masks, rois.shape[0], gt_masks.shape[0]))

      labels[keep_inds] = gt_boxes[gt_assignment[keep_inds], -1]
        
      # rois = rois[inds]
      # labels = labels[inds].astype(np.int32)
      # gt_assignment = gt_assignment[inds]

      # ignore rois with overlaps between fg_threshold and bg_threshold 
      # mask are only defined on positive rois
      ignore_inds = np.where((max_overlaps < cfg.FLAGS.fg_threshold))[0]
      labels[ignore_inds] = -1 

      mask_targets = np.zeros((total_masks, mask_height, mask_width, num_classes), dtype=np.int32)
      mask_inside_weights = np.zeros((total_masks, mask_height, mask_width, num_classes), dtype=np.float32)
      rois [rois < 0] = 0
      
      # TODO: speed bottleneck?
      for i in keep_inds:
        roi = rois[i, :4]
        cropped = gt_masks[gt_assignment[i], int(roi[1]):int(roi[3])+1, int(roi[0]):int(roi[2])+1]
        cropped = cv2.resize(cropped, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)
        
        mask_targets[i, :, :, int(labels[i])] = cropped
        mask_inside_weights[i, :, :, int(labels[i])] = 1
  else:
      # there is no gt
      labels = np.zeros((total_masks, ), np.float32)
      labels[:] = -1
      mask_targets = np.zeros((total_masks, mask_height, mask_width, num_classes), dtype=np.int32)
      mask_inside_weights = np.zeros((total_masks, mask_height, mask_height, num_classes), dtype=np.float32)
  return labels, mask_targets, mask_inside_weights

def decode(mask_targets, rois, classes, ih, iw):
  """Decode outputs into final masks
  Params
  ------
  mask_targets: of shape (N, h, w, K)
  rois: of shape (N, 4) [x1, y1, x2, y2]
  classes: of shape (N, 1) the class-id of each roi
  height: image height
  width:  image width
  
  Returns
  ------
  M: a painted image with all masks, of shape (height, width), in [0, K]
  """
  Mask = np.zeros((ih, iw), dtype=np.float32)
  assert rois.shape[0] == mask_targets.shape[0], \
    '%s rois vs %d masks' %(rois.shape[0], mask_targets.shape[0])
  num = rois.shape[0]
  rois = clip_boxes(rois, (ih, iw))
  for i in np.arange(num):
    k = classes[i]
    mask = mask_targets[i, :, :, k]
    h, w = rois[i, 3] - rois[i, 1] + 1, rois[i, 2] - rois[i, 0] + 1
    x, y = rois[i, 0], rois[i, 1]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask *= k
    
    # paint
    Mask[y:y+h, x:x+w] = mask
  
  return Mask



if __name__ == '__main__':
  
  import time
  import matplotlib.pyplot as plt
  
  t = time.time()
  
  for i in range(10):
    cfg.FLAGS.mask_threshold = 0.2
    N = 50
    W, H = 200, 200
    M = 50
    
    gt_masks = np.zeros((2, H, W), dtype=np.int32)
    gt_masks[0, 50:150, 50:150] = 1
    gt_masks[1, 100:150, 50:150] = 1
    gt_boxes = np.asarray(
      [
        [20, 20, 100, 100, 1],
        [100, 100, 180, 180, 2]
      ])
    rois = gt_boxes[:, :4]
    print (rois)
    rois, labels, mask_targets, mask_inside_weights = encode(gt_masks, gt_boxes, rois, 3, 7, 7)
    print (rois)
    Mask = decode(mask_targets, rois, labels, H, W)
    if True:
      plt.figure(1)
      plt.imshow(Mask)
      plt.show()
      time.sleep(2)
  print(labels)
  print('average time: %f' % ((time.time() - t) / 10.0))
  
