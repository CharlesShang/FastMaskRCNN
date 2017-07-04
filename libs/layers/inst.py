from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import libs.configs.config_v1 as cfg
import libs.boxes.nms_wrapper as nms_wrapper
import libs.boxes.cython_bbox as cython_bbox
from libs.boxes.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes
from libs.logs.log import LOG

_DEBUG=False

def inference(boxes, classes, prob, class_agnostic=True):

    min_size = cfg.FLAGS.min_size
    inst_nms_threshold = cfg.FLAGS.inst_nms_threshold
    post_nms_inst_n = cfg.FLAGS.post_nms_inst_n
    if class_agnostic is True:
        scores = prob[range(prob.shape[0]),classes]

        boxes = boxes.reshape((-1, 4))
        scores = scores.reshape((-1, 1))
        assert scores.shape[0] == boxes.shape[0], 'scores and boxes dont match'

        # filter background
        keeps = np.where(classes != 0)[0]
        scores = scores[keeps]
        boxes = boxes[keeps, :]
        classes = classes[keeps]
        prob = prob[keeps, :]
        print("after filter bg:", len(classes))

        # filter minimum size
        keeps = _filter_boxes(boxes, min_size=min_size)
        scores = scores[keeps]
        boxes = boxes[keeps, :]
        classes = classes[keeps]
        prob = prob[keeps, :]
        

        #filter with scores
        keeps = np.where(scores > 0.5)[0]
        scores = scores[keeps]
        boxes = boxes[keeps, :]
        classes = classes[keeps]
        prob = prob[keeps, :]

        # filter with nms
        det = np.hstack((boxes, scores)).astype(np.float32)
        keeps = nms_wrapper.nms(det, inst_nms_threshold)
        

        # filter low score
        if post_nms_inst_n > 0:
            keeps = keeps[:post_nms_inst_n]
        scores = scores[keeps]
        boxes = boxes[keeps, :]
        classes = classes[keeps]
        prob = prob[keeps, :]
        print("after nms:", len(classes))

        if len(classes) is 0:
            scores = np.zeros((1,81))
            boxes = np.array([[0.0,0.0,2.0,2.0]])
            classes = np.array([[0]])

    else:
        raise "inference nms type error"
    
    batch_inds = np.zeros([boxes.shape[0]], dtype=np.int32)

    return boxes.astype(np.float32), classes.astype(np.int32), prob.astype(np.float32), batch_inds

def _jitter_boxes(boxes, jitter=0.1):
    """ jitter the boxes before appending them into rois
    """
    jittered_boxes = boxes.copy()
    ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
    hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
    width_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * ws
    height_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * hs
    jittered_boxes[:, 0] += width_offset
    jittered_boxes[:, 2] += width_offset
    jittered_boxes[:, 1] += height_offset
    jittered_boxes[:, 3] += height_offset

    return jittered_boxes

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def _apply_nms(boxes, scores, threshold = 0.5):
    """After this only positive boxes are left
    Applying this class-wise
    """
    num_class = scores.shape[1]
    assert boxes.shape[0] == scores.shape[0], \
        'Shape dismatch {} vs {}'.format(boxes.shape, scores.shape)
  
    final_boxes = []
    final_scores = []
    for cls in np.arange(1, num_class):
        cls_boxes = boxes[:, 4*cls: 4*cls+4]
        cls_scores = scores[:, cls]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))
        keep = nms_wrapper.nms(dets, thresh=0.3)
        dets = dets[keep, :]
        dets = dets[np.where(dets[:, 4] > threshold)]
        final_boxes.append(dets[:, :4])
        final_scores.append(dets[:, 4])
  
    final_boxes = np.vstack(final_boxes)
    final_scores = np.vstack(final_scores)
  
    return final_boxes, final_scores

if __name__ == '__main__':
    import time
    t = time.time()
  
    for i in range(10):
        N = 200000
        boxes = np.random.randint(0, 50, (N, 2))
        s = np.random.randint(10, 40, (N, 2))
        s = boxes + s
        boxes = np.hstack((boxes, s))
        
        scores = np.random.rand(N, 1)
        # scores_ = 1 - np.random.rand(N, 1)
        # scores = np.hstack((scores, scores_))
      
        boxes, scores = sample_rpn_outputs(boxes, scores, only_positive=False)
  
    print ('average time %f' % ((time.time() - t) / 10))
