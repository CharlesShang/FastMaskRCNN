# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import libs.configs.config_v1 as cfg
import libs.nms.gpu_nms as gpu_nms
import libs.nms.cpu_nms as cpu_nms

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    return gpu_nms.gpu_nms(dets, thresh, device_id=0)

def nms_wrapper(scores, boxes, threshold = 0.7, class_sets = None):
    """
    post-process the results of im_detect
    :param boxes: N * (K * 4) numpy
    :param scores: N * K numpy
    :param class_sets: e.g. CLASSES = ('__background__','person','bike','motorbike','car','bus')
    :return: a list of K-1 dicts, no background, each is {'class': classname, 'dets': None | [[x1,y1,x2,y2,score],...]}
    """
    num_class = scores.shape[1] if class_sets is None else len(class_sets)
    assert num_class * 4 == boxes.shape[1],\
        'Detection scores and boxes dont match %d vs %d' % (num_class, boxes.shape[1])
    class_sets = ['class_' + str(i) for i in range(0, num_class)] if class_sets is None else class_sets

    res = []
    for ind, cls in enumerate(class_sets[1:]):
        ind += 1 # skip background
        cls_boxes =  boxes[:, 4*ind : 4*(ind+1)]
        cls_scores = scores[:, ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, thresh=0.3)
        dets = dets[keep, :]
        dets = dets[np.where(dets[:, 4] > threshold)]
        r = {}
        if dets.shape[0] > 0:
            r['class'], r['dets'] = cls, dets
        else:
            r['class'], r['dets'] = cls, None
        res.append(r)
    return res

if __name__=='__main__':
  
  score = np.random.rand(10, 21)
  boxes = np.random.randint(0, 100, (10, 21, 2))
  s = np.random.randint(0, 100, (10, 21, 2))
  s = boxes + s
  boxes = np.concatenate((boxes, s), axis=2)
  boxes = np.reshape(boxes, [boxes.shape[0], -1])
  # score = np.reshape(score, [score.shape[0], -1])
  res = nms_wrapper(score, boxes)
  print (res)