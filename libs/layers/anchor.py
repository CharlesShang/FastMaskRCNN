from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import libs.boxes.cython_bbox as cython_bbox
import libs.configs.config_v1 as cfg
from libs.boxes.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes
from libs.boxes.anchor import anchors_plane, jitter_gt_boxes
from libs.logs.log import LOG
# FLAGS = tf.app.flags.FLAGS

_DEBUG = False

def encode(gt_boxes, all_anchors, feature_height, feature_width, stride, image_height, image_width, ignore_cross_boundary=True):
    """Matching and Encoding groundtruth into learning targets
    Sampling
    
    Parameters
    ---------
    gt_boxes: an array of shape (G x 5), [x1, y1, x2, y2, class]
    all_anchors: an array of shape (h, w, A, 4),
    feature_height: height of feature
    feature_width: width of feature
    image_height: height of image
    image_width: width of image
    stride: downscale factor w.r.t the input size, e.g., [4, 8, 16, 32]
    Returns
    --------
    labels:   Nx1 array in [0, num_classes]
    bbox_targets: N x (4) regression targets
    bbox_inside_weights: N x (4), in {0, 1} indicating to which class is assigned.
    """
    # TODO: speedup this module
    allow_border = cfg.FLAGS.allow_border
    all_anchors = all_anchors.reshape([-1, 4])
    total_anchors = all_anchors.shape[0]

    labels = np.empty((total_anchors, ), dtype=np.int32)
    labels.fill(-1)

    jittered_gt_boxes = jitter_gt_boxes(gt_boxes[:, :4])
    clipped_gt_boxes = clip_boxes(jittered_gt_boxes, (image_height, image_width))

    if gt_boxes.size > 0:
        overlaps = cython_bbox.bbox_overlaps(
                   np.ascontiguousarray(all_anchors, dtype=np.float),
                   np.ascontiguousarray(clipped_gt_boxes, dtype=np.float))

        gt_assignment = overlaps.argmax(axis=1)  # (A)
        max_overlaps = overlaps[np.arange(total_anchors), gt_assignment]
        gt_argmax_overlaps = overlaps.argmax(axis=0)  # G
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]

        # bg label: less than threshold IOU
        labels[max_overlaps < cfg.FLAGS.rpn_bg_threshold] = 0
        # fg label: above threshold IOU 
        labels[max_overlaps >= cfg.FLAGS.rpn_fg_threshold] = 1

        # ignore cross-boundary anchors
        if ignore_cross_boundary is True:
            cb_inds = _get_cross_boundary(all_anchors, image_height, image_width, allow_border)
            labels[cb_inds] = -1

        # this is sentive to boxes of little overlaps, use with caution!
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        # fg label: for each gt, hard-assign anchor with highest overlap despite its overlaps
        labels[gt_argmax_overlaps] = 1

        # subsample positive labels if there are too many
        num_fg = int(cfg.FLAGS.fg_rpn_fraction * cfg.FLAGS.rpn_batch_size)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1
    else:
        # if there is no gt
        labels[:] = 0

    # TODO: mild hard negative mining
    # subsample negative labels if there are too many
    num_fg = np.sum(labels == 1)
    num_bg = max(min(cfg.FLAGS.rpn_batch_size - num_fg, num_fg * 3), 8)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    bbox_targets = np.zeros((total_anchors, 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets = _compute_targets(all_anchors, gt_boxes[gt_assignment, :])
    bbox_inside_weights = np.zeros((total_anchors, 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = 1.0#0.1

    labels = labels.reshape((1, feature_height, feature_width, -1))
    bbox_targets = bbox_targets.reshape((1, feature_height, feature_width, -1))
    bbox_inside_weights = bbox_inside_weights.reshape((1, feature_height, feature_width, -1))

    return labels, bbox_targets, bbox_inside_weights

def decode(boxes, scores, all_anchors, image_height, image_width):
    """Decode outputs into boxes
    Parameters
    ---------
    boxes: an array of shape (1, h, w, Ax4)
    scores: an array of shape (1, h, w, Ax2),
    all_anchors: an array of shape (1, h, w, Ax4), [x1, y1, x2, y2]
    
    Returns
    --------
    final_boxes: of shape (R x 4)
    classes: of shape (R) in {0,1,2,3... K-1}
    scores: of shape (R) in [0 ~ 1]
    """
    all_anchors = all_anchors.reshape((-1, 4))
    boxes = boxes.reshape((-1, 4))
    scores = scores.reshape((-1, 2))

    assert scores.shape[0] == boxes.shape[0] == all_anchors.shape[0], \
      'Anchor layer shape error %d vs %d vs %d' % (scores.shape[0], boxes.shape[0], all_anchors.reshape[0])

    boxes = bbox_transform_inv(all_anchors, boxes)
    boxes = clip_boxes(boxes, (image_height, image_width))
    classes = np.argmax(scores, axis=1).astype(np.int32)
    scores = scores[:, 1]
    
    return boxes, classes, scores

def sample(boxes, scores, ih, iw, is_training):
    """
    Sampling the anchor layer outputs for next stage, mask or roi prediction or roi
    
    Params
    ----------
    boxes:  of shape (? ,4)
    scores: foreground prob
    ih:     image height
    iw:     image width
    is_training:  'test' or 'train'
    
    Returns
    ----------
    rois: of shape (N, 4)
    scores: of shape (N, 1)
    batch_ids:
    """
    return


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
  
def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

def _get_cross_boundary(anchors, image_height, image_width, allow_border):

    cb_inds = np.where((anchors[:, 0] <= -(anchors[:, 2] - anchors[:, 0]) * allow_border) &
                       (anchors[:, 1] <= -(anchors[:, 3] - anchors[:, 1]) * allow_border) &
                       (anchors[:, 2] >= image_width + (anchors[:, 2] - anchors[:, 0]) * allow_border) &
                       (anchors[:, 3] >= image_height + (anchors[:, 3] - anchors[:, 1]) * allow_border))[0]

    return cb_inds

if __name__ == '__main__':
  
    import time
    t = time.time()
    
    for i in range(10):
        cfg.FLAGS.fg_threshold = 0.1
        classes = np.random.randint(0, 1, (50, 1))
        boxes = np.random.randint(10, 50, (50, 2))
        s = np.random.randint(20, 50, (50, 2))
        s = boxes + s
        boxes = np.concatenate((boxes, s), axis=1)
        gt_boxes = np.hstack((boxes, classes))
        # gt_boxes = boxes

        N = 100
        rois = np.random.randint(10, 50, (N, 2))
        s = np.random.randint(0, 20, (N, 2))
        s = rois + s
        rois = np.concatenate((rois, s), axis=1)
        indexs = np.arange(N)

        all_anchors = anchors_plane(200, 300, stride = 4, scales=[2, 4, 8, 16, 32], ratios=[0.5, 1, 2.0], base=16)
        labels, bbox_targets, bbox_inside_weights = encode(gt_boxes, all_anchors=all_anchors, height=200, width=300, stride=4, indexs=indexs)

        all_anchors = anchors_plane(100, 150, stride = 8, scales=[2, 4, 8, 16, 32], ratios=[0.5, 1, 2.0], base=16)
        labels, bbox_targets, bbox_inside_weights = encode(gt_boxes, all_anchors=all_anchors, height=100, width=150, stride=8, indexs=indexs)

        all_anchors = anchors_plane(50, 75, stride = 16, scales=[2, 4, 8, 16, 32], ratios=[0.5, 1, 2.0], base=16)
        labels, bbox_targets, bbox_inside_weights = encode(gt_boxes, all_anchors=all_anchors, height=50, width=75, stride=16, indexs=indexs)

        all_anchors = anchors_plane(25, 37, stride = 32, scales=[2, 4, 8, 16, 32], ratios=[0.5, 1, 2.0], base=16)
        labels, bbox_targets, bbox_inside_weights = encode(gt_boxes, all_anchors=all_anchors, height=25, width=37, stride=32, indexs=indexs)
        # anchors, _, _ = anchors_plane(200, 300, stride=4, boarder=0)
  
    print('average time: %f' % ((time.time() - t)/10.0))
