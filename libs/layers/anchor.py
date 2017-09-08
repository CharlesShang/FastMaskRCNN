from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import libs.boxes.cython_bbox as cython_bbox
import libs.configs.config_v1 as cfg
from libs.boxes.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes
from libs.boxes.anchor import anchors_plane
from libs.logs.log import LOG
# FLAGS = tf.app.flags.FLAGS

_DEBUG = False

def encode(gt_boxes, all_anchors, height, width, stride, ih, iw, ignore_cross_boundary=True):
    """Matching and Encoding groundtruth into learning targets
    Sampling
    
    Parameters
    ---------
    gt_boxes: an array of shape (G x 5), [x1, y1, x2, y2, class]
    all_anchors: an array of shape (h, w, A, 4),
    width: width of feature
    height: height of feature
    stride: downscale factor w.r.t the input size, e.g., [4, 8, 16, 32]
    Returns
    --------
    labels:   Nx1 array in [0, num_classes]
    bbox_targets: N x (4) regression targets
    bbox_inside_weights: N x (4), in {0, 1} indicating to which class is assigned.
    """
    # TODO: speedup this module
    # if all_anchors is None:
    #   all_anchors = anchors_plane(height, width, stride=stride)

    # # anchors, inds_inside, total_anchors
    # border = cfg.FLAGS.allow_border
    # all_anchors = all_anchors.reshape((-1, 4))
    # inds_inside = np.where(
    #   (all_anchors[:, 0] >= -border) &
    #   (all_anchors[:, 1] >= -border) &
    #   (all_anchors[:, 2] < (width * stride) + border) &
    #   (all_anchors[:, 3] < (height * stride) + border))[0]
    # anchors = all_anchors[inds_inside, :]

    all_anchors = all_anchors.reshape([-1, 4])
    anchors = all_anchors
    total_anchors = all_anchors.shape[0]

    # labels = np.zeros((anchors.shape[0], ), dtype=np.float32)
    labels = np.empty((anchors.shape[0], ), dtype=np.int32)
    labels.fill(-1)

    jittered_gt_boxes = _jitter_gt_boxes(gt_boxes[:, :4])
    clipped_gt_boxes = clip_boxes(jittered_gt_boxes, (ih, iw))

    if gt_boxes.size > 0:
        overlaps = cython_bbox.bbox_overlaps(
                   np.ascontiguousarray(anchors, dtype=np.float),
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
        # LOG ("all_anchors anchor above threshold\n%s" %all_anchors[labels==1, :])

        # ignore cross-boundary anchors
        if ignore_cross_boundary is True:
            cb0_inds = np.where(all_anchors[:, 0] <= 0  - (all_anchors[:, 2] - all_anchors[:, 0]) * cfg.FLAGS.allow_border)
            cb1_inds = np.where(all_anchors[:, 1] <= 0  - (all_anchors[:, 3] - all_anchors[:, 1]) * cfg.FLAGS.allow_border)
            cb2_inds = np.where(all_anchors[:, 2] >= iw + (all_anchors[:, 2] - all_anchors[:, 0]) * cfg.FLAGS.allow_border)
            cb3_inds = np.where(all_anchors[:, 3] >= ih + (all_anchors[:, 3] - all_anchors[:, 1]) * cfg.FLAGS.allow_border)
            cb_inds = np.unique(np.concatenate((cb0_inds, cb1_inds, cb2_inds, cb3_inds), axis =1))
            labels[cb_inds] = -1
            #LOG ("stride: %d total anchor: %d\tremained anchor: %d\t ih:%d iw:%d min size %d %d \t max size %d %d" % (stride, total_anchors, total_anchors-len(cb_inds), ih, iw, np.min(all_anchors[:, 0]), np.min(all_anchors[:, 1]), np.max(all_anchors[:, 2]), np.max(all_anchors[:, 3])))
        # LOG ("above threshold: %s"% np.where(labels==1)) 
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        labels[gt_argmax_overlaps] = 1
        
        # LOG ("all_anchors anchor closest box\n%s" %all_anchors[labels==2, :])
        # LOG ("gt anchor\n%s" %gt_boxes)
        # LOG ("closest box: %s"% np.where(labels==2))
        # LOG ("stride: %d total anchor: %d\tremained anchor: %d\t ih:%d iw:%d min size %d %d \t max size %d %d" % (stride, total_anchors, total_anchors-len(cb_inds), ih, iw, np.min(all_anchors[labels!=-2, 0]), np.min(all_anchors[labels!=-2, 1]), np.max(all_anchors[labels!=-2, 2]), np.max(all_anchors[labels!=-2, 3])))

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
    num_bg = max(min(cfg.FLAGS.rpn_batch_size - num_fg, num_fg * 3), 2)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    bbox_targets = np.zeros((total_anchors, 4), dtype=np.float32)
    if gt_boxes.size > 0:
        bbox_targets = _compute_targets(anchors, gt_boxes[gt_assignment, :])
    bbox_inside_weights = np.zeros((total_anchors, 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = 1.0#0.1

    # # mapping to whole outputs
    # labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    # bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    # bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)

    labels = labels.reshape((1, height, width, -1))
    bbox_targets = bbox_targets.reshape((1, height, width, -1))
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, -1))

    return labels, bbox_targets, bbox_inside_weights

def decode(boxes, scores, all_anchors, ih, iw):
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
    # h, w = boxes.shape[1], boxes.shape[2]
    # if all_anchors is  None:
    #   stride = 2 ** int(round(np.log2((iw + 0.0) / w)))
    #   all_anchors = anchors_plane(h, w, stride=stride)
    all_anchors = all_anchors.reshape((-1, 4))
    boxes = boxes.reshape((-1, 4))
    scores = scores.reshape((-1, 2))
    assert scores.shape[0] == boxes.shape[0] == all_anchors.shape[0], \
      'Anchor layer shape error %d vs %d vs %d' % (scores.shape[0],boxes.shape[0],all_anchors.reshape[0])
    boxes = bbox_transform_inv(all_anchors, boxes)
    classes = np.argmax(scores, axis=1)
    scores = scores[:, 1]
    final_boxes = boxes  
    final_boxes = clip_boxes(final_boxes, (ih, iw))
    classes = classes.astype(np.int32)

    return final_boxes, classes, scores

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

if __name__ == '__main__':
  
    import time
    t = time.time()
    
    cfg.FLAGS.fg_threshold = 0.5
    # classes = np.ones((2,1))#random.randint(1, 1, (2, 1))
    # boxes = np.random.randint(10, 50, (2, 2))
    # s = np.random.randint(20, 50, (2, 2))
    # s = boxes + s
    # boxes = np.concatenate((boxes, s), axis=1)
    # gt_boxes = np.hstack((boxes, classes))
    # print(gt_boxes)

    gt_boxes = np.array([[0, 0, 5, 5],[6, 6, 8, 8]])
    print(gt_boxes)
    anchors = np.array([[-10,-10, 5, 5],[6, 6, 8, 8]])
    print(anchors)
    overlaps = cython_bbox.bbox_overlaps(
                   np.ascontiguousarray(anchors, dtype=np.float),
                   np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    print(overlaps)

    # all_anchors = anchors_plane(25, 37, stride = 32, scales=[8, 16, 32], ratios=[0.5, 1, 2.0], base=16)
    # print(all_anchors)
    # print(all_anchors.shape)
    # all_anchors = all_anchors.reshape([-1, 4])

    # for i in range(10):
    #     cfg.FLAGS.fg_threshold = 0.5
    #     classes = np.random.randint(0, 1, (50, 1))
    #     boxes = np.random.randint(10, 50, (50, 2))
    #     s = np.random.randint(20, 50, (50, 2))
    #     s = boxes + s
    #     boxes = np.concatenate((boxes, s), axis=1)
    #     gt_boxes = np.hstack((boxes, classes))
    #     # gt_boxes = boxes

    #     N = 100
    #     rois = np.random.randint(10, 50, (N, 2))
    #     s = np.random.randint(0, 20, (N, 2))
    #     s = rois + s
    #     rois = np.concatenate((rois, s), axis=1)
        
    #     indexs = np.arange(5*3*200*300)
    #     all_anchors = anchors_plane(200, 300, stride = 4, scales=[2, 4, 8, 16, 32], ratios=[0.5, 1, 2.0], base=16)
    #     labels, bbox_targets, bbox_inside_weights, indexs = encode(gt_boxes, all_anchors=all_anchors, height=200, width=300, stride=4, indexs=indexs)

    #     indexs = np.arange(5*3*100*150)
    #     all_anchors = anchors_plane(100, 150, stride = 8, scales=[2, 4, 8, 16, 32], ratios=[0.5, 1, 2.0], base=16)
    #     labels, bbox_targets, bbox_inside_weights, indexs  = encode(gt_boxes, all_anchors=all_anchors, height=100, width=150, stride=8, indexs=indexs)

    #     indexs = np.arange(5*3*50*75)
    #     all_anchors = anchors_plane(50, 75, stride = 16, scales=[2, 4, 8, 16, 32], ratios=[0.5, 1, 2.0], base=16)
    #     labels, bbox_targets, bbox_inside_weights, indexs  = encode(gt_boxes, all_anchors=all_anchors, height=50, width=75, stride=16, indexs=indexs)

    #     indexs = np.arange(5*3*25*37)
    #     all_anchors = anchors_plane(25, 37, stride = 32, scales=[2, 4, 8, 16, 32], ratios=[0.5, 1, 2.0], base=16)
    #     labels, bbox_targets, bbox_inside_weights, indexs  = encode(gt_boxes, all_anchors=all_anchors, height=25, width=37, stride=32, indexs=indexs)
    #     # anchors, _, _ = anchors_plane(200, 300, stride=4, boarder=0)
  
    # print('average time: %f' % ((time.time() - t)/10.0))

def _jitter_gt_boxes(gt_boxes, jitter=0.05):
    """ jitter the gtboxes, before adding them into rois, to be more robust for cls and rgs
    gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
    """
    jittered_boxes = gt_boxes.copy()
    ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
    hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
    width_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * ws
    height_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * hs
    jittered_boxes[:, 0] += width_offset
    jittered_boxes[:, 2] += width_offset
    jittered_boxes[:, 1] += height_offset
    jittered_boxes[:, 3] += height_offset

    return jittered_boxes