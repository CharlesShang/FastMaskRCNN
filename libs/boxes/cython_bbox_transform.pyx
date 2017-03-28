# --------------------------------------------------------
# Mask RCNN
# Licensed under The MIT License [see LICENSE for details]
# Written by CharlesShang@github
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t
# ctypedef float DTYPE_t

#def bbox_transform(
#        np.ndarray[DTYPE_t, ndim=2] ex_rois,
#        np.ndarray[DTYPE_t, ndim=2] gt_rois):
def bbox_transform(
        np.ndarray[DTYPE_t, ndim=2] ex_rois,
        np.ndarray[DTYPE_t, ndim=2] gt_rois):
    """
    Parameters
    ----------
    ex_rois: n * 4 numpy array, given boxes
    gt_rois: n * 4 numpy array, ground-truth boxes
    Returns
    -------
    targets: (n, 4) ndarray
    """
    cdef unsigned int R = ex_rois.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] targets = np.zeros((R, 4), dtype=DTYPE)
    cdef unsigned int i
    cdef DTYPE_t gt_w
    cdef DTYPE_t gt_h
    cdef DTYPE_t gt_cx
    cdef DTYPE_t gt_cy
    cdef DTYPE_t ex_w
    cdef DTYPE_t ex_h
    cdef DTYPE_t ex_cx
    cdef DTYPE_t ex_cy
    for i in range(R):
        gt_w = gt_rois[i, 2] - gt_rois[i, 0] + 1.0
        gt_h = gt_rois[i, 3] - gt_rois[i, 1] + 1.0
        ex_w = ex_rois[i, 2] - ex_rois[i, 0] + 1.0
        ex_h = ex_rois[i, 3] - ex_rois[i, 1] + 1.0
        gt_cx = gt_rois[i, 0] + gt_w * 0.5
        gt_cy = gt_rois[i, 1] + gt_h * 0.5
        ex_cx = ex_rois[i, 0] + ex_w * 0.5
        ex_cy = ex_rois[i, 1] + ex_h * 0.5
        targets[i, 0] = (gt_cx - ex_cx) / ex_w
        targets[i, 1] = (gt_cy - ex_cy) / ex_h
        targets[i, 2] = np.log(gt_w / ex_w)
        targets[i, 3] = np.log(gt_h / ex_h)
    return targets

cdef inline DTYPE_t my_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b
cdef inline DTYPE_t my_min(DTYPE_t a, DTYPE_t b): return a if a <= b else b

def bbox_transform_inv(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] deltas):
    """
    Parameters
    ----------
    boxes: n * 4 numpy array, given boxes
    deltas: (n, kx4) numpy array,
    Returns
    -------
    pred_boxes: (n, kx4) ndarray
    """
    cdef unsigned int R = boxes.shape[0]
    cdef unsigned int k4 = deltas.shape[1]
    cdef unsigned int k
    k = k4 / 4
    cdef np.ndarray[DTYPE_t, ndim=2] pred_boxes = np.zeros((R, k4), dtype=DTYPE)
    if R == 0:
        return pred_boxes

    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int j4
    cdef DTYPE_t w
    cdef DTYPE_t h
    cdef DTYPE_t cx
    cdef DTYPE_t cy
    cdef DTYPE_t px
    cdef DTYPE_t py
    cdef DTYPE_t pw
    cdef DTYPE_t ph
    for i in range(R):
        w = boxes[i, 2] - boxes[i, 0] + 1.0
        h = boxes[i, 3] - boxes[i, 1] + 1.0
        cx = boxes[i, 0] + w * 0.5
        cy = boxes[i, 1] + h * 0.5
        for j in range(k):
            j4 = j * 4
            px = deltas[i, j4    ] * w + cx
            py = deltas[i, j4 + 1] * h + cy
            pw = np.exp(deltas[i, j4 + 2]) * w
            ph = np.exp(deltas[i, j4 + 3]) * h
            pred_boxes[i, j4    ] = px - 0.5 * pw
            pred_boxes[i, j4 + 1] = py - 0.5 * ph
            pred_boxes[i, j4 + 2] = px + 0.5 * pw
            pred_boxes[i, j4 + 3] = py + 0.5 * ph
    return pred_boxes

def clip_boxes(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=1] im_shape):
    """
    Parameters
    ----------
    boxes: (n ,kx4) numpy array, given boxes
    im_shape:(2,) numpy array, (image_height, image_width)
    Returns
    -------
    clipped: (n, kx4) ndarray
    """
    cdef unsigned int R = boxes.shape[0]
    cdef unsigned int k4 = boxes.shape[1]
    cdef unsigned int k  = k4 / 4
    cdef np.ndarray[DTYPE_t, ndim=2] clipped = np.zeros((R, k4), dtype=DTYPE)
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int j4
    for i in range(R):
        for j in range(k):
            j4 = j * 4
            clipped[i, j4    ] = my_max(my_min(boxes[i, j4    ], im_shape[1]-1), 0)
            clipped[i, j4 + 1] = my_max(my_min(boxes[i, j4 + 1], im_shape[0]-1), 0)
            clipped[i, j4 + 2] = my_max(my_min(boxes[i, j4 + 2], im_shape[1]-1), 0)
            clipped[i, j4 + 3] = my_max(my_min(boxes[i, j4 + 3], im_shape[0]-1), 0)
    return clipped