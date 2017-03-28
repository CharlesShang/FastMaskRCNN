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

def anchors_plane(
        int height, int width, int stride,
        np.ndarray[DTYPE_t, ndim=2] anchors_base):
    """
    Parameters
    ----------
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: (A, 4) a base set of anchors
    Returns
    -------
    all_anchors: (height, width, A, 4) ndarray of anchors spreading over the plane
    """
    cdef unsigned int A = anchors_base.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=4] all_anchors = np.zeros((height, width, A, 4), dtype=DTYPE)
    cdef unsigned int iw, ih
    cdef unsigned int k
    cdef unsigned int A4
    cdef unsigned int sh
    cdef unsigned int sw
    A4 = A*4
    for iw in range(width):
        sw = iw * stride
        for ih in range(height):
            sh = ih * stride
            for k in range(A):
                all_anchors[ih, iw, k, 0] = anchors_base[k, 0] + sw
                all_anchors[ih, iw, k, 1] = anchors_base[k, 1] + sh
                all_anchors[ih, iw, k, 2] = anchors_base[k, 2] + sw
                all_anchors[ih, iw, k, 3] = anchors_base[k, 3] + sh
    return all_anchors