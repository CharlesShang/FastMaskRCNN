#!/usr/bin/env python 
# coding=utf-8

import numpy as np
import sys
import os
import tensorflow as tf 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libs.boxes.roi import roi_cropping
from libs.layers import anchor_encoder
from libs.layers import anchor_decoder
from libs.layers import roi_encoder
from libs.layers import roi_decoder
from libs.layers import mask_encoder
from libs.layers import mask_decoder
from libs.layers import gen_all_anchors
from libs.layers import ROIAlign
from libs.layers import sample_rpn_outputs
from libs.layers import assign_boxes
import  libs.configs.config_v1 as cfg

class layer_test(object):

    def __init__(self, N, num_classes, height, width, gt_boxes=None, gt_masks=None, rois=None, classes=None):
        self.N = N
        self.num_classes = num_classes 
        self.height = height
        self.width = width 
        if gt_boxes is not None:
            self.gt_boxes = gt_boxes
            self.gt_masks = gt_masks 
            self.rois = rois 
            self.classes = classes
        else:
            self.gt_boxes = np.random.randint(0, 50, (self.N, 2))
            s = np.random.randint(30, 40, (self.N, 2))
            c = np.random.randint(1, self.num_classes, (self.N, 1))
            self.gt_boxes = np.hstack((self.gt_boxes, self.gt_boxes + s, c))
            self.gt_boxes = self.gt_boxes.astype(np.float32)
            gt_masks = np.zeros((self.N, height, width), dtype=np.int32)
            for i in range(self.N):
                gt_masks[i, int(self.gt_boxes[i, 1]):int(self.gt_boxes[i, 3]), int(self.gt_boxes[i, 0]):int(self.gt_boxes[i, 2])] = 1
            self.gt_masks = gt_masks

            # rois 
            noises = np.random.randint(-3, 3, (self.N, 4))
            self.rois = self.gt_boxes[:, :4] + noises 

            # classes 
            self.classes = self.gt_boxes[:, -1]

    def test(self):
        return


class anchor_test(layer_test):

    def __init__(self, N, num_classes, height, width, gt_boxes=None, gt_masks=None, rois=None, classes=None):
        super(anchor_test, self).__init__(N, num_classes, height, width, gt_boxes, gt_masks, rois, classes)
    def test(self):
        cfg.FLAGS.fg_threshold = 0.7
        with tf.Session() as sess:
            all_anchors = gen_all_anchors(self.height / 4, self.width / 4, stride = 4, scales = 2**np.arange(1,5))
            all_anchors = tf.reshape(all_anchors, [-1, 4])
            self.all_anchors =  np.reshape(all_anchors.eval(), (-1, 4))
            labels, bbox_targets, bbox_inside_weights = \
                    anchor_encoder(self.gt_boxes, all_anchors, self.height / 4, self.width / 4, 4)
            self.labels = labels.eval()
            self.bbox_targets = bbox_targets.eval()
            self.bbox_inside_weights = bbox_inside_weights.eval()
            print (self.labels.shape)
            print (self.bbox_targets.shape)
            print (self.bbox_inside_weights.shape)
            print (self.gt_boxes)
            # print (self.all_anchors[0:120:15, ])
            np_labels = self.labels.reshape((-1,))
            np_bbox_targets = self.bbox_targets.reshape((-1, 4))
            np_bbox_inside_weights = self.bbox_inside_weights.reshape((-1, 4))
            encoded_gt_boxes = []
            for i in range(np_labels.shape[0]):
                if np_labels[i] >= 1:
                    # print (self.all_anchors[i, :], np_bbox_targets[i, :], np_bbox_inside_weights[i, :])
                    encoded_gt_boxes.append (np_bbox_targets[i, :])
            encoded_gt_boxes = np.asarray(encoded_gt_boxes, dtype = np.float32)
            encoded_gt_boxes = encoded_gt_boxes.reshape((-1, 4))
            # print (np.max(np_labels))
            # print (np.sum(np_labels >= 1))
            scores = np.zeros((np_labels.shape[0], 2), dtype=np.float32)
            for i in range(np_labels.shape[0]):
                if np_labels[i] > 0:
                    scores[i, 0] = 0
                    scores[i, 1] = 1
            scores = scores.astype(np.float32)
            boxes, classes, scores = \
                    anchor_decoder(self.bbox_targets, scores, all_anchors, self.height, self.width)
            self.npboxes = boxes.eval().reshape((-1, 4))
            npscores = scores.eval().reshape((-1, 1))
            self.npboxes = np.hstack((self.npboxes, npscores))
            # print (self.npboxes.shape, npscores.shape)
            bbox_targets_np = self.bbox_targets.reshape([-1, 4])
            all_anchors_np = all_anchors.eval().reshape([-1, 4])
            for i in range(self.npboxes.shape[0]):
                if self.npboxes[i, 4] >= 1:
                    print (bbox_targets_np[i], self.npboxes[i], all_anchors_np[i])

class roi_test(layer_test):

    def __init__(self, N, num_classes, height, width, gt_boxes=None, gt_masks=None, rois=None, classes=None):
        super(roi_test, self).__init__(N, num_classes, height, width, gt_boxes, gt_masks, rois, classes)

    def test(self):
        import time
        print (self.gt_boxes)
        # time.sleep(10)
        with tf.Session() as sess:
            rois = self.gt_boxes[:, :4]
            rois = rois + np.random.randint(-3, 3, (self.N, 4))
            bgs = np.random.randint(0, 60, (self.N + 2, 2))
            bgs = np.hstack((bgs, bgs + np.random.randint(20, 30, (self.N + 2, 2))))
            bgs = bgs.astype(np.float32)
            rois = np.vstack((rois, bgs))
            self.rois = rois
            print (rois)
            print (self.gt_boxes)
            labels, bbox_targets, bbox_inside_weights = \
                    roi_encoder(self.gt_boxes, self.rois, self.num_classes)
            self.labels = labels.eval()
            self.bbox_targets = bbox_targets.eval()
            self.bbox_inside_weights = bbox_inside_weights.eval()

            print (self.labels.shape)
            print (self.labels)
            print (self.bbox_targets.shape)
            print (self.bbox_inside_weights.shape)
            print ('learning targets:')
            for i in range(self.labels.size):
                s = int(4 * self.labels[i])
                e = s + 4
                print (self.labels[i], self.bbox_targets[i, s:e], self.bbox_inside_weights[i, s:e])

            scores = np.random.rand(self.rois.shape[0], self.num_classes)
            scores = scores.astype(np.float32)
            final_boxes, classes, scores = \
                    roi_decoder(self.bbox_targets, scores, self.rois, 100, 100)
            self.final_boxes = final_boxes.eval()
            self.scores = scores.eval()
            self.classes = classes.eval()
            print ('rois:')
            print (self.rois)
            print ('final_boxes:')
            print (self.final_boxes)


class mask_test(layer_test):

    def __init__(self, N, num_classes, height, width, gt_boxes=None, gt_masks=None, rois=None, classes=None):
        super(mask_test, self).__init__(N, num_classes, height, width, gt_boxes, gt_masks, rois, classes)

    def test(self):
        # mask 
        with tf.Session() as sess:
            gt_masks = np.zeros((self.N, 100, 100), dtype=np.int32)
         
            rois = self.gt_boxes[:, :4]
            rois = rois + np.random.randint(-5, 5, (self.N, 4))
            rois[rois < 0] = 0
            bgs = np.random.randint(0, 60, (self.N + 2, 2))
            bgs = np.hstack((bgs, bgs + np.random.randint(20, 30, (self.N + 2, 2))))
            bgs = bgs.astype(np.float32)
            rois = np.vstack((rois, bgs))
            print (rois)

            for i in range(self.N):
                x1, y1 = int(self.gt_boxes[i, 0] + 2), int(self.gt_boxes[i, 1] + 2)
                x2, y2 = int(self.gt_boxes[i, 2] - 1), int(self.gt_boxes[i, 3] - 1)
                gt_masks[i, y1:y2, x1:x2] = 1
            self.gt_masks = gt_masks

            labels, mask_targets, mask_inside_weights = \
                    mask_encoder(self.gt_masks, self.gt_boxes, rois, self.num_classes, 15, 15)
            self.labels = labels.eval()
            self.mask_targets = mask_targets.eval()
            self.mask_inside_weights = mask_inside_weights.eval()

            # print (self.mask_targets)
            print (self.labels)
            for i in range(rois.shape[0]):
                print(i, 'label:', self.labels[i])
                print (self.mask_targets[i, :, :, int(self.labels[i])])

class sample_test(layer_test):

    def __init__(self, N, num_classes, height, width, gt_boxes=None, gt_masks=None, rois=None, classes=None):
        super(sample_test, self).__init__(N, num_classes, height, width, gt_boxes, gt_masks, rois, classes)

    def test(self):
        with tf.Session() as sess:
            boxes = np.random.randint(0, 50, [self.N, 2])
            s = np.random.randint(20, 30, [self.N, 2])
            boxes = np.hstack((boxes, boxes + s)).astype(np.float32)

            scores = np.random.rand(self.N, 1).astype(np.float32)
            boxes, scores, batch_inds= \
                    sample_rpn_outputs(boxes, scores, is_training=False,) 
            self.boxes = boxes.eval()
            self.scores = scores.eval()
            bs = np.hstack((self.boxes, self.scores))
            np.set_printoptions(precision=3, suppress=True)
            print (bs)

class ROIAlign_test(layer_test):

    def __init__(self, N, num_classes, height, width, gt_boxes=None, gt_masks=None, rois=None, classes=None):
        super(ROIAlign_test, self).__init__(N, num_classes, height, width, gt_boxes, gt_masks, rois, classes)

    def test(self):
        with tf.Session() as sess:
            npimg = np.random.rand(1, self.height, self.width, 2).astype(np.float32)
            npimg = np.zeros((1, self.height, self.width, 1), dtype=np.float32)

            boxes = np.random.randint(0, 50, [self.N, 2])
            s = np.random.randint(20, 30, [self.N, 2])
            boxes = np.hstack((boxes, boxes + s)).astype(np.float32)

            stride = 2.0 
            for i in range(self.N):
                b = boxes[i, :] / stride
                npimg[:, 
                      int(b[1]):int(b[3]+1),
                      int(b[0]):int(b[2]+1),
                      :] = 1 
                
            img = tf.constant(npimg)
            pooled_height = 5
            pooled_width = 5
            batch_inds = np.zeros((self.N, ), dtype=np.int32)
            batch_inds = tf.convert_to_tensor(batch_inds)
            feats = ROIAlign(img, boxes, batch_inds, stride=stride, pooled_height=pooled_height, pooled_width=pooled_width,)
            self.feats = feats.eval()
            print (self.feats.shape)
            print (self.feats.reshape((self.N, pooled_height, pooled_width)))

class assign_test(layer_test):

    def __init__(self, N, num_classes, height, width, gt_boxes=None, gt_masks=None, rois=None, classes=None):
        super(assign_test, self).__init__(N, num_classes, height, width, gt_boxes, gt_masks, rois, classes)
    def test(self):

        self.gt_boxes = np.random.randint(0, int(self.width/1.5), (self.N, 2))
        s = np.random.randint(30, int(self.width/1), (self.N, 2))
        c = np.random.randint(1, self.num_classes, (self.N, 1))
        self.gt_boxes = np.hstack((self.gt_boxes, self.gt_boxes + s, c))
        batch_inds = np.zeros((self.N, ), np.int32)
        with tf.Session() as sess:

            batch_inds = tf.convert_to_tensor(batch_inds)
            [assigned_boxes, assigned_batch, inds] = \
                assign_boxes(self.gt_boxes, [self.gt_boxes, batch_inds], [2,3,4,5])
            [b1, b2, b3, b4] = assigned_boxes
            [ind1, ind2, ind3, ind4] = assigned_batch
            b1n, b2n, b3n, b4n, indsn, ind1n, ind2n, ind3n, ind4n= \
                    sess.run([b1, b2, b3, b4, inds, ind1, ind2, ind3, ind4])
            print (b1n)
            print (b2n)
            print (b3n)
            print (b4n)
            print (np.hstack((self.gt_boxes, indsn[:, np.newaxis])))

            print (ind1n, ind2n, ind3n, ind4n)

if __name__ == '__main__':
    print ('##############################')
    print ('Anchor Test')
    print ('##############################')
    an_test = anchor_test(0, 5, 100, 100)
    an_test.test()
    an_test = anchor_test(5, 5, 100, 100)
    an_test.test()
    print ('##############################')
    print ('ROI Test')
    print ('##############################')
    r_test  = roi_test(10, 9, 100, 100)
    r_test.test()
    r_test  = roi_test(0, 9, 100, 100)
    r_test.test()
    print ('##############################')
    print ('Mask Test')
    print ('##############################')
    m_test  = mask_test(5, 4, 100, 100)
    m_test.test()
    m_test  = mask_test(0, 4, 100, 100)
    m_test.test()
    print ('##############################')
    print ('Sample Test')
    print ('##############################')
    s_test  = sample_test(20, 4, 100, 100)
    s_test.test()
    print ('##############################')
    print ('ROIAlign Test')
    print ('##############################')
    c_test  = ROIAlign_test(8, 4, 100, 100)
    c_test.test()
    c_test  = ROIAlign_test(0, 4, 100, 100)
    c_test.test()
    print ('##############################')
    print ('Assign Test')
    print ('##############################')
    c_test  = assign_test(15, 2, 800, 800)
    c_test.test()
