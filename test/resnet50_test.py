#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.logs.log import LOG
import libs.nets.resnet_v1 as resnet_v1
resnet50 = resnet_v1.resnet_v1_50

images = tf.placeholder(tf.float32, [16, 224, 224, 3], name='image')
logits, end_points = resnet50(images, 1000, is_training=False)
end_points['inputs'] = images

for x in sorted(end_points.keys()):
  print (x, end_points[x].name, end_points[x].shape)
  
import libs.nets.pyramid_network as pyramid_network
pyramid = pyramid_network.build_pyramid('resnet50', end_points)
for p in pyramid:
  print (p, pyramid[p])

outputs = pyramid_network.build_head(pyramid, num_classes=81, base_anchors=15, is_training=True)
