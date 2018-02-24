from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

##########################
#                  restore
##########################
tf.app.flags.DEFINE_string(
    'train_dir', './output/mask_rcnn/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'pretrained_model', './data/pretrained_models/resnet_v1_50.ckpt',
    'Path to pretrained model')

##########################
#                  network
##########################
tf.app.flags.DEFINE_string(
    'network', 'resnet50',
    'name of backbone network')

##########################
#                  dataset
##########################
tf.app.flags.DEFINE_bool(
    'update_bn', False,
    'Whether or not to update bacth normalization layer')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'coco',
    'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train2014',
    'The name of the train/test/val split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', 'data/coco/',
    'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'im_batch', 1,
    'number of images in a mini-batch')


tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 60,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 7200,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'max_iters', 2500000,
    'max iterations')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00005, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'momentum',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.99,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.99, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.99, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type', 'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.002,
                          'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################


tf.app.flags.DEFINE_string(
    'model_name', 'resnet50',
    'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'coco',
    'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1,
    'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_string(
    'classes', None,
    'The classes to classify.')

tf.app.flags.DEFINE_integer(
    'image_min_size', 640,
    'resize image so that the min edge equals to image_min_size')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'checkpoint_include_scopes', None,
    'Comma-separated list of scopes of variables to include when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_boolean(
    'restore_previous_if_exists', True,
    'When restoring a checkpoint would ignore missing variables.')

#######################
# BOX Flags #
#######################
tf.app.flags.DEFINE_float(
    'rpn_bg_threshold', 0.3,
    'Only regions which intersection is larger than fg_threshold are considered to be fg')

tf.app.flags.DEFINE_float(
    'rpn_fg_threshold', 0.7,
    'Only regions which intersection is larger than fg_threshold are considered to be fg')

tf.app.flags.DEFINE_float(
    'fg_threshold', 0.7,
    'Only regions which intersection is larger than fg_threshold are considered to be fg')

tf.app.flags.DEFINE_float(
    'bg_threshold', 0.3,
    'Only regions which intersection is less than bg_threshold are considered to be bg')

tf.app.flags.DEFINE_integer(
    'rois_per_image', 256,
    'Number of rois that should be sampled to train this network')

tf.app.flags.DEFINE_float(
    'fg_roi_fraction', 0.25,
    'Number of rois that should be sampled to train this network')

tf.app.flags.DEFINE_float(
    'fg_rpn_fraction', 0.25,
    'Number of rois that should be sampled to train this network')

tf.app.flags.DEFINE_integer(
    'rpn_batch_size', 500,
    'Number of rpn anchors that should be sampled to train this network')

tf.app.flags.DEFINE_integer(
    'allow_border', 10,
    'How many pixels out of an image')

##################################
#            NMS                #
##################################

tf.app.flags.DEFINE_integer(
    'pre_nms_top_n', 12000,
    'Number of rpn anchors that should be sampled before nms')

tf.app.flags.DEFINE_integer(
    'post_nms_top_n', 2000,
    'Number of rpn anchors that should be sampled after nms')

tf.app.flags.DEFINE_float(
    'rpn_nms_threshold', 0.7,
    'NMS threshold')

##################################
#            Mask                #
##################################

tf.app.flags.DEFINE_boolean(
    'mask_allow_bg', True,
    'Allow to add bg masks in the masking stage')

tf.app.flags.DEFINE_float(
    'mask_threshold', 0.50,
    'Least intersection of a positive mask')
tf.app.flags.DEFINE_integer(
    'masks_per_image', 64,
    'Number of rois that should be sampled to train this network')

tf.app.flags.DEFINE_float(
    'min_size', 2,
    'minimum size of an object')

FLAGS = tf.app.flags.FLAGS
