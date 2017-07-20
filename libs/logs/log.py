from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import libs.configs.config_v1 as cfg
import os

def LOG(mssg):
  if not os.path.exists(cfg.FLAGS.train_dir):
    os.makedirs(cfg.FLAGS.train_dir)
  logging.basicConfig(filename=cfg.FLAGS.train_dir + '/maskrcnn.log',
                      level=logging.INFO,
                      datefmt='%m/%d/%Y %I:%M:%S %p', format='%(asctime)s %(message)s')
  logging.info(mssg)
