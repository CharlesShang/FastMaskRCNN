from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import libs.configs.config_v1 as cfg

def LOG(mssg):
  logging.basicConfig(filename=cfg.FLAGS.train_dir + '/maskrcnn.log',
                      level=logging.INFO,
                      datefmt='%m/%d/%Y %I:%M:%S %p', format='%(asctime)s %(message)s')
  logging.info(mssg)