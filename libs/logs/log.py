from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

def LOG(mssg):
  logging.basicConfig(filename='maskrcnn.log', level=logging.INFO,
                      datefmt='%m/%d/%Y %I:%M:%S %p', format='%(asctime)s %(message)s')
  logging.info(mssg)