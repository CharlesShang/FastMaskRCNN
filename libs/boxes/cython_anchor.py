from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, pkg_resources, imp

def __bootstrap__():
  global __bootstrap__, __loader__, __file__
  __file__ = pkg_resources.resource_filename(__name__, 'cython_anchor.so')
  __loader__ = None
  del __bootstrap__, __loader__
  imp.load_dynamic(__name__, __file__)

__bootstrap__()