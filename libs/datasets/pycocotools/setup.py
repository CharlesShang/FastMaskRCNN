from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

ext_modules = [

	Extension(
        "_mask",
        sources=['common/maskApi.c', '_mask.pyx'],
        include_dirs = [np.get_include(), 'common'],
        extra_compile_args={'gcc': ['/Qstd=c99']},
    )
]

setup(name='pycocotools',
      packages=['pycocotools'],
      package_dir = {'pycocotools': 'pycocotools'},
      version='2.0',
      ext_modules=
          cythonize(ext_modules)
      )