from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("CWTextrema", ["CWTextrema.pyx"],
              include_dirs=[np.get_include()]),
]

setup(
    name="CWTextrema",
    ext_modules=cythonize(extensions, annotate=True),
    )