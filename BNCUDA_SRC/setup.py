from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
setup(name='BatchNorm',
      ext_modules=[CppExtension('BatchNorm', ['BN.cpp'])],
      cmdclass={'build_ext': BuildExtension})