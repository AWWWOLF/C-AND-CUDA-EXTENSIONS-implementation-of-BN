from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='BatchNormCuda',
    ext_modules=[
        CUDAExtension('BatchNormCuda', [
            'bn_cuda.cpp',
            'bn_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })