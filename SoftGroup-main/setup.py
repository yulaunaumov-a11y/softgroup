import os
import sys
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Sparsehash include path
sparsehash_include = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'third_party', 'sparsehash', 'src')

# Platform-specific compile args
if sys.platform == 'win32':
    cxx_args = ['/O2', '/std:c++17', '/Zc:preprocessor']
    nvcc_args = ['-O2', '--std=c++17', '-allow-unsupported-compiler', '-Xcompiler', '/Zc:preprocessor']
else:
    cxx_args = ['-g']
    nvcc_args = ['-O2']

if __name__ == '__main__':
    setup(
        name='softgroup',
        version='1.0',
        description='SoftGroup: SoftGroup for 3D Instance Segmentation [CVPR 2022]',
        author='Thang Vu',
        author_email='thangvubk@kaist.ac.kr',
        packages=['softgroup'],
        package_data={'softgroup.ops': ['*/*.so', '*/*.pyd']},
        ext_modules=[
            CUDAExtension(
                name='softgroup.ops.ops',
                sources=[
                    'softgroup/ops/src/softgroup_api.cpp', 'softgroup/ops/src/softgroup_ops.cpp',
                    'softgroup/ops/src/cuda.cu'
                ],
                include_dirs=[
                    sparsehash_include,
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'softgroup', 'ops', 'src'),
                ] + ([
                    os.path.join(os.environ.get('CUDA_HOME', ''),
                                 'include', 'targets', 'x64'),
                    os.path.join(os.environ.get('CUDA_HOME', ''),
                                 'include', 'targets', 'x64', 'cccl'),
                ] if sys.platform == 'win32' else []),
                extra_compile_args={
                    'cxx': cxx_args,
                    'nvcc': nvcc_args
                })
        ],
        cmdclass={'build_ext': BuildExtension})
