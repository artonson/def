# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified by Ze Liu

import glob
import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = "_ext_src"
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(osp.join(_ext_src_root, "src", "*.cu"))
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

setup(
    name='pt_custom_ops',
    ext_modules=[
        CUDAExtension(
            name='pt_custom_ops._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", f"-I{osp.join(this_dir, _ext_src_root, 'include')}"],
                "nvcc": ["-O2", f"-I{osp.join(this_dir, _ext_src_root, 'include')}"],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
