#!/usr/bin/env python

from setuptools import setup, find_packages

__version__ = '0.1.0'
url = 'https://github.com/artonson/sharp_features'
install_requires = [
    'hydra-core',
    'pytorch-lightning'
]

setup(name='defs',
      version=__version__,
      description='Deep Estimation of Sharp Geometric Features in 3D Shapes',
      author='adase-3ddl',
      author_email='',
      url=url,
      install_requires=install_requires,
      packages=find_packages()
      )
