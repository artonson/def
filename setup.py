#!/usr/bin/env python

from setuptools import setup, find_packages

__version__ = '0.0.1'
url = 'https://github.com/artonson/sharp_features'
install_requires = [
    'hydra-core',
    'pytorch-lightning',
    'msgpack-python'
]

setup(name='sharpf',
      version=__version__,
      description='sharp_features',
      author='3ddl',
      author_email='',
      url=url,
      install_requires=install_requires,
      packages=find_packages()
      )
