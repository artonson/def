from setuptools import setup, find_packages

__version__ = '0.0.1'
url = 'https://github.com/artonson/sharp_features'

install_requires = [
    'torch-geometric',
    'torch-scatter',
    'torch-sparse',
    'torch-cluster',
    'torch-spline-conv',
]

setup(
    name='sharp_features',
    version=__version__,
    author='Alexey Artemov',
    author_email='artonson@yandex.ru',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch', 'geometric-deep-learning',
        'neural-networks', 'point-clouds'
    ],
    install_requires=install_requires,
    packages=find_packages())
