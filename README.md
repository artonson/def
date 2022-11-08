# DEF: Deep Estimation of Sharp Geometric Features in 3D Shapes

**SIGGRAPH 2022 [[Project Page](https://artonson.github.io/publications/def)] [[Arxiv](https://arxiv.org/abs/2011.15081)] [[Bibtex](docs/bib.bib)]**

This is an official implementation of the paper 
_Albert Matveev, Ruslan Rakhimov, Alexey Artemov, Gleb Bobrovskikh, Vage Egiazarian, Emil Bogomolov, Daniele Panozzo, Denis Zorin, and Evgeny Burnaev. "DEF: Deep estimation of sharp geometric features in 3D shapes". ACM Trans. Graph. 41, 4, Article 108 (July 2022), 22 pages._

![Teaser Image](docs/images/teaser.jpg)


## :construction: Construction Alert! :construction:

We are currently in the process of updating and release the source code for our project. 
As soon as we finish this housekeeping, this message will disappear.
If you would like to use our method now, please email the corresponding authors of the paper.


## Installation
The preferred way of setting up the environment required to run out method
is by building and using a docker image. The docker image contains all the 
necessary software required to run, debug, and develop the components of 
our approach and its constituent parts. We thus strongly recommend using
either a pre-built docker image available on DockerHub, or using the 
ready-made scripts to build your own docker image locally.

To build the docker image from the source code available in this repository,
run the command:
```bash
bash docker/build_docker.sh 
```
To get the docker image from DockerHub, run the command
```bash
docker pull artonson/def:latest
```


## Getting started
Below, we enumerate the major steps required for our method to work, and 
provide the links to the respective documentation. To get familiar with more
details of how our method works, please refer to the respective documentation
pages, the source code, contact the authours via [artonson at yandex ru],
or [open an issue](https://github.com/artonson/def/issues/new).
 * [Generating synthetic training datasets](https://github.com/artonson/def/blob/main/docs/synthetic_data.md)
 * [Generating real-world training datasets](https://github.com/artonson/def/blob/main/docs/real_data.md)
 * [Training point-based and image-based DEF networks](https://github.com/artonson/def/blob/main/docs/training.md)
 * [Running inference with a pre-trained DEF network](https://github.com/artonson/def/blob/main/docs/inference.md)
 * [Running reconstruction on complete 3D models](https://github.com/artonson/def/blob/main/docs/fusion.md)
 * [Running extraction of parametric curves](https://github.com/artonson/def/blob/main/docs/parametric.md)


## Pre-trained models
We provide a variety of pre-trained DEF networks (both image-based and point-based).
The table below summarizes these models and provides links for downloading 
the respective weight files.

| **Link** | **Modality**   | **Resolution** | **Noise level** | **Trained on** | 
|----------|----------------|----------------|-----------------|----------------|
 | [x](y)   | Image-based | 0.02           |                 | DEF-Sim |


## Training and evaluation datasets


## Citing

```
@article{10.1145/3528223.3530140,
author = {Matveev, Albert and Rakhimov, Ruslan and Artemov, Alexey and Bobrovskikh, Gleb and Egiazarian, Vage and Bogomolov, Emil and Panozzo, Daniele and Zorin, Denis and Burnaev, Evgeny},
title = {DEF: Deep Estimation of Sharp Geometric Features in 3D Shapes},
year = {2022},
issue_date = {July 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {41},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3528223.3530140},
doi = {10.1145/3528223.3530140},
journal = {ACM Trans. Graph.},
month = {jul},
articleno = {108},
numpages = {22},
keywords = {curve extraction, sharp geometric features, deep learning}
}
```


## Acknowledgements




