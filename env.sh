#!/bin/bash

# name of the project
export PROJECT="def"
# name of the docker image
export IMAGE_NAME="artonson/${PROJECT}"
# version of the docker image
export IMAGE_VERSION="latest"
# full tag of the docker image to checkotu
export IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"

# path to singularity images on skoltech cluster
export SIMAGES_DIR=/gpfs/gpfs0/3ddl/singularity-images
# full path of the singularity filename on the skoltech cluster
export SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"
