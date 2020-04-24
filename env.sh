#!/bin/bash

# obligatory docker/singularity container namings
export PROJECT="sharp_features"
export SIMAGES_DIR=/gpfs/gpfs0/3ddl/singularity-images
export IMAGE_NAME="artonson/${PROJECT}"
export IMAGE_VERSION="latest"
#export IMAGE_VERSION="pointweb-ops"
export IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
export SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"

