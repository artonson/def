#!/bin/bash

set -e
set -x

# example launch string:
# ./build_singularity.sh [-f]
#     -f:       overwrite the existing singularity image (false by default)

module load apps/singularity-3.2.0

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd )"
source "${PROJECT_ROOT}"/env.sh

[[ -d ${SIMAGES_DIR} ]] || mkdir "${SIMAGES_DIR}"

IMAGE_NAME="artonson/sharp_features"
IMAGE_VERSION="latest"
IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"

echo "******* PULLING IMAGE FROM DOCKER HUB AND BUILDING SINGULARITY IMAGE *******"
if echo $* | grep -e "-f" -q
then
    singularity pull -F "${SIMAGE_FILENAME}" "docker://${IMAGE_NAME_TAG}"
else
    singularity pull "${SIMAGE_FILENAME}" "docker://${IMAGE_NAME_TAG}"
fi
