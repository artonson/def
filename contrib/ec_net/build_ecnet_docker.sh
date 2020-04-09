#!/bin/bash

set -e

# example launch string:#
# ./build_ecnet_docker.sh -u <dockerhub username> [-p]
# 	 -u:  dockerhub username to create an image under
#	 -p:  push the build image to the dockerhub under given username

usage() { echo "Usage: $0 [-u <dockerhub_username>] [-p]" >&2; }

while getopts "u:p" opt
do
    case ${opt} in
        u) USERNAME=${OPTARG}; echo "Building container under dockerhub_username ${USERNAME}";;
        p) PUSH_FLAG=true; echo "Will push image to dockerhub";;
        *) usage; exit 1 ;;
    esac
done

if [[ ! ${USERNAME} ]]; then
    USERNAME=$(whoami)
    echo "dockerhub_username is not set; building container under username ${USERNAME}"
fi

IMAGE_NAME="${USERNAME}/sharp_features_ec_net:latest"
CONTAINER_NAME="sharp_features_ec_net_container"
DOCKERFILE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/Dockerfile

echo "******* BUILDING THE IMAGE FROM DOCKERFILE *******"
nvidia-docker build \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_NAME}" \
    . 

echo "******* BUILDING CUSTOM TF OPS *******"
TF_OPS_DIR="/home/EC-Net/code/tf_ops"
docker run \
    --runtime=nvidia \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}" \
    /bin/sh \
        -c "cd ${TF_OPS_DIR}/grouping && ./tf_grouping_compile.sh && \\
            cd ${TF_OPS_DIR}/interpolation && ./tf_interpolate_compile.sh && \\
            cd ${TF_OPS_DIR}/sampling && ./tf_sampling_compile.sh"

echo "******* COMMITTING THE CONTAINER  *******"
docker commit "${CONTAINER_NAME}" "${IMAGE_NAME}"

if [[ ${PUSH_FLAG} ]]
then
    echo "******* LOGGING TO DOCKER HUB *******"
    docker login

    echo "******* PUSHING IMAGE TO DOCKER HUB *******"
    docker push "${IMAGE_NAME}"
fi

docker container rm "${CONTAINER_NAME}"
