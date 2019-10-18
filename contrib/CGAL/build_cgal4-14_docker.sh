#!/bin/bash

set -e

# example launch string:#
# ./build_cgal4-14_docker.sh -u <dockerhub username> [-p]
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

IMAGE_NAME="${USERNAME}/cgal_4-14:latest"
DOCKERFILE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/Dockerfile

echo "******* BUILDING THE IMAGE FROM DOCKERFILE *******"
docker build \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_NAME}" \
    .

if [[ ${PUSH_FLAG} ]]
then
    echo "******* LOGGING TO DOCKER HUB *******"
    docker login

    echo "******* PUSHING IMAGE TO DOCKER HUB *******"
    docker push "${IMAGE_NAME}"
fi
