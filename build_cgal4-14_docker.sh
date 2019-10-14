#!/bin/bash

set -e
set -x

# example launch string:
# ./build_docker.sh [-p]
#     -p:       push the build image to the dockerhub under 'gbobrovskih' username

IMAGE_NAME="gbobrovskih/cgal_4-14"
IMAGE_NAME_TAG="${IMAGE_NAME}:latest"
DOCKERFILE="$(dirname `realpath $0`)/Dockerfile"     # full pathname of Dockerfile

echo "******* BUILDING IMAGE ${IMAGE_NAME} *******"

docker build \
    --file ${DOCKERFILE} \
    --tag ${IMAGE_NAME_TAG} \
    .


if echo $* | grep -e "-p" -q
then
    echo "******* LOGGING TO DOCKER HUB *******"
    docker login

    echo "******* PUSHING IMAGE TO DOCKER HUB *******"
    docker push ${IMAGE_NAME_TAG}
fi
