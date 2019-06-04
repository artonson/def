#!/bin/bash

set -e
set -x

# example launch string:
# ./build_docker.sh [-p]
#     -p:       push the build image to the dockerhub under 'artonson' username

IMAGE_NAME="artonson/sharp_features"
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
