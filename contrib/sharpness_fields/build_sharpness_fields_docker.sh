#!/bin/bash

set -e

# example launch string:#
# ./build_sharpness_fields_docker.sh -u <dockerhub username> [-p]
#        -u:  dockerhub username to create an image under
#        -p:  push the build image to the dockerhub under given username

if [[ $# -lt 2 ]]
then
    echo "./build_sharpness_fields_docker.sh -u <dockerhub_username> -p"
    exit 1
fi

while getopts "u:" opt
do
    case ${opt} in
        u) USERNAME=$OPTARG;;
        *) echo "No reasonable options found!";;
    esac
done

if [[ ! ${USERNAME} ]]; then
    echo "dockerhub username is not set";
    exit 1
fi

IMAGE_NAME="${USERNAME}/sharp_features_sharpness_fields:latest"
CONTAINER_NAME="sharp_features_sharpness_fields_container"
DOCKERFILE="$(dirname `realpath $0`)/Dockerfile"

echo "******* BUILDING THE IMAGE FROM DOCKERFILE *******"
docker build \
    --file ${DOCKERFILE} \
    --tag ${IMAGE_NAME} \
    .

if echo $* | grep -e "-p" -q
then
    echo "******* LOGGING TO DOCKER HUB *******"
    docker login

    echo "******* PUSHING IMAGE TO DOCKER HUB *******"
    docker push ${IMAGE_NAME}
fi

