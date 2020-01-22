#!/bin/bash

set -e

# example launch string:
# ./build_docker.sh [-p] [-v]
#     -p:       push the build image to the dockerhub under 'artonson' username
#     -v:       be verbose

usage() { echo "Usage: $0 [-p] [-v]" >&2; }

VERBOSE=false
PUSH=false
while getopts "pv" opt
do
    case ${opt} in
        p) PUSH=true;;
        v) VERBOSE=true;;
        *) usage; exit 1 ;;
    esac
done

if [ "${VERBOSE}" = true ]; then
    set -x
fi

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd )"
source "${PROJECT_ROOT}"/env.sh

DOCKERFILE="${PROJECT_ROOT}/docker/Dockerfile"     # full pathname of Dockerfile

echo "******* BUILDING IMAGE ${IMAGE_NAME} *******"

docker build \
    --file "${DOCKERFILE}" \
    --tag "${IMAGE_NAME_TAG}" \
    .


if [ "${PUSH}" = true ]; then
    echo "******* LOGGING TO DOCKER HUB *******"
    docker login

    echo "******* PUSHING IMAGE TO DOCKER HUB *******"
    docker push "${IMAGE_NAME_TAG}"
fi
