#!/bin/bash

set -e

# example launch string:#
# ./build_ecnet_docker.sh -u <dockerhub username> [-p]
# 	 -u:  dockerhub username to create an image under
#	 -p:  push the build image to the dockerhub under given username

if [[ $# -lt 2 ]]
then
    echo "build_ecnet_docker.sh -u <dockerhub_username> -p"
    exit 1
fi

while getopts "u:" opt
do
    case ${opt} in
        u) USERNAME=$OPTARG;;
    esac
done

if [[ ! ${USERNAME} ]]; then
    echo "dockerhub username is not set";
    exit 1
fi

IMAGE_NAME="${USERNAME}/sharp_features_ec_net:latest"
CONTAINER_NAME="sharp_features_ec_net_container"
DOCKERFILE="$(dirname `realpath $0`)/Dockerfile"

echo "******* BUILDING THE IMAGE FROM DOCKERFILE *******"
nvidia-docker build \
    --file ${DOCKERFILE} \
    --tag ${IMAGE_NAME} \
    . 

echo "******* BUILDING CUSTOM TF OPS *******"
docker run --runtime=nvidia --name "${CONTAINER_NAME}" ${IMAGE_NAME} /bin/bash -c "cd /home/EC-Net/code/tf_ops/grouping && ./tf_grouping_compile.sh && \
    						  cd ../interpolation && ./tf_interpolate_compile.sh && \ 
                                                  cd ../sampling && ./tf_sampling_compile.sh" 

echo "******* COMMITTING THE CONTAINER  *******"
docker commit ${CONTAINER_NAME} ${IMAGE_NAME}

if echo $* | grep -e "-p" -q
then
    echo "******* LOGGING TO DOCKER HUB *******"
    docker login

    echo "******* PUSHING IMAGE TO DOCKER HUB *******"
    docker push ${IMAGE_NAME}
fi

docker container rm "${CONTAINER_NAME}"
