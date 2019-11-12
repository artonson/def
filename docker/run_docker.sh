#!/bin/bash

set -e
#set -x

# example launch string:
# ./run_docker.sh -d <server_data_dir> -l <server_logs_dir> -g gpu
#   server_data_dir:        the data directory where the training sample resides
#   server_logs_dir:        the directory where the output logs are supposed to be written
#   gpu:                    comma-separated list of gpus

if [[ $# -lt 2 ]]
then
    echo "run_docker.sh -d <server_data_dir> -l <server_logs_dir> -g <gpu-indexes>"
    exit 1
fi

while getopts "d:l:g:" opt
do
    case ${opt} in
        d) HOST_DATA_DIR=$OPTARG;;
        l) HOST_LOG_DIR=$OPTARG;;
        g) GPU_ENV=$OPTARG;;
        *) echo "No reasonable options found!";;
    esac
done

if [[ ! -d ${HOST_DATA_DIR} ]]; then
    echo "server_data_dir is not set or not a directory";
    exit 1
fi
if [[ ! -d ${HOST_LOG_DIR} ]]; then
    echo "server_logs_dir is not set or not a directory";
    exit 1
fi


# HOST_<anything> refers to paths OUTSIDE container, i.e. on host machine
# CONT_<anything> refers to paths INSIDE container
SHARED_MEM="25g"        # amount of shared memory to reserve for the prefetchers

CONTAINER="artonson/sharp_features:pointweb-ops"
docker inspect --type=image ${CONTAINER} >/dev/null || docker pull ${CONTAINER}

HOST_CODE_DIR=$(realpath $(dirname `realpath $0`)/..)     # dirname of THIS file
CONT_CODE_DIR="/code"
CONT_DATA_DIR="/data"
CONT_LOG_DIR="/logs"

if [[ -z "${GPU_ENV}" ]] ; then
    # set all GPUs as visible in the docker
    num_gpus=`nvidia-smi -L | wc -l`
    GPU_ENV=`seq -s, 0 $((num_gpus-1))`
fi

echo "******* LAUNCHING CONTAINER ${CONTAINER} *******"
echo "      Pushing you to ${CONT_CODE_DIR} directory"
echo "      Data is at ${CONT_DATA_DIR}"
echo "      Writable logs are at ${CONT_LOG_DIR}"
echo "      Environment: PYTHONPATH=${CONT_CODE_DIR}"
echo "      Environment: CUDA_VISIBLE_DEVICES=${GPU_ENV}"

NAME="3ddl.`whoami`.`uuidgen`.`echo ${GPU_ENV} | tr , .`.sharp_features"
docker run \
    --name ${NAME} \
    --interactive=true \
    --runtime=nvidia \
    --rm \
    --tty=true \
    --env CUDA_VISIBLE_DEVICES=${GPU_ENV} \
    --env PYTHONPATH=${CONT_CODE_DIR} \
    --shm-size=${SHARED_MEM} \
    -v ${HOST_CODE_DIR}:${CONT_CODE_DIR} \
    -v ${HOST_DATA_DIR}:${CONT_DATA_DIR} \
    -v ${HOST_LOG_DIR}:${CONT_LOG_DIR} \
    --workdir ${CONT_CODE_DIR} \
    ${CONTAINER}
