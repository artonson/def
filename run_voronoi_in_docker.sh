#!/bin/bash

set -e

# example launch string:
# ./run_voronoi_in_docker.sh -R <offset_radius> -r <convolution_radius> -t <threshold> -d <server_data_dir> -l <server_logs_dir> -g gpu
#   R, r, threshold:        Voronoi method parameters
#   server_data_dir:        the data directory where the training sample resides
#   server_logs_dir:        the directory where the output logs are supposed to be written
#   gpu:                    comma-separated list of gpus

if [[ $# -lt 2 ]]
then
    echo "run_docker.sh -R <offset_radius> -r <convolution_radius> -t <threshold> -d <server_data_dir> -l <server_logs_dir> -g <gpu-indexes>"
    exit 1
fi

while getopts "d:l:g:" opt
do
    case ${opt} in
        R) RR=$OPTARG;;
        r) Rr=$OPTARG;;
        t) THRESH=$OPTARG;;
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

CONTAINER="gbobrovskih/cgal_4-14:latest"
docker inspect --type=image ${CONTAINER} >/dev/null || docker pull ${CONTAINER}

HOST_CODE_DIR=$(realpath $(dirname `realpath $0`))     # dirname of THIS file
CONT_CODE_DIR="/home/usr/code/"
CONT_DATA_DIR="/home/usr/data/"
CONT_LOG_DIR="/home/usr/logs/"

if [[ -z "${RR}" ]] ; then
    # set offset_radius to default
    RR=0.2
fi
if [[ -z "${Rr}" ]] ; then
    # set convolution_radius to default
    Rr=0.1
fi
if [[ -z "${THRESH}" ]] ; then
    # set convolution_radius to default
    THRESH=0.16
fi
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
echo ""
NAME="3ddl.`whoami`.`uuidgen`.`echo ${GPU_ENV} | tr , .`.voronoi_R${RR}_r${Rr}_thresh${THRESH}.sharp_features"
docker run \
    --name ${NAME} \
    --interactive=true \
    --runtime=nvidia \
    --rm \
    --tty=true \
    --env CUDA_VISIBLE_DEVICES=${GPU_ENV} \
    --env PYTHONPATH=${CONT_CODE_DIR} \
    --shm-size=${SHARED_MEM} \
    --mount type=bind,source=${HOST_CODE_DIR},target=${CONT_CODE_DIR} \
    --mount type=bind,source=${HOST_DATA_DIR},target=${CONT_DATA_DIR} \
    --mount type=bind,source=${HOST_LOG_DIR},target=${CONT_LOG_DIR} \
    --workdir ${CONT_CODE_DIR} \
    ${CONTAINER} /bin/bash -c "echo \"compiling voronoi_1.cpp\"; g++ -o voronoi voronoi_1.cpp -lCGAL -I/CGAL-4.14.1/include -lgmp; python read_data_voronoi.py -d ${CONT_DATA_DIR} -o ${CONT_LOG_DIR} -R ${RR} -r ${Rr} -t ${THRESH};" 