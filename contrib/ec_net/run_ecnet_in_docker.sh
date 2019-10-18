#!/usr/bin/env bash

set -e

# example launch string:
# ./run_ecnet_in_docker.sh -u <dockerhub_username> -i <input_dir> -o <output_dir> -d <docker image name> -c <docker container name> -g <gpu-indexes>
# 	-u:	username, under which a docker image was created
#	-i: 	input directory with .xyz files
#	-o: 	output directory
#	-d: 	docker image name
#	-c: 	docker container name
#	-g: 	comma-separated gpu indexes

if [[ $# -lt 2 ]]
then
    echo "run_ecnet_in_docker.sh -u <dockerhub_username> -i <input_dir> -o <output_dir> -d <docker image name> -c <docker container name> -g <gpu-indexes>"
    exit 1
fi

while getopts "u:i:o:d:c:g:" opt
do
    case ${opt} in
        u) USERNAME=$OPTARG;;
        i) INPUT=$OPTARG;;
        o) OUTPUT=$OPTARG;;
        d) IMAGE_NAME=$OPTARG;;
        c) CONTAINER_NAME=$OPTARG;;
        g) GPU_ENV=$OPTARG;;
        *) echo "No reasonable options found!";;
    esac
done

if [[ ! ${USERNAME} ]]; then
    echo "dockerhub username is not set";
    exit 1
fi

if [[ ! ${INPUT} ]]; then
    echo "input_file is not set";
    exit 1
fi

if [[ ! ${OUTPUT} ]]; then
    echo "output_file is not set";
    exit 1
fi

if [[ -z "${GPU_ENV}" ]] ; then
    # set all GPUs as visible in the docker
    num_gpus=`nvidia-smi -L | wc -l`
    GPU_ENV=`seq -s, 0 $((num_gpus-1))`
fi

DATA_PATH_HOST="$(dirname `realpath ${INPUT}`)"
DATA_PATH_CONTAINER="/home/data"
LOGS_PATH_HOST="$(dirname `realpath $0`)/logs"
LOGS_PATH_CONTAINER="/home/logs"

SPLITTED_INPUT="${DATA_PATH_CONTAINER}/xyz_splitted/*.xyz"
INPUT_FILE="${DATA_PATH_CONTAINER}/$(basename ${INPUT})"
OUTPUT_FILE="${DATA_PATH_CONTAINER}/$(basename ${OUTPUT})"


echo "******* LAUNCHING CONTAINER ${IMAGE_NAME} UNDER NAME ${CONTAINER_NAME} *******"
echo " "
echo "	CONTAINER OPTIONS:"
echo "	code directory: 	/home/EC-Net/code"
echo "	model directory: 	/home/EC-Net/model/pretrain"
echo "	input path: 		${INPUT_FILE}"
echo "	splitted input path 	${DATA_PATH_CONTAINER}/xyz_splitted"
echo "	output path: 		${OUTPUT_FILE}"
echo "	logs path:		${LOGS_PATH_CONTAINER}"
echo " "
echo "	HOST OPTIONS:"
echo "	input path:		${INPUT}"
echo "	output path:		${OUTPUT}"
echo "	logs path:		${LOGS_PATH_HOST}"

nvidia-docker run \
    --rm \
    --name ${CONTAINER_NAME} \
    --env CUDA_VISIBLE_DEVICES=${GPU_ENV} \
    -v ${DATA_PATH_HOST}:${DATA_PATH_CONTAINER} \
    -v ${LOGS_PATH_HOST}:${LOGS_PATH_CONTAINER} \
    -v "$(dirname `realpath $0`)/src:/home/split" \
     ${IMAGE_NAME} \
     /bin/bash -c "cd /home/split &&
        python split_hdf5.py ${INPUT_FILE} --output_dir '${DATA_PATH_CONTAINER}/xyz_splitted' --output_format 'xyz' --label 'data' &&
	cd /home/EC-Net/code && 
	python main.py --phase test --log_dir ../model/pretrain --eval_input '${SPLITTED_INPUT}' --eval_output ${OUTPUT_FILE} 1>${LOGS_PATH_CONTAINER}/out.out 2>${LOGS_PATH_CONTAINER}/err.err"
