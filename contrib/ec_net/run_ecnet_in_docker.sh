#!/usr/bin/env bash

set -e

# example launch string:
# ./run_ecnet_in_docker.sh -i <input_dir> -o <output_dir> -d <docker image name> -c <docker container name> -g <gpu-indexes>
#	-i: 	input directory with .xyz files
#	-o: 	output directory
#	-d: 	docker image name
#	-c: 	docker container name
#	-g: 	comma-separated gpu indexes

usage() { echo "Usage: $0 -i <input_file> -o <output_file> -d <docker_image_name> -c <container_name> -g <gpu_indexes>" >&2; }

while getopts "i:o:d:c:g:" opt
do
    case ${opt} in
        i) INPUT_FILE=$OPTARG;;
        o) OUTPUT_FILE=$OPTARG;;
        d) IMAGE_NAME=$OPTARG;;
        c) CONTAINER_NAME=$OPTARG;;
        g) GPU_ENV=$OPTARG;;
        *) usage; exit 1 ;;
    esac
done

if [[ ! ${INPUT_FILE} ]]; then
    echo "input_file is not set";
    usage
    exit 1
fi

if [[ ! ${OUTPUT_FILE} ]]; then
    echo "output_file is not set";
    usage
    exit 1
fi

if [[ -z "${GPU_ENV}" ]] ; then
    echo "gpu_indexes not set; selecting GPU 0";
    GPU_ENV=0
fi

OFFICIAL_IMAGE_NAME="artonson/sharp_features_ec_net:latest"
if [[ ! ${IMAGE_NAME} ]]; then
    echo "docker_image_name is not set; selecting the official docker image ${OFFICIAL_IMAGE_NAME}";
    IMAGE_NAME=${OFFICIAL_IMAGE_NAME}
    docker pull ${IMAGE_NAME}
fi

DEFAULT_CONTAINER_NAME="3ddl.$( whoami ).$( uuidgen ).$( echo "${GPU_ENV}" | tr , . ).sharp_features_ec_net"
if [[ ! ${CONTAINER_NAME} ]]; then
    echo "container_name is not set; generated container name: ${DEFAULT_CONTAINER_NAME}";
    CONTAINER_NAME=${DEFAULT_CONTAINER_NAME}
fi

DATA_PATH_HOST="$( cd "$( dirname "${INPUT_FILE}" )" >/dev/null 2>&1 && pwd )"
DATA_PATH_CONTAINER="/home/data"
LOCAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LOGS_PATH_HOST="${LOCAL_DIR}"/logs
LOGS_PATH_CONTAINER="/home/logs"
SPLITCODE_PATH_HOST="${LOCAL_DIR}"/src
SPLITCODE_PATH_CONTAINER="/home/split"

INPUT_FILE_CONTAINER="${DATA_PATH_CONTAINER}/$(basename "${INPUT_FILE}")"
OUTPUT_FILE_CONTAINER="${DATA_PATH_CONTAINER}/$(basename "${OUTPUT_FILE}")"

SPLIT_DATA_PATH_CONTAINER="${DATA_PATH_CONTAINER}/xyz_splitted"
SPLIT_INPUT_CONTAINER="${SPLIT_DATA_PATH_CONTAINER}"'/*.xyz'
CODE_PATH_CONTAINER="/home/EC-Net/code"
MODEL_PATH_CONTAINER="/home/EC-Net/model/pretrain"

echo "******* LAUNCHING IMAGE ${IMAGE_NAME} IN CONTAINER ${CONTAINER_NAME} *******"
echo "  "
echo "  HOST OPTIONS:"
echo "  input path:           ${INPUT_FILE}"
echo "  output path:          ${OUTPUT_FILE}"
echo "  logs path:            ${LOGS_PATH_HOST}"
echo "  wrapper code path:    ${SPLITCODE_PATH_CONTAINER}"
echo "  "
echo "  CONTAINER OPTIONS:"
echo "  code path:            ${CODE_PATH_CONTAINER}"
echo "  wrapper code path:    ${SPLITCODE_PATH_CONTAINER}"
echo "  model path:           ${MODEL_PATH_CONTAINER}"
echo "  input path:           ${INPUT_FILE_CONTAINER}"
echo "  split input path:     ${SPLIT_INPUT_CONTAINER}"
echo "  output path:          ${OUTPUT_FILE_CONTAINER}"
echo "  logs path:            ${LOGS_PATH_CONTAINER}"

nvidia-docker run \
    --rm \
    --name "${CONTAINER_NAME}" \
    --env CUDA_VISIBLE_DEVICES="${GPU_ENV}" \
    -v "${DATA_PATH_HOST}":"${DATA_PATH_CONTAINER}" \
    -v "${LOGS_PATH_HOST}":"${LOGS_PATH_CONTAINER}" \
    -v "${SPLITCODE_PATH_HOST}":"${SPLITCODE_PATH_CONTAINER}" \
    "${IMAGE_NAME}" \
    /bin/bash \
        -c "cd ${SPLITCODE_PATH_CONTAINER} && \\
        python split_hdf5.py \\
          ${INPUT_FILE_CONTAINER} \\
          --output_dir ${SPLIT_DATA_PATH_CONTAINER} \\
          --output_format 'xyz' \\
          --label 'data' && \\
        cd ${CODE_PATH_CONTAINER} && \\
        python main.py \\
          --phase test \\
          --log_dir ${MODEL_PATH_CONTAINER} \\
          --eval_input ${SPLIT_INPUT_CONTAINER} \\
          --eval_output ${OUTPUT_FILE_CONTAINER} \\
          1>${LOGS_PATH_CONTAINER}/out.out \\
          2>${LOGS_PATH_CONTAINER}/err.err"
