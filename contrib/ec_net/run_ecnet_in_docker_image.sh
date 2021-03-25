#!/usr/bin/env bash

#SBATCH --job-name=ecnet
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem 50G


# example launch string:
# ./run_sharpness_fields_in_docker.sh -i <input_dir> -l <label> -s <meshlab_script> -d <docker image name> -c <docker container name> -g <gpu-indexes>
#	-i: 	input directory with .hdf5 files
#	-d: 	docker image name
#	-c: 	docker container name
#	-g: 	comma-separated gpu indexes

usage() { echo "Usage: $0 -i <input_file> -l <label> -s <meshlab_script> -d <docker_image_name> -c <container_name> -g <gpu_indexes>" >&2; }

while getopts "i:o:l:s:d:c:g:v:" opt
do
    case ${opt} in
        i) INPUT_FILE=$OPTARG;;
        o) OUTPUT_FOLDER=$OPTARG;;
        l) DATA_LABEL=$OPTARG;;
        s) MESHLAB_SCRIPT=$OPTARG;;
        d) IMAGE_NAME=$OPTARG;;
        c) CONTAINER_NAME=$OPTARG;;
        g) GPU_ENV=$OPTARG;;
        v) VARLEN=${OPTARG};;
        *) usage; exit 1 ;;
    esac
done

if [[ ! ${INPUT_FILE} ]]; then
    echo "input_file is not set";
    usage
    exit 1
fi

if [[ ! ${OUTPUT_FOLDER} ]]; then
    echo "output_folder is not set";
    usage
    exit 1
fi

if [[ ! ${DATA_LABEL} ]]; then
    echo "data_label is not set";
    usage
    exit 1
fi

if [[ -z "${GPU_ENV}" ]] ; then
    echo "gpu_indexes not set; selecting GPU 0";
    GPU_ENV=0
fi

# OFFICIAL_IMAGE_NAME=/gpfs/gpfs0/a.matveev/mariataktasheva_sharp_features_sharpness_fields-2020-05-09-d61d7dfe0e3f.sif
OFFICIAL_IMAGE_NAME=/gpfs/gpfs0/a.matveev/am_ecnet_fix-2020-11-08-df7dd1f547d4.sif  
if [[ ! ${IMAGE_NAME} ]]; then
    echo "docker_image_name is not set; selecting the official docker image ${OFFICIAL_IMAGE_NAME}";
    IMAGE_NAME=${OFFICIAL_IMAGE_NAME}
#    docker pull ${IMAGE_NAME}
fi

DEFAULT_CONTAINER_NAME="3ddl.$( whoami ).$( uuidgen ).$( echo "${GPU_ENV}" | tr , . ).sharp_features_sharpness_fields"
if [[ ! ${CONTAINER_NAME} ]]; then
    echo "container_name is not set; generated container name: ${DEFAULT_CONTAINER_NAME}";
    CONTAINER_NAME=${DEFAULT_CONTAINER_NAME}
fi

DATA_PATH_HOST="$( cd "$( dirname "${INPUT_FILE}" )" >/dev/null 2>&1 && pwd )"
DATA_PATH_CONTAINER="/home/data"
OUT_PATH_HOST="${OUTPUT_FOLDER}"
OUT_PATH_CONTAINER="/home/out"

LOCAL_DIR=/trinity/home/a.matveev/sharp_features/contrib/ec_net
LOGS_PATH_HOST="${LOCAL_DIR}/logs"
LOGS_PATH_CONTAINER="/home/logs"

SPLITCODE_PATH_HOST="${LOCAL_DIR}/../hdf5_utils"
SPLITCODE_PATH_CONTAINER="/home/hdf5_utils"

INPUT_FILE_CONTAINER="${DATA_PATH_CONTAINER}/$(basename "${INPUT_FILE}")"

SPLIT_DATA_PATH_CONTAINER="${OUT_PATH_CONTAINER}/xyz_splitted_ec"
SPLIT_INPUT_CONTAINER="${SPLIT_DATA_PATH_CONTAINER}/*.xyz"
SPLIT_OUTPUT_CONTAINER="${OUT_PATH_CONTAINER}/ec_net_results"
CODE_PATH_CONTAINER="/home/EC-Net/code"
MODEL_PATH_CONTAINER="/home/EC-Net/model/pretrain"

echo "******* LAUNCHING IMAGE ${IMAGE_NAME} IN CONTAINER ${CONTAINER_NAME} *******"
echo "  "
echo "  HOST OPTIONS:"
echo "  input path:           ${INPUT_FILE}"
echo "  output path:          ${OUT_PATH_HOST}/ec_net_results"
echo "  split code path:      ${SPLITCODE_PATH_HOST}"
echo "  logs path:            ${LOGS_PATH_HOST}"
echo "  "
echo "  CONTAINER OPTIONS:"
echo "  code path:            ${CODE_PATH_CONTAINER}"
echo "  split code path:      ${SPLITCODE_PATH_CONTAINER}"
echo "  model path:           ${MODEL_PATH_CONTAINER}"
echo "  input path:           ${INPUT_FILE_CONTAINER}"
echo "  split input path:     ${SPLIT_INPUT_CONTAINER}"
echo "  split output path:    ${SPLIT_OUTPUT_CONTAINER}"
echo "  logs path:            ${LOGS_PATH_CONTAINER}"

singularity exec \
    --nv \
    --bind "${DATA_PATH_HOST}":"${DATA_PATH_CONTAINER}" \
    --bind "${OUT_PATH_HOST}":"${OUT_PATH_CONTAINER}" \
    --bind "${LOGS_PATH_HOST}":"${LOGS_PATH_CONTAINER}" \
    --bind "${SPLITCODE_PATH_HOST}":"${SPLITCODE_PATH_CONTAINER}" \
    "${IMAGE_NAME}" \
    /bin/bash -c "cd ${SPLITCODE_PATH_CONTAINER} && \\
        echo 'Splitting input files...' && \\
        python split_hdf5_images_2.py \\
          ${INPUT_FILE_CONTAINER} \\
          --output_dir ${SPLIT_DATA_PATH_CONTAINER} \\
          --output_format 'xyz' \\
          --label ${DATA_LABEL} && \\
        cd ${CODE_PATH_CONTAINER} && \\
        echo 'Evaluating the model...' && \\
        python main.py \\
          --phase test \\
          --batch_size 1 \\
          --log_dir ${MODEL_PATH_CONTAINER} \\
          --eval_input '${SPLIT_INPUT_CONTAINER}' \\
          --eval_output ${SPLIT_OUTPUT_CONTAINER} \\
          1>${LOGS_PATH_CONTAINER}/out.out \\
          2>${LOGS_PATH_CONTAINER}/err.err && \\
       rm -rf ${SPLIT_DATA_PATH_CONTAINER} && \\
       echo 'Output is in ${DATA_PATH_HOST}/ec_net_results'"
