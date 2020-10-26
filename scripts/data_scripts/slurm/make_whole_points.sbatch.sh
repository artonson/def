#!/bin/bash

#SBATCH --job-name=sharpf-whole-points
#SBATCH --output=/trinity/home/a.artemov/tmp/sharpf_whole_points/%A_%a.out
#SBATCH --error=/trinity/home/a.artemov/tmp/sharpf_whole_points/%A_%a.err
#SBATCH --array=1-80
#SBATCH --time=02:00:00
#SBATCH --partition=htc
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8g
#SBATCH --oversubscribe

# module load apps/singularity-3.2.0

__usage="
Usage: $0 -c chunk -o output_dir -d data_dir -l logs_dir -f config_file -i input_filelist -t item_offset [-v]

  -c:   zero-based chunk identifier
  -o: 	output directory where patches will be written
  -d: 	input data directory
  -l:   server logs dir
  -f:   dataset config file (from scripts/data_scripts/configs/pointcloud_datasets dir
  -v:   if set, verbose mode is activated (more output from the script generally)
  -i:   input filename with item ids from a given chunk to process

Example:
sbatch make_points.sbatch.sh
  -d /gpfs/gpfs0/3ddl/datasets/abc \\
  -o /gpfs/gpfs0/3ddl/datasets/abc/eccv  \\
  -l /home/artonson/tmp/logs  \\
  -c 22
  -v
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
VERBOSE=false
while getopts "c:o:d:l:f:vi:t:" opt
do
    case ${opt} in
        c) CHUNK=$OPTARG;;
        o) OUTPUT_PATH_HOST=$OPTARG;;
        d) DATA_PATH_HOST=$OPTARG;;
        l) LOGS_PATH_HOST=$OPTARG;;
        f) DATASET_CONFIG=$OPTARG;;
        v) VERBOSE=true;;
        i) INPUT_FILELIST=$OPTARG;;
        t) ITEM_OFFSET=$OPTARG;;
        *) usage; exit 1 ;;
    esac
done

if [[ "${VERBOSE}" = true ]]; then
    set -x
    VERBOSE_ARG="--verbose"
fi

# get image filenames from here
PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features
source "${PROJECT_ROOT}"/env.sh

if [[ ! ${DATASET_CONFIG} ]]; then
    echo "config_file is not set" && usage && exit 1
fi

if [[ ! ${CHUNK} ]]; then
    echo "chunk is not set" && usage && exit 1
fi

OUTPUT_PATH_CONTAINER="/out"
if [[ ! ${OUTPUT_PATH_HOST} ]]; then
    echo "output_dir is not set" && usage && exit 1
fi

DATA_PATH_CONTAINER="/data"
if [[ ! ${DATA_PATH_HOST} ]]; then
    echo "data_dir is not set" && usage && exit 1
fi

LOGS_PATH_CONTAINER="/logs"
if [[ ! ${LOGS_PATH_HOST} ]]; then
    echo "logs_dir is not set" && usage && exit 1
fi

if [[ ! ${INPUT_FILELIST} ]]; then
    echo "input_filelist is not set" && usage && exit 1
fi
TASK_ID=$(( SLURM_ARRAY_TASK_ID + ITEM_OFFSET ))

if [[ $( wc -l <"${INPUT_FILELIST}" ) -lt ${TASK_ID} ]]; then
    echo "SLURM task ID exceeds number of files to process, exiting" && exit 1
fi
INPUT_ITEM_ID=$( head -n "${TASK_ID}" <"${INPUT_FILELIST}" | tail -1 )

CODE_PATH_CONTAINER="/code"
CODE_PATH_HOST=${PROJECT_ROOT}

echo "******* LAUNCHING IMAGE ${SIMAGE_FILENAME} *******"
echo "  "
echo "  HOST OPTIONS:"
echo "  data path:            ${DATA_PATH_HOST}"
echo "  code path:            ${CODE_PATH_HOST}"
echo "  logs path:            ${LOGS_PATH_HOST}"
echo "  output path:          ${OUTPUT_PATH_HOST}"
echo "  "
echo "  CONTAINER OPTIONS:"
echo "  data path:            ${DATA_PATH_CONTAINER}"
echo "  code path:            ${CODE_PATH_CONTAINER}"
echo "  logs path:            ${LOGS_PATH_CONTAINER}"
echo "  output path:          ${OUTPUT_PATH_CONTAINER}"
echo "  "

OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
MAKE_DATA_SCRIPT="${CODE_PATH_CONTAINER}/scripts/data_scripts/generate_fused_pointcloud_data.py"
PC_CONFIGS_PATH_CONTAINER="${CODE_PATH_CONTAINER}/scripts/data_scripts/configs/pointcloud_datasets"
DATASET_PATH="${PC_CONFIGS_PATH_CONTAINER}/${DATASET_CONFIG}"

SLICE_START=$(( SLICE_SIZE * SLURM_ARRAY_TASK_ID ))
SLICE_END=$(( SLICE_SIZE * (SLURM_ARRAY_TASK_ID + 1) ))
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} SLICE_START=${SLICE_START} SLICE_END=${SLICE_END}"

singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind "${DATA_PATH_HOST}":${DATA_PATH_CONTAINER} \
  --bind "${LOGS_PATH_HOST}":${LOGS_PATH_CONTAINER} \
  --bind "${OUTPUT_PATH_HOST}":${OUTPUT_PATH_CONTAINER} \
  --bind "${PWD}":/run/user \
  "${SIMAGE_FILENAME}" \
      bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
      python3 ${MAKE_DATA_SCRIPT} \\
        --input-dir ${DATA_PATH_CONTAINER} \\
        --chunk ${CHUNK} \\
        --output-dir ${OUTPUT_PATH_CONTAINER} \\
        --jobs ${OMP_NUM_THREADS} \\
        --item-id ${INPUT_ITEM_ID} \\
        --dataset-config ${DATASET_PATH} \\
         ${VERBOSE_ARG} \\
           1> >(tee ${LOGS_PATH_CONTAINER}/${CHUNK}_${SLURM_ARRAY_TASK_ID}.out) \\
           2> >(tee ${LOGS_PATH_CONTAINER}/${CHUNK}_${SLURM_ARRAY_TASK_ID}.err)"
