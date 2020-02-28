#!/bin/bash

#SBATCH --job-name=sharpf-train-points
#SBATCH --output=logs/sharpf-train-points_%A_%a.out
#SBATCH --error=logs/sharpf-train-points_%A_%a.err
#SBATCH --array=1-1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_debug
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=40000

module load apps/singularity-3.2.0

__usage="
Usage: $0 -d data_dir -o output_dir -l logs_dir -f model_config [-v]

  -d: 	input data directory
  -o: 	output directory where model weights get written
  -l:   logs directory
  -f:   model config file (from sharpf/models/specs dir)
  -v:   if set, verbose mode is activated (more output from the script generally)

Example:
sbatch make_patches.sbatch.sh
  -d /gpfs/gpfs0/3ddl/datasets/abc \\
  -o /gpfs/gpfs0/3ddl/datasets/abc/eccv  \\
  -l /home/artonson/tmp/logs  \\
  -v
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
VERBOSE=false
while getopts "c:o:d:l:f:v" opt
do
    case ${opt} in
        c) CHUNK=$OPTARG;;
        o) OUTPUT_PATH_HOST=$OPTARG;;
        d) DATA_PATH_HOST=$OPTARG;;
        l) LOGS_PATH_HOST=$OPTARG;;
        f) MODEL_CONFIG=$OPTARG;;
        v) VERBOSE=true;;
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

DATA_PATH_CONTAINER="/data"
if [[ ! ${DATA_PATH_HOST} ]]; then
    echo "data_dir is not set" && usage && exit 1
fi

OUTPUT_PATH_CONTAINER="/out"
if [[ ! ${OUTPUT_PATH_HOST} ]]; then
    echo "output_dir is not set" && usage && exit 1
fi

LOGS_PATH_CONTAINER="/logs"
if [[ ! ${LOGS_PATH_HOST} ]]; then
    echo "logs_dir is not set" && usage && exit 1
fi

if [[ ! ${MODEL_CONFIG} ]]; then
    echo "config_file is not set" && usage && exit 1
fi

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

N_TASKS=1
OMP_NUM_THREADS=4
TRAIN_SCRIPT="${CODE_PATH_CONTAINER}/scripts/train_scripts/train_sharp.py"
MODEL_CONFIGS_PATH_CONTAINER="${CODE_PATH_CONTAINER}/sharpf/models/specs"
MODEL_SPEC_PATH="${MODEL_CONFIGS_PATH_CONTAINER}/${MODEL_CONFIG}"

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind ${DATA_PATH_HOST}:${DATA_PATH_CONTAINER} \
  --bind ${LOGS_PATH_HOST}:${LOGS_PATH_CONTAINER} \
  --bind ${OUTPUT_PATH_HOST}:${OUTPUT_PATH_CONTAINER} \
  --bind "${PWD}":/run/user \
  "${SIMAGE_FILENAME}" \
      bash -c 'export OMP_NUM_THREADS='"${OMP_NUM_THREADS}; \\
      python3 ${TRAIN_SCRIPT} \\
        --gpu ${GPU_ID} \\
        --model-spec ${MODEL_SPEC_PATH} \\
        --epochs ${NUM_EPOCHS} \\
        --model-spec ${MODEL_SPEC} \\
        --infer-from-spec \\
        --overwrite \\
        --log-dir-prefix ${LOGS_PATH_CONTAINER} \\
        --loss-funct ${LOSS_FUNCTION} \\
        --train-batch-size ${TRAIN_BATCH_SIZE} \\
        --val-batch-size ${VAL_BATCH_SIZE} \\
        --save-model-file ${SAVE_MODEL_FILEPREFIX} \\
        --data-root ${DATA_PATH_CONTAINER} \\
        --data-label points \\
        --target-label distances \\
         ${VERBOSE_ARG} \\
           1> >(tee ${LOGS_PATH_CONTAINER}/${SLURM_ARRAY_TASK_ID}.out) \\
           2> >(tee ${LOGS_PATH_CONTAINER}/${SLURM_ARRAY_TASK_ID}.err)"

