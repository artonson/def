#!/bin/bash

#SBATCH --job-name=sharpf-data
#SBATCH --output=make_canonical_datasets_%A.out
#SBATCH --error=make_canonical_datasets_%A.err
#SBATCH --time=2:00:00
#SBATCH --partition=cpu_big
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=80000

module load apps/singularity-3.2.0

__usage="
Usage: $0 -i input_dir -o output_dir [-v]

  -i: 	input directory with source HDF5 with data
  -o: 	output directory with resulting
  -v:   if set, verbose mode is activated (more output from the script generally)

Example:
sbatch make_patches.sbatch.sh
  -i /gpfs/gpfs0/3ddl/datasets/abc/raw \\
  -o /gpfs/gpfs0/3ddl/datasets/abc/canonical  \\
  -v
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
VERBOSE=false
while getopts "i:o:v" opt
do
    case ${opt} in
        i) INPUT_PATH_HOST=$OPTARG;;
        o) OUTPUT_PATH_HOST=$OPTARG;;
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

INPUT_PATH_CONTAINER="/in"
if [[ ! ${INPUT_PATH_HOST} ]]; then
    echo "input_dir is not set" && usage && exit 1
fi

OUTPUT_PATH_CONTAINER="/out"
if [[ ! ${OUTPUT_PATH_HOST} ]]; then
    echo "output_dir is not set" && usage && exit 1
fi

CODE_PATH_CONTAINER="/code"
CODE_PATH_HOST=${PROJECT_ROOT}

echo "******* LAUNCHING IMAGE ${SIMAGE_FILENAME} *******"
echo "  "
echo "  HOST OPTIONS:"
echo "  code path:            ${CODE_PATH_HOST}"
echo "  input path:           ${INPUT_PATH_HOST}"
echo "  output path:          ${OUTPUT_PATH_HOST}"
echo "  "
echo "  CONTAINER OPTIONS:"
echo "  code path:            ${CODE_PATH_CONTAINER}"
echo "  input path:           ${INPUT_PATH_CONTAINER}"
echo "  output path:          ${OUTPUT_PATH_CONTAINER}"
echo "  "

N_TASKS=16
MAKE_DATA_SCRIPT="${CODE_PATH_CONTAINER}/sharpf/utils/scripts/defrag_shuffle_split_hdf5.py"
CHUNK_SIZE=16000
TRAIN_FRACTION=0.8
RANDOM_SEED=9675

singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind "${INPUT_PATH_HOST}":${INPUT_PATH_CONTAINER} \
  --bind "${OUTPUT_PATH_HOST}":${OUTPUT_PATH_CONTAINER} \
  --bind "${PWD}":/run/user \
  "${SIMAGE_FILENAME}" \
      "python3 ${MAKE_DATA_SCRIPT} \\
        --input-dir ${INPUT_PATH_CONTAINER} \\
        --output-dir ${OUTPUT_PATH_CONTAINER} \\
        --num-items-per-file ${CHUNK_SIZE} \\
        --train-fraction ${TRAIN_FRACTION} \\
        --random-shuffle \\
        --random-seed ${RANDOM_SEED} \\
        --jobs ${N_TASKS} \\
         ${VERBOSE_ARG}"
