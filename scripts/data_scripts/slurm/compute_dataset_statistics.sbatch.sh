#!/bin/bash

#SBATCH --job-name=sharpf-stats
#SBATCH --output=/trinity/home/a.artemov/tmp/sharpf-stats/%A.out
#SBATCH --error=/trinity/home/a.artemov/tmp/sharpf-stats/%A.err
#SBATCH --time=01:00:00
#SBATCH --partition=htc
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2g
#SBATCH --oversubscribe

__usage="
Usage: $0 -a abc_input_dir -i input_file -o output_file [-v]

  -i: 	input directory with source HDF5 with data
  -a: 	input ABC directory
  -o: 	output file with resulting output
  -v:   if set, verbose mode is activated (more output from the script generally)
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
VERBOSE=false
while getopts "a:c:i:o:vf:" opt
do
    case ${opt} in
        c) CHUNK=$OPTARG;;
        a) ABC_PATH_HOST=$OPTARG;;
        i) INPUT_FILE_HOST=$OPTARG;;
        o) OUTPUT_FILE_HOST=$OPTARG;;
        f) DATASET_CONFIG=$OPTARG;;
        v) VERBOSE=true;;
        *) usage; exit 1 ;;
    esac
done

if [[ "${VERBOSE}" = true ]]; then
    set -x
    VERBOSE_ARG="--verbose"
fi

# get image filenames from here
PROJECT_ROOT=/trinity/home/a.artemov/repos/sharp_features2
source "${PROJECT_ROOT}"/env.sh

if [[ ! ${DATASET_CONFIG} ]]; then
    echo "config_file is not set" && usage && exit 1
fi

ABC_PATH_CONTAINER="/abc"
if [[ ! ${ABC_PATH_HOST} ]]; then
    echo "input_dir is not set" && usage && exit 1
fi

OUTPUT_PATH_CONTAINER="/out"
if [[ ! ${OUTPUT_PATH_HOST} ]]; then
    echo "output_dir is not set" && usage && exit 1
fi
OUTPUT_PATH_HOST=$( dirname "${OUTPUT_FILE_HOST}" )
OUTPUT_BASENAME=$( basename "${OUTPUT_FILE_HOST}" )

INPUT_PATH_CONTAINER="/in"
if [[ ! ${INPUT_FILE_HOST} ]]; then
    echo "output_dir is not set" && usage && exit 1
fi
INPUT_PATH_HOST=$( dirname "${INPUT_FILE_HOST}" )
INPUT_BASENAME=$( basename "${INPUT_FILE_HOST}" )

CODE_PATH_CONTAINER="/code"
CODE_PATH_HOST=${PROJECT_ROOT}

echo "******* LAUNCHING IMAGE ${SIMAGE_FILENAME} *******"
echo "  "
echo "  HOST OPTIONS:"
echo "  code path:            ${CODE_PATH_HOST}"
echo "  abc path:             ${ABC_PATH_HOST}"
echo "  input path:           ${INPUT_PATH_HOST}"
echo "  output path:          ${OUTPUT_PATH_HOST}"
echo "  "
echo "  CONTAINER OPTIONS:"
echo "  code path:            ${CODE_PATH_CONTAINER}"
echo "  abc path:             ${ABC_PATH_CONTAINER}"
echo "  input path:           ${INPUT_PATH_CONTAINER}"
echo "  output path:          ${OUTPUT_PATH_CONTAINER}"
echo "  "

N_TASKS=$(( SLURM_CPUS_PER_TASK ))
MAKE_DATA_SCRIPT="${CODE_PATH_CONTAINER}/scripts/data_scripts/compute_dataset_statistics.py"
CONFIGS_PATH_CONTAINER="${CODE_PATH_CONTAINER}/scripts/data_scripts/configs"
CONFIG="${CONFIGS_PATH_CONTAINER}/${DATASET_CONFIG}"
INPUT_FILE="${INPUT_PATH_CONTAINER}/${INPUT_BASENAME}"
OUTPUT_FILE="${OUTPUT_PATH_CONTAINER}/${OUTPUT_BASENAME}"

singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind "${ABC_PATH_HOST}":${ABC_PATH_CONTAINER} \
  --bind "${OUTPUT_PATH_HOST}":${OUTPUT_PATH_CONTAINER} \
  --bind "${INPUT_PATH_HOST}":${INPUT_PATH_CONTAINER} \
  --bind "${PWD}":/run/user \
  "${SIMAGE_FILENAME}" \
      bash -c 'export OMP_NUM_THREADS='"${SLURM_CPUS_PER_TASK}; \\
      python3 ${MAKE_DATA_SCRIPT} \\
        --input-file ${INPUT_FILE} \\
        --chunk ${CHUNK} \\
        --abc-input-dir ${ABC_PATH_CONTAINER} \\
        --output-file ${OUTPUT_FILE} \\
        --dataset-config ${CONFIG} \\
        --io-schema ${SCHEMA} \\
        --jobs ${N_TASKS} \\
        ${VERBOSE_ARG} \\
        "
