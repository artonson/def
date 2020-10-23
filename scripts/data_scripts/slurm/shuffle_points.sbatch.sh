#!/bin/bash

#SBATCH --job-name=sharpf-shuffle
#SBATCH --output=logs/shuffle_points_%A.out
#SBATCH --error=logs/shuffle_points_%A.err
#SBATCH --time=24:00:00
#SBATCH --partition=htc
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16g
#SBATCH --oversubscribe

#module load apps/singularity-3.2.0

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

N_TASKS=${SLURM_CPUS_PER_TASK}
MAKE_DATA_SCRIPT="${CODE_PATH_CONTAINER}/sharpf/utils/abc_utils/scripts/defrag_shuffle_split_hdf5.py"
CHUNK_SIZE=16384
TRAIN_FRACTION=1.0
RANDOM_SEED=9675
MAX_LOADED_FILES=10

singularity exec \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind "${INPUT_PATH_HOST}":${INPUT_PATH_CONTAINER} \
  --bind "${OUTPUT_PATH_HOST}":${OUTPUT_PATH_CONTAINER} \
  --bind /gpfs:/gpfs \
  --bind "${PWD}":/run/user \
  "${SIMAGE_FILENAME}" \
      bash -c 'export OMP_NUM_THREADS='"${SLURM_CPUS_PER_TASK}; \\
      python3 ${MAKE_DATA_SCRIPT} \\
        --input-dir ${INPUT_PATH_CONTAINER} \\
        --output-dir ${OUTPUT_PATH_CONTAINER} \\
        --num-items-per-file ${CHUNK_SIZE} \\
        --train-fraction ${TRAIN_FRACTION} \\
        --jobs ${N_TASKS} \\
        --random-shuffle \\
        --random-seed ${RANDOM_SEED} \\
        --max-loaded-files ${MAX_LOADED_FILES} \\
         ${VERBOSE_ARG} \\
       -fk has_smell_coarse_surfaces_by_num_faces \\
       -fk has_smell_coarse_surfaces_by_angles \\
       -fk has_smell_sharpness_discontinuities \\
       "

#      -fk has_smell_mismatching_surface_annotation
#      -fk has_smell_bad_face_sampling
#      -fk has_smell_deviating_resolution

#      -fk has_smell_raycasting_background \\
#      -fk has_smell_depth_discontinuity \\
#      -fk has_smell_mesh_self_intersections \\

