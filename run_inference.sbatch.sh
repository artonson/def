#!/bin/bash

#SBATCH --job-name=def-inference
#SBATCH --output=/trinity/home/a.artemov/tmp/def_inference/%A_%a.out
#SBATCH --error=/trinity/home/a.artemov/tmp/def_inference/%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_devel
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8g

__usage="
Usage: $0 -i input_file -o output_dir [-t task_type -d data_reader_type -c checkpoint_file -m model_type -f transform_type -s system_type] [-v]

  -i:   HDF5 input filename
  -o:   directory where predictions will be written to
  -t: 	task type [segmentation, regression]
  -d: 	data type [unlabeled-image, regression]
  -c: 	checkpoint filename
  -s: 	system name [def-image-regression, def-image-segmentation,
          def-points-regression, def-points-segmentation]
  -m: 	model type [unet2d-hist, unet2d-seg, dgcnn-d6w158-seg, dgcnn-d6w158-hist]
  -f: 	transform type [depth-regression, depth-regression-seg,
          depth-sl-regression-arbitrary, depth-regression-arbitrary, depth-regression-seg-arbitrary,
          pc-voronoi, pc-voronoi-segmentation, pc-basic, def-points-segmentation]
  -l:   server logs dir
"

usage() { echo "$__usage" >&2; }

# Get all the required options and set the necessary variables
VERBOSE=false
TASK_TYPE=regression
DATA_READER_TYPE=unlabeled-image
MODEL_TYPE=unet2d-hist
TRANSFORM_TYPE=depth-regression-arbitrary
SYSTEM=def-image-regression
CHECKPOINT=

while getopts "i:o:vt:d:c:m:s:l:" opt
do
    case ${opt} in
        i) INPUT_PATH_HOST=$OPTARG;;
        o) OUTPUT_PATH_HOST=$OPTARG;;
        t) TASK_TYPE=$OPTARG;;
        d) DATA_READER_TYPE=$OPTARG;;
        c) CHECKPOINT=$OPTARG;;
        m) MODEL_TYPE=$OPTARG;;
        s) SYSTEM=$OPTARG;;
        l) LOGS_PATH_HOST=$OPTARG;;
        v) VERBOSE=true;;
        *) usage; exit 1 ;;
    esac
done

if [[ "${VERBOSE}" = true ]]; then
    set -x
    VERBOSE_ARG="--verbose"
fi

INPUT_PATH_CONTAINER="/in"
if [[ ! ${INPUT_PATH_HOST} ]]; then
    echo "input_file is not set" && usage && exit 1
fi

OUTPUT_PATH_CONTAINER="/out"
if [[ ! ${OUTPUT_PATH_HOST} ]]; then
    echo "output_dir is not set" && usage && exit 1
fi

if [[ ! ${TASK_TYPE} ]]; then
    echo "task_type (callback) is not set" && usage && exit 1
fi

if [[ ! ${CHECKPOINT} ]]; then
    echo "checkpoint_file is not set" && usage && exit 1
fi

LOGS_PATH_CONTAINER="/logs"
if [[ ! ${LOGS_PATH_HOST} ]]; then
    echo "logs_dir is not set" && usage && exit 1
fi

CODE_PATH_CONTAINER="/code"
CODE_PATH_HOST="/trinity/home/a.artemov/repos/sharp_features_ruslan"


echo "******* LAUNCHING IMAGE ${SIMAGE_FILENAME} *******"
echo "  "
echo "  HOST OPTIONS:"
echo "  code path:            ${CODE_PATH_HOST}"
echo "  logs path:            ${LOGS_PATH_HOST}"
echo "  input path:           ${INPUT_PATH_HOST}"
echo "  output path:          ${OUTPUT_PATH_HOST}"
echo "  "
echo "  CONTAINER OPTIONS:"
echo "  code path:            ${CODE_PATH_CONTAINER}"
echo "  logs path:            ${LOGS_PATH_CONTAINER}"
echo "  input path:           ${INPUT_PATH_CONTAINER}"
echo "  output path:          ${OUTPUT_PATH_CONTAINER}"
echo "  "

INPUT_FILENAME_CONTAINER=${INPUT_PATH_CONTAINER}/$( basename "${INPUT_PATH_HOST}" )
SCRIPT=/code/train_net.py

SIMAGE_FILENAME=/gpfs/gpfs0/3ddl/env/a1.sif

singularity exec \
  --nv \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  --bind "${DATA_PATH_HOST}":${DATA_PATH_CONTAINER} \
  --bind "${LOGS_PATH_HOST}":${LOGS_PATH_CONTAINER} \
  --bind "${OUTPUT_PATH_HOST}":${OUTPUT_PATH_CONTAINER} \
  --bind /gpfs:/gpfs \
  --bind "${PWD}":/run/user \
  "${SIMAGE_FILENAME}" \
      bash -c "cd ${CODE_PATH_CONTAINER} && \\
        python setup.py build develop --user && \\
        python ${SCRIPT} \\
          trainer.gpus=1 \\
          eval_only=true \\
          callbacks=${TASK_TYPE} \\
          datasets=${DATA_READER_TYPE} \\
          datasets.path=${INPUT_FILENAME_CONTAINER} \\
          model=${MODEL_TYPE} \\
          transform=${TRANSFORM_TYPE} \\
          system=${SYSTEM} \\
          hydra.run.dir=${OUTPUT_PATH_CONTAINER} \\
          test_weights=${CHECKPOINT} \\
           1> >(tee ${LOGS_PATH_CONTAINER}/${INPUT_FILENAME}.out) \\
           2> >(tee ${LOGS_PATH_CONTAINER}/${INPUT_FILENAME}.err)"
