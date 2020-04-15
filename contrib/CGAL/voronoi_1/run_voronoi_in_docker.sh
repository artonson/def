#!/bin/bash

set -e

# example launch string:
# ./run_voronoi_in_docker.sh -R 0.2 -r 0.05 -t 0.15 -i /home/artonson/data -o /home/artonson/output -v
#   R, r, t:                    Voronoi method parameters
#   /home/artonson/data:        the data directory where the training sample resides
#   /home/artonson/output:      the directory where the output files are to be written
#   v:                          be verbose

usage() { echo "Usage: ${0} -i input_file -o output_file [-R offset_radius] [-r conv_radius] [-t threshold] [-j num_jobs]" >&2; }

V_OFFSET_RADIUS=0.2
V_CONV_RADIUS=0.1
V_THRESHOLD=0.16
NUM_JOBS=1
while getopts "i:o:R:r:t:j:" opt
do
    case ${opt} in
        i) INPUT_HDF5_FILENAME=${OPTARG} ;;
        o) OUTPUT_HDF5_FILENAME=${OPTARG} ;;
        R) V_OFFSET_RADIUS=${OPTARG} ;;
        r) V_CONV_RADIUS=${OPTARG} ;;
        t) V_THRESHOLD=${OPTARG} ;;
        j) NUM_JOBS=${OPTARG} ;;
        *) usage; exit 1 ;;
    esac
done

# HOST_<anything> refers to paths OUTSIDE container, i.e. on host machine
# CONT_<anything> refers to paths INSIDE container

[[ -f ${INPUT_HDF5_FILENAME} ]] || { echo "input_file not set or empty"; usage; exit 1; }
HOST_INPUT_DIR="$( cd "$( dirname "${INPUT_HDF5_FILENAME}" )" >/dev/null 2>&1 && pwd )"
CONT_INPUT_DIR="/input"

[[ -f ${OUTPUT_HDF5_FILENAME} ]] || { echo "output_file not set or empty"; usage; exit 1; }
HOST_OUTPUT_DIR="$( cd "$( dirname "${OUTPUT_HDF5_FILENAME}" )" >/dev/null 2>&1 && pwd )"
CONT_OUTPUT_DIR="/output"

CONT_CODE_DIR="/home/user/code"

HOST_PY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )../../hdf5_utils" >/dev/null 2>&1 && pwd )"
CONT_PY_DIR="/code/hdf5_utils"


IMAGE_NAME="gbobrovskih/cgal_4-14:latest"
docker inspect --type=image ${IMAGE_NAME} >/dev/null || docker pull ${IMAGE_NAME}

CONTAINER_NAME="3ddl.$(whoami).$(uuidgen).voronoi_R${V_OFFSET_RADIUS}_r${V_CONV_RADIUS}_thresh${V_THRESHOLD}.sharp_features"

echo "******* LAUNCHING IMAGE ${IMAGE_NAME} IN CONTAINER ${CONTAINER_NAME} *******"
echo "  "
echo "  HOST OPTIONS:"
echo "  input path:           ${HOST_INPUT_DIR}"
echo "  output path:          ${HOST_OUTPUT_DIR}"
#echo "  code path:            ${HOST_CODE_DIR}"
echo "  py code path:         ${HOST_PY_DIR}"
echo "  "
echo "  CONTAINER OPTIONS:"
echo "  input path:           ${CONT_INPUT_DIR}"
echo "  output path:          ${CONT_OUTPUT_DIR}"
echo "  code path:            ${CONT_CODE_DIR}"
echo "  py code path:         ${CONT_PY_DIR}"
echo "  logs path:            ${CONT_OUTPUT_DIR}"


docker run \
    --name ${CONTAINER_NAME} \
    --rm \
    --env PYTHONPATH=${CONT_CODE_DIR} \
    -v ${HOST_PY_DIR}:${CONT_PY_DIR} \
    -v ${HOST_INPUT_DIR}:${CONT_INPUT_DIR} \
    -v ${CONT_OUTPUT_DIR}:${CONT_OUTPUT_DIR} \
    ${IMAGE_NAME} \
    /bin/bash \
        -c "${CONT_CODE_DIR}/run_voronoi.sh \\
            -i input_file \\
            -o output_file \\
            -R ${V_OFFSET_RADIUS} \\
            -r ${V_CONV_RADIUS} \\
            -t ${V_THRESHOLD} \\
            -j ${NUM_JOBS}
        1>${CONT_OUTPUT_DIR}/out.out \\
        2>${CONT_OUTPUT_DIR}/err.err"

