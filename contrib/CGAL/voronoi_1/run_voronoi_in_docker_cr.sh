#!/bin/bash

#SBATCH --job-name=sharpf-data
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=cpu_big
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1

# set -e
# set -x

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
INPUT_FILENAME="$( basename "${INPUT_HDF5_FILENAME}" )"
CONT_INPUT_DIR="/input"

HOST_OUTPUT_DIR="$( cd "$( dirname "${OUTPUT_HDF5_FILENAME}" )" >/dev/null 2>&1 && pwd )"
[[ -d ${HOST_OUTPUT_DIR} ]] || { echo "output directory ${HOST_OUTPUT_DIR} needs to be created first"; usage; exit 1; }
OUTPUT_FILENAME="$( basename "${OUTPUT_HDF5_FILENAME}" )"
CONT_OUTPUT_DIR="/output"

HOST_CODE_DIR="/trinity/home/a.matveev/sharp_features/contrib/CGAL/voronoi_1"
CONT_CODE_DIR="/code"

HOST_PY_DIR="/trinity/home/a.matveev/sharp_features/contrib/hdf5_utils"
CONT_PY_DIR="/code/hdf5_utils"


SIMAGE_FILENAME=/gpfs/gpfs0/a.matveev/root_cgal_4-14-2020-08-27-8fd1f854b75e.sif
# docker inspect --type=image ${IMAGE_NAME} >/dev/null || docker pull ${IMAGE_NAME}

CONTAINER_NAME="3ddl.$(whoami).$(uuidgen).voronoi_R${V_OFFSET_RADIUS}_r${V_CONV_RADIUS}_thresh${V_THRESHOLD}.sharp_features"

echo "******* LAUNCHING IMAGE ${IMAGE_NAME} IN CONTAINER ${CONTAINER_NAME} *******"
echo "  "
echo "  HOST OPTIONS:"
echo "  input path:           ${HOST_INPUT_DIR}"
echo "  output path:          ${HOST_OUTPUT_DIR}"
echo "  code path:            ${HOST_CODE_DIR}"
echo "  py code path:         ${HOST_PY_DIR}"
echo "  "
echo "  CONTAINER OPTIONS:"
echo "  input path:           ${CONT_INPUT_DIR}"
echo "  output path:          ${CONT_OUTPUT_DIR}"
echo "  code path:            ${CONT_CODE_DIR}"
echo "  py code path:         ${CONT_PY_DIR}"
echo "  logs path:            ${CONT_OUTPUT_DIR}"


singularity exec \
    --bind ${HOST_INPUT_DIR}:${CONT_INPUT_DIR} \
    --bind ${HOST_OUTPUT_DIR}:${CONT_OUTPUT_DIR} \
    --bind ${HOST_CODE_DIR}:${CONT_CODE_DIR} \
    --bind ${HOST_PY_DIR}:${CONT_PY_DIR} \
    --bind "${PWD}":/run/user \
  "${SIMAGE_FILENAME}" \
    /bin/bash \
        -c "for R_param in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;
        do for r_param in 0.05 0.1 0.2 0.25 0.3 0.4 0.5 0.6 0.7;
        do for t_param in 0.05 0.1 0.2 0.25 0.3 0.4 0.5 0.6 0.7;
        do ${CONT_CODE_DIR}/run_voronoi_points.sh \\
            -i ${CONT_INPUT_DIR}/${INPUT_FILENAME} \\
            -o ${CONT_OUTPUT_DIR}/${R_param}_${r_param}_${t_param}.hdf5 \\
            -R  $R_param \\
            -r $r_param \\
            -t $t_param \\
            -j ${NUM_JOBS}
        1>${CONT_OUTPUT_DIR}/out.out \\
        2>${CONT_OUTPUT_DIR}/err.err;
        done;
        done;
        done"

