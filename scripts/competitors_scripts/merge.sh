#!/usr/bin/env bash

#SBATCH --job-name=sharpf-data
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --partition=htc
set -x

# example launch string:
# ./run_sharpness_fields_in_docker.sh -i <input_dir> -l <label> -s <meshlab_script> -d <docker image name> -c <docker container name> -g <gpu-indexes>
#	-i: 	input directory with .hdf5 files
#	-d: 	docker image name
#	-c: 	docker container name
#	-g: 	comma-separated gpu indexes

usage() { echo "Usage: $0 -i <input_folder> -f <input_file> -d <data_label> -o <output_file>" >&2; }

while getopts "i:f:d:o:" opt
do
    case ${opt} in
        i) INPUT_FOLDER=$OPTARG;;
        f) INPUT_FILE=$OPTARG;;
        d) DATA_LABEL=$OPTARG;;
        o) OUTPUT_FILE=$OPTARG;;
        *) usage; exit 1 ;;
    esac
done

if [[ ! ${INPUT_FOLDER} ]]; then
    echo "input_folder is not set";
    usage
    exit 1
fi

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

if [[ ! ${DATA_LABEL} ]]; then
    echo "data_label is not set";
    usage
    exit 1
fi

module load python/python-3.7.1 

pip3.7 install plyfile

python3.7 ~/sharp_features/contrib/hdf5_utils/merge_hdf5_ecnet.py \
        -i ${INPUT_FOLDER} \
        --input_file ${INPUT_FILE} \
        --input_label ${DATA_LABEL} \
        -o ${OUTPUT_FILE} \
        --input_format ply
