#!/usr/bin/env bash

module load apps/singularity-3.2.0

# defaults

CONTAINER=${CONTAINER:-/trinity/home/a.artemov/vectran_sandbox}
# CODE_DIR=${CODE_DIR:-/trinity/home/a.artemov/FloorplanVectorization}
CODE_DIR=${CODE_DIR:-/trinity/home/a.artemov}
DATA_TRAIN=${DATA_TRAIN:-/gpfs/gpfs0/o.voinov/datasets/svg_datasets/preprocessed/sesyd_walls.train}
DATA_VAL=${DATA_VAL:-/gpfs/gpfs0/o.voinov/datasets/svg_datasets/preprocessed/sesyd_walls.val}
LOG_DIR=${LOG_DIR:-/trinity/home/a.artemov/vectran-logs-0}

echo container: $CONTAINER
echo code dir: $CODE_DIR
echo data train: $DATA_TRAIN
echo data_val $DATA_VAL
echo log dir: $LOG_DIR

echo "******* LAUNCHING CONTAINER ${CONTAINER} *******"

singularity shell --nv --bind $CODE_DIR:/code \
 --bind $DATA_TRAIN:/data_train \
 --bind $DATA_VAL:/data_val \
 --bind $LOG_DIR:/logs \
 --bind $PWD:/run/user \
$CONTAINER
