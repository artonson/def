#!/bin/bash

#SBATCH --job-name=ecnet-train
#SBATCH --output=ecnet-train_%A_%a.out
#SBATCH --error=ecnet-train_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=40000

CODE_PATH_CONTAINER="/code"
CODE_PATH_HOST="/trinity/home/a.matveev/EC-Net"

SIMAGE_FILENAME=/gpfs/gpfs0/a.matveev/ec_net_tf.sif

echo "******* LAUNCHING IMAGE ${SIMAGE_FILENAME} *******"
echo "  "
echo "  HOST OPTIONS:"
echo "  code path:            ${CODE_PATH_HOST}"
echo "  "
echo "  CONTAINER OPTIONS:"
echo "  code path:            ${CODE_PATH_CONTAINER}"
echo "  "

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"



singularity exec \
  --nv \
  --bind ${CODE_PATH_HOST}:${CODE_PATH_CONTAINER} \
  "${SIMAGE_FILENAME}" \
      bash -c "cd ${CODE_PATH_CONTAINER}/code; \\
      python main.py --phase train --gpu 0 --batch_size 16 --log_dir ../model/my_model --max_epoch 200"

