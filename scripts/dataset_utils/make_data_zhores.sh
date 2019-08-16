#!/usr/bin/env bash

sbatch dataset_filter_zhores.sh

sbatch --dependency=afterany: dataset_patching_zhores.sh

sbatch --dependency=afterany: dataset_torch_zhores.sh

