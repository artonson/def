# Generating synthetic training datasets

To generate either point-based or image-based training data, we use several similarly 
structured Python scripts:
 * [`scripts/data_scripts/generate_depthmap_data.py`](https://github.com/artonson/def/blob/main/scripts/data_scripts/generate_depthmap_data.py) 
 * [`scripts/data_scripts/generate_pointcloud_data.py`](https://github.com/artonson/def/blob/main/scripts/data_scripts/generate_pointcloud_data.py) 
* [`scripts/data_scripts/generate_fused_depthmap_data.py`](https://github.com/artonson/def/blob/main/scripts/data_scripts/generate_fused_depthmap_data.py)
* [`scripts/data_scripts/generate_fused_pointcloud_data.py`](https://github.com/artonson/def/blob/main/scripts/data_scripts/generate_fused_pointcloud_data.py) 

## Example data generation 

Assuming your ABC 7z files are in `/path/to/data`, 
code is in `/path/to/code`,
and desired output is in `/path/to/outputs`
you can run the docker image via 
```bash
cd /path/to/code
docker/run_docker.sh \
  -d /path/to/data \
  -l /path/to/outputs \
  -u 
```
you will see a message produced in the form
```
**** LAUNCHING CONTAINER artonson/def:latest *******
      Pushing you to /code directory
      Data is at /data
      Writable logs are at /logs
      Environment: PYTHONPATH=/code
      Environment: CUDA_VISIBLE_DEVICES=0,1,2,3
      Exposed ports:
      User in container: --user=1001:1001
groups: cannot find name for group ID 1001
```
then you can run generation with the following shell code. 
This will process items from 0th to 100th index in chunk 0 of ABC:
```bash
cd /code 

export DATA_PATH_CONTAINER=/data  # path to ABC *obj*.7z and *feat*.7z files
export CODE_PATH_CONTAINER=/code  # path to this repo root
export OUTPUT_PATH_CONTAINER=/logs   # path to outputs and logs
export MAKE_DATA_SCRIPT="${CODE_PATH_CONTAINER}/scripts/data_scripts/generate_depthmap_data.py"
export N_TASKS=4  # process this many items in parallel
export CONFIGS_PATH="${CODE_PATH_CONTAINER}/scripts/data_scripts/configs/depthmap_datasets"  # 
export DATASET_PATH="${CONFIGS_PATH}/high_res_whole.json"
export SLICE_START=0  # item ID to start at (inclusive)
export SLICE_END=100  # item ID to stop at (exclusive)
export CHUNK=0  # which chunk to use 
export VERBOSE_ARG="--verbose"  # whether to use verbose logging

python3 ${MAKE_DATA_SCRIPT} \\
    --input-dir ${DATA_PATH_CONTAINER} \\
    --chunk ${CHUNK} \\
    --output-dir ${OUTPUT_PATH_CONTAINER} \\
    --jobs ${N_TASKS} \\
    -n1 ${SLICE_START} \\
    -n2 ${SLICE_END} \\
    --dataset-config ${DATASET_PATH} \\
     ${VERBOSE_ARG}
```
Let the script run for a while, and it will produce 
data and logs (both in console and in the output 
folder). 
Running the other data generation scripts is completely similar
to running the previous code snippet. 


## Precomputed datasets *(DEF-Sim)*

To foster further research, we have precomputed and publish
a few datasets as our DEF-Sim data. 

### Patch-based datasets (training and evaluation)

These datasets were used to train our models. 
For image-based datasets, the resolution of 64x64 (4096 points, maximum) has been chosen. 
For point-based datasets, 4096 points are sampled from each patch. 
We do not release datasets corresponding to high-resolution (r = 0.02)
models with noise levels 0.0025, 0.01, and 0.04 to save storage. 

The following datasets are available by the 
[synthetic_data/patches/](https://www.dropbox.com/scl/fo/o1iwodlqs1ksd0riiymuq/h?dl=0&rlkey=37oc14dg1m5f0jzh6t1prjtbw) folder: 
| **Link**                                                  | **Modality** | **Resolution** | **Sampling distance** | **Noise level**  |
|-----------------------------------------------------------|--------------|----------------|-----------------------|------------------|
| `def_sim-images-high-0-arbitrary-patches-*.tar.gz`        | depth images | 64 x 64        | 0.02                  | 0                |
| `def_sim-images-high-0.005-arbitrary-patches-*.tar.gz`    | depth images | 64 x 64        | 0.02                  | 0.005 (SNR=4:1)  |
| `def_sim-images-high-0.02-arbitrary-patches-*.tar.gz`     | depth images | 64 x 64        | 0.02                  | 0.02 (SNR=1:1)   |
| `def_sim-images-high-0.08-arbitrary-patches-*.tar.gz`     | depth images | 64 x 64        | 0.02                  | 0.08 (SNR=1:4)   |
| `def_sim-images-med-0-arbitrary-patches-*.tar.gz`         | depth images | 64 x 64        | 0.05 (2.5x)           | 0                |
| `def_sim-images-low-0-arbitrary-patches-*.tar.gz`         | depth images | 64 x 64        | 0.125 (2.5^2x)        | 0                |
| `def_sim-points-high-0-arbitrary-patches-*.tar.gz`        | point clouds | 4096           | 0.02                  | 0                |
| `def_sim-points-high-0.005-arbitrary-patches-*.tar.gz`    | point clouds | 4096           | 0.02                  | 0.005 (SNR=4:1)  |
| `def_sim-points-high-0.02-arbitrary-patches-*.tar.gz`     | point clouds | 4096           | 0.02                  | 0.02 (SNR=1:1)   |
| `def_sim-points-high-0.08-arbitrary-patches-*.tar.gz`     | point clouds | 4096           | 0.02                  | 0.08 (SNR=1:4)   |
| `def_sim-points-med-0-arbitrary-patches-*.tar.gz`         | point clouds | 4096           | 0.05 (2.5x)           | 0                |
| `def_sim-points-low-0-arbitrary-patches-*.tar.gz`         | point clouds | 4096           | 0.125 (2.5^2x)        | 0                |

### Complete 3D model datasets (intended for evaluation only)

Please note that these shapes are high-resolution, densely 
sampled point clouds typically with millions of points. 

The folder [synthetic_data/complete_models/](https://www.dropbox.com/scl/fo/o1iwodlqs1ksd0riiymuq/h?dl=0&rlkey=37oc14dg1m5f0jzh6t1prjtbw)
contains all the files.

| **Link**                                  | **Modality** | **Resolution** | **Sampling distance** | **Noise level**  | **Num. views** | **Num. shapes** |
|-------------------------------------------|--------------|----------------|-----------------------|------------------|----------------|-----------------|
| `def_sim-images-high-0-18views.tar.gz`    | depth images | 1024 x 1024    | 0.02                  | 0                | 18             | 85              | 
| `def_sim-images-high-0-128views.tar.gz`   | depth images | 1024 x 1024    | 0.02                  | 0                | 128            | 95              |
| `def_sim-images-high-0.005.tar.gz`        | depth images | 1024 x 1024    | 0.02                  | 0.005 (SNR=4:1)  | 18             | 87              |
| `def_sim-images-high-0.02.tar.gz`         | depth images | 1024 x 1024    | 0.02                  | 0.02 (SNR=1:1)   | 18             | 87              |
| `def_sim-images-high-0.08.tar.gz`         | depth images | 1024 x 1024    | 0.02                  | 0.08 (SNR=1:4)   | 18             | 87              |
| `def_sim-images-med.tar.gz`               | depth images | 1024 x 1024    | 0.05 (2.5x)           | 0                | 18             | 100             |
| `def_sim-images-low.tar.gz`               | depth images | 1024 x 1024    | 0.125 (2.5^2x)        | 0                | 18             | 104             |
| `def_sim-points-high-0.tar.gz`            | point clouds | various        | 0.02                  | 0                | point patches  | 84              |
| `def_sim-points-high-0.005.tar.gz`        | point clouds | various        | 0.02                  | 0.005 (SNR=4:1)  | point patches  | 83              |
| `def_sim-points-high-0.02.tar.gz`         | point clouds | various        | 0.02                  | 0.02 (SNR=1:1)   | point patches  | 84              |
| `def_sim-points-high-0.08.tar.gz`         | point clouds | various        | 0.02                  | 0.08 (SNR=1:4)   | point patches  | 83              |
| `def_sim-points-med-0.tar.gz`             | point clouds | various        | 0.05 (2.5x)           | 0                | point patches  | 103             |
| `def_sim-points-low-0.tar.gz`             | point clouds | various        | 0.125 (2.5^2x)        | 0                | point patches  | 75              |
