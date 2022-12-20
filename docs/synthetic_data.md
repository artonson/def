# Generating synthetic training datasets

To generate either point-based or image-based training data, we use several similarly 
structured Python scripts:
 * [`scripts/data_scripts/generate_depthmap_data.py`](scripts/data_scripts/generate_depthmap_data.py) 
 * [`scripts/data_scripts/generate_pointcloud_data.py`](scripts/data_scripts/generate_pointcloud_data.py) 
* [`scripts/data_scripts/generate_fused_depthmap_data.py`](scripts/data_scripts/generate_fused_depthmap_data.py)
* [`scripts/data_scripts/generate_fused_pointcloud_data.py`](scripts/data_scripts/generate_fused_pointcloud_data.py) 

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
you will see logs produced in the form
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

export DATA_PATH_CONTAINER=/data  # path to ABC *obj*.7z and *features*.7z files
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
Running the other scripts from this folder is fully similar. 
