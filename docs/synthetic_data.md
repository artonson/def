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

### Complete 3D model datasets (intended for evaluation only)

Please note that these shapes are high-resolution, densely 
sampled point clouds typically with millions of points. 

| **Link**                                                                                                                      | **Modality** | **Resolution** | **Sampling distance** | **Noise level** | **Num. views** | **Num. shapes** |
|-------------------------------------------------------------------------------------------------------------------------------|--------------|----------------|-----------------------|-----------------|---------------|-----------------|
| [def_sim-images-high-0-18views.tar.gz](https://www.dropbox.com/s/v27gwxm9js3r6p7/def_sim-images-high-0-18views.tar.gz?dl=0)   | depth images | 1024 x 1024    | 0.02                  | 0               | 18            | 85              | 
| [def_sim-images-high-0-128views.tar.gz](https://www.dropbox.com/s/2klbt953s684dn1/def_sim-images-high-0-128views.tar.gz?dl=0) | depth images | 1024 x 1024    | 0.02                  | 0               | 128           |                 |
| [def_sim-images-high-0.005.tar.gz](https://www.dropbox.com/s/6foffi2unaoqtk4/def_sim-images-high-0.005.tar.gz?dl=0)           | depth images | 1024 x 1024    | 0.02                  | 0.005 (SNR=4:1) | 18            |                 |
| [def_sim-images-high-0.02.tar.gz](https://www.dropbox.com/s/o19o35zzq12ui49/def_sim-images-high-0.02.tar.gz?dl=0)             | depth images | 1024 x 1024    | 0.02                  | 0.02 (SNR=1:1)  | 18            |                 |
| [def_sim-images-high-0.08.tar.gz](https://www.dropbox.com/s/cg25f2z7olkayto/def_sim-images-high-0.08.tar.gz?dl=0)             | depth images | 1024 x 1024    | 0.02                  | 0.08 (SNR=1:4)  | 18            |                 |
| [def_sim-images-med.tar.gz](https://www.dropbox.com/s/ea166q3rmjubw30/def_sim-images-med-0.tar.gz?dl=0)                       | depth images | 1024 x 1024    | 0.05 (2.5x)           | 0               | 18            |                 |
| [def_sim-images-low.tar.gz](https://www.dropbox.com/s/rnuvw2j2id5ntdf/def_sim-images-low-0.tar.gz?dl=0)                       | depth images | 1024 x 1024    | 0.125 (2.5^2x)        | 0               | 18            |                 |
