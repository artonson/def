# Running reconstruction on complete 3D models

We expect that predictions obtained by the DEF model are located in 
a known directory and have the following example structure
(see [network inference docs](inference.md) on how to run predictions): 
```bash
(base) [a.artemov@an01:/gpfs/gpfs0/3ddl/sharp_features/predictions/images_align4mm_fullmesh_whole/amed/92side_folder_images__align4mm_fullmesh_whole]$ll
total 3.5K
drwxrwsr-x  6 3ddl 3ddl 4.0K May  6  2021 .
drwxrwsr-x 56 3ddl 3ddl 4.0K May 17  2021 ..
drwxrwsr-x  3 3ddl 3ddl 4.0K May  6  2021 default
drwxrwsr-x  2 3ddl 3ddl 4.0K May  6  2021 .hydra
drwxrwsr-x  2 3ddl 3ddl 4.0K May  6  2021 predictions
drwxrwsr-x  3 3ddl 3ddl 4.0K May  6  2021 tb_logs
-rwxrwsr-x  1 3ddl 3ddl 1.1K May  6  2021 train_net.log
```

For obtaining predictions on complete **synthetic** 3D models,
run the following shell code in docker container:
```bash
fuse_script="/code/scripts/fusion/fuse_images_synthetic.py"

shape="abc_0051_00512867_bb8ce171738b5deacb786b2d_008"
views_gt="/path/to/gt/${shape}.hdf5"
views_pred_dir="/path/to/preds/${shape}/predictions/"
output_path="/path/to/output/"

N_JOBS=10
PARAM_RESOLUTION_3D=0.02
PARAM_DISTANCE_INTERP_FACTOR=6.0
PARAM_NN_SET_SIZE=8
PARAM_INTERPOLATOR_FUNCTION=bisplrep

python3 ${fuse_script} \\
    --true-filename ${views_gt} \\
    --pred-dir ${views_pred_dir} \\
    --output-dir ${output_path} \\
    --jobs ${N_JOBS} \\
    --nn_set_size ${PARAM_NN_SET_SIZE} \\
    --resolution_3d ${PARAM_RESOLUTION_3D} \\
    --distance_interp_factor ${PARAM_DISTANCE_INTERP_FACTOR} \\
    --interpolator_function ${PARAM_INTERPOLATOR_FUNCTION}
```

For obtaining predictions on complete **real-world** 3D models,
run the following shell code in docker container:
```bash
fuse_script="/code/scripts/fusion/fuse_images.py"
config=/code/scripts/fusion/configs/real_images_base.yml

shape="92side_folder_images__align4mm_fullmesh_whole"
views_gt="/data/images_align4mm_fullmesh_whole/test/${shape}.hdf5"
views_pred_dir="/data/images_align4mm_fullmesh_whole/test/${shape}/predictions/"
output_path="/path/to/output/"

${fuse_script} \
    -t ${views_gt} \
    -p ${views_pred_dir} \
    -o ${output_path} \
    -j 10 \
    -f ${config} \
    -s 10.0 \
    -r 10.0
```
