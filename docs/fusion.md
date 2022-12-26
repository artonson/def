# Running reconstruction on complete 3D models

We expect that predictions obtained by the DEF model are located in 
a known directory and have the following example structure: 
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
run the following script in docker container:
```bash

```


For obtaining predictions on complete **real-world** 3D models,
run the following script in docker container:
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
