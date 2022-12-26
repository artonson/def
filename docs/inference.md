# Running inference with a pre-trained DEF network

For the most part, running testing with a pre-trained DEF 
model is described in [training manual](training.md). 

Here, we mention a helper SLURM script [`run_inference.sbatch.sh`]()https://github.com/artonson/def/blob/main/sharpf/neural/run_inference.sbatch.sh
which serves as an interface to the inference script.
You can use the script as an example for building you own
inference script. 

The output structure for the inference looks like this:
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
which is what is required for the [fusion method](fusion.md)
to work. 
