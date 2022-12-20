# `scripts/data_scripts/`

The following is the brief explanation of the code in this directory. 
If you would like to know more about something, please feel free to read 
the sources.
If you find yourself really lost, you can try to contact the authors 
via [artonson at yandex ru], or [open an issue](https://github.com/artonson/def/issues/new).

This folder contains executable Python and shell scripts used within the project.
The use of these scripts is expected from a commandline within the Docker container,
or from a shell/SLURM script.
If the scripts are being run from a SLURM script, they are expected to spawn 
a Singularity container with an appropriate command inside it. 
Below we only describe the most relevant scripts.

 * `configs/` configuration files to generate datasets. 
 * `slurm/` a set of (example) scripts for running oru pipelines using a Singularity cluster
orchestrated with SLURM system. Please adapt the scripts to your cluster configuration.
 * `wrappers/` a set of launcher scripts (commonly invoking respective slurm jobs) 
for batch processing of multiple  
 * `utils/` additional scripts used for auxiliary tasks (please refer to their source code).
 * `compute_abcchunk_statistics.py` computes various statistics (e.g. distribution 
of numbers of curves/surfacs for an ABC chunk).
 * `compute_dataset_statistics.py` computes statistics of patch datasets (e.g. distribution 
of numbers of curves/surfaces for an slice of data).
 * `compute_mesh_patches.py` for real-world/synthetic data (entire point clouds), 
computes correspondences between point patches and local mesh regions. 
 * `generate_parametric_corners.py` extract corner information from ABC data. 
 * `generate_parametric_edges.py` extract parametric edge information from ABC data. 
 * `generate_depthmap_data.py` generate patch datasets in the form of depth images.
 * `generate_fused_depthmap_data.py` generate whole-model datasets in the form of depth images.
 * `generate_pointcloud_data.py` generate patch datasets in the form of point patches.
 * `generate_fused_pointcloud_data.py` generate whole-model datasets in the form of point patches.
 * `prepare_real_scans.py` given whole-model real-world scans converted from .x to .hdf5 format, 
save them in the ViewIO format (before annotation). 
 * `prepare_real_images_dataset.py` given data in ViewIO format (before annotation), 
compute annotations and save data in depth image-based AnnotatedViewIO format.
 * `prepare_real_points_dataset.py` given data in ViewIO format (before annotation), 
extract point patches, compute annotations, and save data in point patch-based 
AnnotatedViewIO format.
 * `prepare_points_fused_images_dataset.py` crop point patches from fused images. 
