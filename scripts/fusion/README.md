# `scripts/fusion/`

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

 * `configs/` configuration files used to pass parameters for fusion.
 * `shellscripts/` a set of (example) scripts for running our pipelines. 
Please adapt the scripts to your server configuration.
 * `slurm/` a set of (example) scripts for running oru pipelines using a Singularity cluster
orchestrated with SLURM system. Please adapt the scripts to your cluster configuration.
 * `wrappers/` a set of launcher scripts (commonly invoking respective slurm jobs) 
for batch processing of multiple IDs. 
 * `fuse_images.py` given predictions for **real-world** individual views in the range-image format, 
compute fused predictions for complete 3D models. 
 * `fuse_points.py` given predictions for **real-world** individual views in the point patch format,
  compute fused predictions for complete 3D models. 
 * `fuse_images_synthetic.py` given predictions for **synthetic** individual views in the range-image format,
  compute fused predictions for complete 3D models.
 * `fuse_points_synthetic.py` given predictions for **synthetic** individual views in the point patch format,
  compute fused predictions for complete 3D models. 
 * `fusion_analysis.py` computes absolute differences between predictions and ground-truth. 
 * `fused_images_to_pienet.py` converts data to PIE-Net format. 