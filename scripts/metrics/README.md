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

 * `compute_metrics_paper.py` computes FPR, Recall, RMSE metrics (PyTorch Lighting 
compatible metrics). 
 * `compute_metrics.py` computes additional metrics.
 * `compute_metrics_pienet_crop_patches.py` metrics for PIE-Net models. 
 * `compute_metrics_pienet_rw.py` metrics for PIE-Net models. 
 