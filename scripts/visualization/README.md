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

 * `export_mesh_for_rendering.py` exports point as mesh for importing 
to Blender for rendering.
 * `plot_depth_sharpness_grid.py` given data in image format, renders images and 
saves to `.png` file.
 * `plot_snapshots.py` saves ground-truth and predictions snapshots in `.html` 
format for opening in the browser.