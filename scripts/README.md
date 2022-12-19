# `scripts/`

The following is the brief explanation of the code in this directory. 
If you would like to know more about something, please feel free to read 
the sources.
If you find yourself really lost, you can try to contact the authors 
via [artonson at yandex ru], or [open an issue](https://github.com/artonson/def/issues/new).

This folder contains executable Python and shell scripts used within the project.
The use of these scripts is expected from a commandline within the Docker container,
or from a shell/SLURM script.

 * `data_scripts/` a set of scripts for preparing training and evaluation datasets.
 * `fusion/` a set of scripts for running reconstruction on complete 3D models.
 * `metrics/` a set of scripts for computing metrics. 
 * `vectorization/` a set of scripts for running extraction of parametric curves.
 * `visualization/` scripts for plotting images and 3D html snapshots.
 * `convert_views_to_points.py` takes image views with AnnotatedViewIO type and
produces data with WholePointCloudIO type. 
