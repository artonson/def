# `docker/`

The following is the brief explanation of the code in this directory. 
If you would like to know more about something, please feel free to read 
the sources.
If you find yourself really lost, you can try to contact the authors 
via [artonson at yandex ru], or [open an issue](https://github.com/artonson/def/issues/new).

This folder contains shell scripts for building and using Docker 
and Singularity images for our project.
 * `build_docker.sh`: builds the Docker image for this project. 
 * `build_singularity.sh`: builds the Singularity image for this project
from an existing Docker image.
 * `Dockerfile`: contains the commands to build the docker image for this project.
 * `requirements.txt`: contains the set of Python dependencies for this project.
 * `run_docker.sh`: runs the Docker container for this project.
 * `run_singularity.sh`: runs the Singularity container for this project.
