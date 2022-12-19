
## Installation

The preferred way of setting up the environment required to run out method
is by building and using a docker image. The docker image contains all the 
necessary software required to run, debug, and develop all of the 
constituent parts of our approach. We thus strongly recommend using
either a pre-built docker image available on DockerHub, or using the 
ready-made scripts to build your own docker image locally.


### Building the environment
Before building the docker image, take a look at the file `env.sh` in the 
project root. If you plan to build your own docker image, please change 
the `$IMAGE_NAME` variable in `env.sh`.

To build the docker image from the source code available in this repository,
run the command:
```bash
bash docker/build_docker.sh [-p to optionally upload to DockerHub under your username]
```
This command will also install all the required dependences for the project. They are contained
in `requirements.txt` in the project root.

To get the docker image from DockerHub, run the command
```bash
docker pull artonson/def:latest
```


### Running the environment 

To enter the container by running it (don't forget to name the containers 
accordingly and remove them post-usage!), issue the following command
(see [docker run](https://docs.docker.com/engine/reference/run/) for a reference):
```bash
docker run \
  --rm \
  -it \
  --name 3ddl.artonson.0.def \ 
  --runtime=nvidia \
  -v /home/artonson/repos/def:/code \
  -v /home/artonson/data:/data \
  -p 8888:8888 \
  artonson/def:latest
```
Alternatively, for convenience, you can use our prepared script like this: 
```bash
./docker/run_docker.sh -d <server_data_dir> -l <server_logs_dir> -g <gpu-indexes>
```
See [run_docker.sh](docker/run_docker.sh) for a usage reference. 
Container name will be automatically assigned using this script as: 
`3ddl.<username>.<gpu-ids-list>.<customsuffix>` to prevent name clashes in the server.

After entering the container shell, you will be able to run the Jupyter notebook 
from under the container to run experiments interactively via:
```bash
jupyter notebook --NotebookApp.token=abcd --ip=0.0.0.0 --port 8888 --no-browser
```
and your token will be `abcd`.

Remember to perform port forwarding by logging in to your server using:
```bash
ssh -A -L 8888:localhost:8888 <servername>
```
Then just open `http://localhost:8888` and enter the `abcd` token.


### Working pipeline 

We followed the normal working pipeline which enables code editing 
without re-launching the docker container. For this, the typical pipeline 
looks like this:
 * Run the docker container, mounting the source code folder
into the container via `-v` or `run_docker.sh`
 * Use `autoreload` option for ipython notebooks or run code
using python executable scripts. 
 * Whenever an update to the code is needed (e.g. for fixing a bug),
either edit the code directly on the server or perform `git pull`ing
(in this way, docker container performs update of the mounted `/code`
directory automatically). 
