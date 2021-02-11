### Installation

Build the docker container:
```bash
./docker/build_docker.sh [-p to optionally upload to DockerHub under your username]
```
This command will also install all the required dependences for the project. They are contained
in `requirements.txt` in the project root.

Then enter the container by running it (don't forget to name the containers accordingly and remove them post-usage!):
```bash
docker run --rm -it --name 3ddl.artonson.0.sharp_features --runtime=nvidia -v /home/artonson/repos/FloorplanVectorization:/code -p 3340:3340 artonson:vectran
```
or
```bash
./docker/run_docker.sh -d <server_data_dir> -l <server_logs_dir> -g <gpu-indexes>
```
Remember our container naming conventions: `3ddl.<username>.<gpu-ids-list>.<customsuffix>`.

After entering the container shell, you will be able to run the Jupyter notebook:
```bash
jupyter notebook --NotebookApp.token=abcd --ip=0.0.0.0 --port 10003 --no-browser
```
and your token will be `abcd`.

Remember to login into the server using:
```bash
ssh -A -L 10003:localhost:10003 <servername>
```
to enable port forwarding.

Then just open `http://localhost:10003` and enter the `abcd` token.
