# sharp_features
Learning to Detect Sharp Geometric Features for 3D Shape Reconstruction

### Using the code

Build the docker container:
```bash
./docker/build_docker.sh [-p to optionally upload to DockerHub]
```

Then enter the container by running it (don't forget to name the containers accordingly and remove them post-usage!):
```bash
docker run --rm -it --name 3ddl.artonson.0.sharp_features --runtime=nvidia -v /home/artonson/repos/FloorplanVectorization:/code -p 3340:3340 artonson:vectran
```
or
```
./docker/run_docker.sh -d <server_data_dir> -l <server_logs_dir> -g <gpu-indexes>
```
Remember our container naming conventions: `3ddl.<username>.<gpu-ids-list>.<customsuffix>`.
This command also install all the required dependences for the project. They are contained
in `requirements.txt` in the project root.

After entering the container shell, you will be able to run the Jupyter notebook:
```bash
cd /code
jupyter notebook --NotebookApp.token=abcd --ip=0.0.0.0 --port 3340 --allow-root
```
and your token will be `abcd`.

Remember to login into the server using:
```bash
ssh -A -L 3340:localhost:3340 <servername>
```
to enable port forwarding.

Then just open `http://localhost:3340` and enter the `abcd` token.
