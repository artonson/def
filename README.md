# Learning to Detect Sharp Geometric Features in Point Clouds

### Install additional packages
```bash
singularity shell --bind /gpfs:/gpfs --nv /gpfs/gpfs0/3ddl/env/a1.sif
pip install --no-cache-dir --use-feature=2020-resolver 'git+https://github.com/PytorchLightning/pytorch-lightning'
pip install --no-cache-dir --use-feature=2020-resolver 'git+https://github.com/facebookresearch/hydra'
pip install --no-cache-dir --use-feature=2020-resolver 'git+https://github.com/rwightman/pytorch-image-models'
python setup.py build develop --user
```

### Train & test the network
```bash
python train_net.py hydra.run.dir=experiments/my_exp trainer.gpus=4 trainer.max_epochs=10 model=dgcnn-4k datasets=abc-pointcloud transforms=pc-basic task=regression evaluators=regression
```

### Test the network
```bash
python train_net.py <args> eval_only=true test_weights=<the weights/checkpoint path>
```

### Send the job to the Slurm server
```bash
sbatch sharp_features/submit.sh <args as above>
```

