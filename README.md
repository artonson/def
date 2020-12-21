# [DEF: Deep Estimation of Sharp Geometric Features in 3D Shapes](https://arxiv.org/abs/2011.15081)

## Dependencies
```bash
python -m venv ~/.venv/defs-env
source ~/.venv/defs-env/bin/activate
pip install -r requirements.txt
python setup.py build develop
```

## Data binding
```bash
mkdir data
# patches
ln -s /gpfs/gpfs0/3ddl/sharp_features/data_v2_cvpr/points data/points
ln -s /gpfs/gpfs0/3ddl/sharp_features/data_v3_cvpr/images data/images
```

## Download pretrained models
You can download weights from [here](https://yadi.sk/d/dAcmcOLk2Q4GVQ?w=1)

## Experiments

##### DEF-Image (high-res, zero noise, regression)
```bash
# test on patches
python train_net.py trainer.gpus=1 callbacks=regression datasets=abc-image-64k model=unet2d-hist transform=depth-regression system=def-image-regression hydra.run.dir=test/def-image-regression eval_only=true test_weights=pretrained_models/def-image-regression-high-0.ckpt

# train
python train_net.py trainer.gpus=1 callbacks=regression datasets=abc-image-64k model=unet2d-hist transform=depth-regression system=def-image-regression hydra.run.dir=experiments/def-image-regression
```

##### DEF-Image (high-res, zero noise, segmentation)
```bash
# test on patches
python train_net.py trainer.gpus=1 callbacks=segmentation datasets=abc-image-64k model=unet2d-seg transform=depth-regression-seg system=def-image-segmentation hydra.run.dir=test/def-image-segmentation eval_only=true test_weights=pretrained_models/def-image-segmentation-high-0.ckpt

# train
python train_net.py trainer.gpus=4 callbacks=segmentation datasets=abc-image-64k model=unet2d-seg transform=depth-regression-seg system=def-image-segmentation hydra.run.dir=experiments/def-image-segmentation
```

##### DEF-Image-Arbitrary (high-res, zero noise, regression)
```bash
# test on unlabeled patches
python train_net.py trainer.gpus=1 datasets.path=\${hydra:runtime.cwd}/examples/20201113_castle_45.hdf5 callbacks=regression datasets=unlabeled-image model=unet2d-hist transform=depth-sl-regression-arbitrary system=def-image-regression hydra.run.dir=test/20201113_castle_45 eval_only=true test_weights=pretrained_models/def-image-arbitrary-regression-high-0.ckpt

# test on patches
python train_net.py trainer.gpus=1 callbacks=regression datasets=abc-image-arbitrary-64k model=unet2d-hist transform=depth-regression-arbitrary system=def-image-regression hydra.run.dir=test/def-image-arbitrary-regression eval_only=true test_weights=pretrained_models/def-image-arbitrary-regression-high-0.ckpt

# train
python train_net.py trainer.gpus=4 callbacks=regression datasets=abc-image-arbitrary-64k model=unet2d-hist transform=depth-regression-arbitrary system=def-image-regression hydra.run.dir=experiments/def-image-arbitrary-regression
```

##### DEF-Image-Arbitrary (high-res, zero noise, segmentation)
```bash
# test on patches
python train_net.py trainer.gpus=1 callbacks=segmentation datasets=abc-image-arbitrary-64k model=unet2d-seg transform=depth-regression-seg-arbitrary system=def-image-segmentation hydra.run.dir=test/def-image-arbitrary-segmentation eval_only=true test_weights=pretrained_models/def-image-arbitrary-segmentation-high-0.ckpt

# train
python train_net.py trainer.gpus=4 callbacks=segmentation datasets=abc-image-arbitrary-64k model=unet2d-seg transform=depth-regression-seg-arbitrary system=def-image-segmentation hydra.run.dir=experiments/def-image-arbitrary-segmentation
```

##### DEF-Points (high-res, zero noise, regression)
```bash
# test on patches
python train_net.py trainer.gpus=1 callbacks=regression datasets=abc-pcv-64k model=dgcnn-d6w158-hist model.in_channels=4 transform=pc-voronoi system=def-points-regression hydra.run.dir=test/def-points-regression eval_only=true test_weights=pretrained_models/def-points-regression-high-0.ckpt dataloader.total_batch_size=4

# train
python train_net.py trainer.gpus=4 callbacks=regression datasets=abc-pcv-64k model=dgcnn-d6w158-hist model.in_channels=4 transform=pc-voronoi system=def-points-regression hydra.run.dir=experiments/def-points-regression
```


##### DEF-Points (high-res, zero noise, segmentation)
```bash
# test on patches
python train_net.py trainer.gpus=1 callbacks=segmentation datasets=abc-pcv-64k model=dgcnn-d6w158-seg model.in_channels=4 transform=pc-voronoi-segmentation system=def-points-segmentation hydra.run.dir=test/def-points-segmentation eval_only=true test_weights=pretrained_models/def-points-segmentation-high-0.ckpt dataloader.total_batch_size=4

# train
python train_net.py trainer.gpus=4 callbacks=segmentation datasets=abc-pcv-64k model=dgcnn-d6w158-seg model.in_channels=4 transform=pc-voronoi-segmentation system=def-points-segmentation hydra.run.dir=experiments/def-points-segmentation
```

##### DEF-Points w/o VCM in input (high-res, zero noise, regression)
```bash
# test on patches
python train_net.py trainer.gpus=1 callbacks=regression datasets=abc-pc-64k model=dgcnn-d6w158-hist transform=pc-basic system=def-points-regression hydra.run.dir=test/def-points-wo-v-regression eval_only=true test_weights=pretrained_models/def-points-wo-v-regression-high-0.ckpt dataloader.total_batch_size=4

# train
python train_net.py trainer.gpus=4 callbacks=regression datasets=abc-pc-64k model=dgcnn-d6w158-hist transform=pc-basic system=def-points-regression hydra.run.dir=experiments/def-points-wo-v-regression
```

##### DEF-Points w/o VCM in input (high-res, zero noise, segmentation)
```bash
# test on patches
python train_net.py trainer.gpus=1 callbacks=segmentation datasets=abc-pc-64k model=dgcnn-d6w158-seg transform=pc-segmentation system=def-points-segmentation hydra.run.dir=test/def-points-wo-v-segmentation eval_only=true test_weights=pretrained_models/def-points-wo-v-segmentation-high-0.ckpt dataloader.total_batch_size=4

# train
python train_net.py trainer.gpus=1 callbacks=segmentation datasets=abc-pc-64k model=dgcnn-d6w158-seg transform=pc-segmentation system=def-points-segmentation hydra.run.dir=experiments/def-points-wo-v-segmentation
```

Some project parts are inspired by or based on [Detectron2](https://github.com/facebookresearch/detectron2) and [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) code.