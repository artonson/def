#!/usr/bin/env python3

import torch
import random
import argparse
import json
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
from tensorboardX import SummaryWriter

from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as torch_transforms
import torchvision.transforms.functional as TF

from seg_models_lib.segmentation_models_pytorch.unet import Unet
from seg_models_lib.segmentation_models_pytorch.utils.losses import BCEDiceLoss
from seg_models_lib.segmentation_models_pytorch.utils.train import PredictEpoch, ValidEpoch, TrainEpoch
from seg_models_lib.raster_metrics import precision_recall_fscore_iou_support

from sharpf.utils.abc_utils.hdf5.dataset import LotsOfHdf5Files

class Transform():
    def __init__(self):
        pass
    def forward(self, image, target):
        # Random rotation
        degrees = [10, 170]
        angle = random.uniform(degrees[0], degrees[1])#transforms.RandomRotation(degrees).get_params(degrees)
        image = TF.rotate(Image.fromarray(image.numpy()), angle)
        target = TF.rotate(Image.fromarray(target.numpy()), angle)

        # Random crop
        i, j, h, w = torch_transforms.RandomCrop.get_params(image, output_size=(512, 512))
        image = TF.crop(image, i, j, h, w)
        target = TF.crop(target, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            target = TF.vflip(target)

        # Transform to tensor
        image = TF.to_tensor(image)
        target = TF.to_tensor(target)

        return image, target

class DepthHDF5Dataset(LotsOfHdf5Files):
    def __init__(self, task, data_dir, data_label, target_label, partition=None,
                 transform=None, target_transform=None, max_loaded_files=0):
        self.task = task
        super().__init__(
            data_dir,
            data_label,
            target_label,
            partition=partition,
            transform=transform,
            target_transform=target_transform,
            max_loaded_files=max_loaded_files
        )

    def __getitem__(self, index):
        file_index = np.searchsorted(self.cum_num_items, index, side='right')
        relative_index = index - self.cum_num_items[file_index] if file_index > 0 else index
        data, target = self.files[file_index][relative_index]
        loaded_file_indexes = [i for i, f in enumerate(self.files) if f.is_loaded()]
        if len(loaded_file_indexes) > self.max_loaded_files:
            file_index_to_unload = np.random.choice(loaded_file_indexes)
            self.files[file_index_to_unload].unload()

        dist_new = np.copy(data)
        mask = ((data.numpy() != 0.0)).astype(float)
        mask[mask == 0.0] = np.nan

        if self.task == 'segmentation':
            dist_new *= mask
            dist1 = np.array((dist_new != np.nan) & (dist_new < 0.25)).astype(float)
            dist2 = np.array((dist_new != np.nan) & (dist_new > 0.25)).astype(float)
            target = torch.cat([torch.FloatTensor(dist1), torch.FloatTensor(dist2)], dim=0)
        elif self.task == 'regression':
            dist_norm = self.normalize(dist_new) * mask  # * 5.0
            dist_norm[np.isnan(dist_norm)] = 1.0
            target = torch.FloatTensor(dist_norm).unsqueeze(0)

        mask[np.isnan(mask)] = 0.0
        data = torch.FloatTensor(self.normalize(data))
        data = torch.cat([data, data, data, torch.FloatTensor(mask)], dim=0)
        print(data.shape, target.shape)
        return data, target



class precision_recall_fscore_iou():
    __name__ = ['prec', 'rec', 'f1', 'iou']
    def forward(self, pred, true):
        per_class_metric = []
        num_classes = pred.shape[0]
        for i_cl in range(num_classes):  # num of classes
            metric_value = precision_recall_fscore_iou_support(true[i_cl], pred[i_cl])
            per_class_metric.append(metric_value)

        return per_class_metric

class mse_loss(torch.nn.MSELoss):
    __name__ = 'mse loss'

LOSS = {
    'seg_bce_dice': BCEDiceLoss,
    'reg_mse_loss': mse_loss
}

def get_args():
    parser = argparse.ArgumentParser(description='Train segmentation model on depth maps and target sharp annotation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', dest='gpu', type=str, default='cpu',
                        help='GPU to use, can use multiple [default: use CPU].')

    parser.add_argument('-e', '--epochs', type=int, default=1, help='how many epochs to train [default: 1].')
    parser.add_argument('-b', '--train-batch-size', type=int, default=32, dest='train_batch_size',
                        help='train batch size [default: 128].')
    parser.add_argument('-b_val', '--val-batch-size', type=int, default=32, dest='val_batch_size',
                        help='val batch size [default: 128].')
    parser.add_argument('-ste', '--start-epoch', metavar='ste', type=int, default=0, dest='start_epoch',
                        help='Start epoch to summate with epoch number while training')
    parser.add_argument('--loss-funct', dest='loss_funct', type=str,
                        help='Choose loss function.')
    parser.add_argument('--lr', type=float, dest='lr', default=0.01)
    parser.add_argument('-aug', '--augmentations', type=bool, dest='aug', default=False)

    parser.add_argument('--model-spec', dest='model_spec_filename', required=True,
                        help='model specification JSON file to use [default: none].')

    parser.add_argument('--log-dir-prefix', dest='logs_dir', default='/logs',
                        help='path to root of logging location [default: /logs].')
    parser.add_argument('-dd', '--data-dir', dest='data_dir', default='/data',
                        help='Path to data directory [default=/data].')
    parser.add_argument('-od', '--output-dir', dest='output_dir', default='/output',
                        help='Path to directory for saving model output [default=/output].')

    parser.add_argument('-s', '--save-model', dest='save_model', default=True,
                        help='[default: true].')
    parser.add_argument('-l', '--load-model-file', dest='load_model_filename', default=None,
                        help='[default: none].')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter(args.logs_dir)
    cfg = json.load(open(args.model_spec_filename))
    model = Unet(
        encoder_name=cfg['encoder'],
        encoder_weights=None,
        classes=len(cfg['classes']),
        activation=cfg['activation']
    )
    model.to(args.gpu)

    loss = LOSS[args.loss_funct]
    if cfg['task'] == 'segmentation':
        loss = loss(eps=0.5, activation=cfg['activation'])
        metrics = [precision_recall_fscore_iou()]
    elif cfg['task'] == 'regression':
        metrics = []
        loss = loss()

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    data_label = 'image'
    target_label = 'distances'
    if args.aug:
        transforms = Transform()
    print('creating a train dataset')
    train_dataset = DepthHDF5Dataset(cfg['task'], args.data_dir, data_label, target_label, partition=None,
                 transform=transforms, target_transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    print('creating validation dataset')
    val_dataset = DepthHDF5Dataset(cfg['task'], args.data_dir, data_label, target_label, partition=None,
                 transform=transforms, target_transform=None)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
    
    train_epoch = TrainEpoch(
        cfg['task'],
        model, 
        loss=loss, 
        metrics=metrics,
        optimizer=optimizer,
        writer=writer,
        save_each_batch=10,
        device=args.gpu,
        verbose=True
    )
    val_epoch = ValidEpoch(
        cfg['task'],
        model, 
        loss=loss, 
        metrics=metrics,
        writer=writer,
        save_each_batch=10,
        device=args.gpu,
        verbose=True
    )

    PATH = '{}/seg_model_{}_epoch_{}'
    if args.load_model_filename is not None:
        model.load_state_dict(torch.load(args.load_model_filename))

    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch: {}'.format(epoch))

        train_logs = train_epoch.run(train_loader, epoch)
        if args.save_model:
            torch.save(model.state_dict(), PATH.format(args.output_dir, cfg['name'], epoch))
        val_logs = val_epoch.run(val_loader, epoch)

        # save log
        with open('{}/train_log_epoch_{}.json'.format(args.logs_dir, epoch), 'w') as fp:
            json.dump(train_logs, fp)
        with open('{}/val_log_epoch_{}.json'.format(args.logs_dir, epoch), 'w') as fp:
            json.dump(val_logs, fp)

    predict_epoch = PredictEpoch(
        cfg['task'],
        model,
        loss=loss,
        metrics=metrics,
        writer=writer,
        save_each_batch=10,
        save_dir=args.output_dir,
        device=args.gpu,
        verbose=True
    )

    with tqdm(val_loader, desc='prediction', file=sys.stdout, disable=False) as iterator:
        for i, iter_ in enumerate(iterator):
            x, y = iter_
            x, y = x.to(args.gpu), y.to(args.gpu)

            loss, y_pred = predict_epoch.batch_update(x, y)

            # save prediction
            torch.save(y_pred, '{}/prediction_{}_by_{}_model'.format(args.output_dir, i+1, cfg['name']))

    writer.close()

