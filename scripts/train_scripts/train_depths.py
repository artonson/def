import os
import torch
import numpy as np
import argparse
from tensorboardX import SummaryWriter
import json

from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torch.nn as nn
import torchvision

from seg_models_lib.segmentation_models_pytorch.unet import Classifier, Unet
from seg_models_lib.segmentation_models_pytorch.utils.losses import BCEDiceLoss
from seg_models_lib.segmentation_models_pytorch.utils.train import ValidEpoch, TrainEpoch

from tqdm import tqdm

from raster_metrics import precision_recall_fscore_iou_support

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
    'reg_mse': mse_loss
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
    parser.add_argument('--loss-funct', required=False, dest='loss_funct',
                        choices=list(LOSS.keys()),
                        help='Choose loss function. Default cross_entropy_loss')
    parser.add_argument('--lr', type=float, required=False, dest='lr', default=0.01)

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
    parser.add_argument('-s', '--load-model-file', dest='load_model_filename', default=None,
                        help='[default: none].')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter(args.logs)
    cfg = json.load(open(args.model_spec_filename))
    model = Unet(
        encoder_name=cfg['encoder'],
        encoder_weights=None,
        classes=len(cfg['classes']),
        activation=cfg['activation']
    )
    model.to(args.gpu)

    loss = LOSS[args.loss]
    if cfg['task'] == 'segmentation':
        loss = loss(eps=0.5, activation=cfg['activation'])
        metrics = precision_recall_fscore_iou()

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    print('creating a train dataset')
    train_dataset =
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    print('creating validation dataset')
    val_dataset =
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
    
    train_epoch = TrainEpoch(
        cfg['task'],
        model, 
        loss=loss, 
        metrics=metrics,
        optimizer=optimizer,
        writer=writer,
        save_each_batch=10,
        device=DEVICE,
        verbose=True
    )
    val_epoch = ValidEpoch(
        cfg['task'],
        model, 
        loss=loss, 
        metrics=metrics,
        writer=writer,
        save_each_batch=10,
        device=DEVICE,
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
        optimizer=optimizer,
        gradient_clipping=args.clip,
        writer=writer,
        save_each_batch=10,
        save_dir=args.dir,
        device=DEVICE,
        verbose=True
    )

    with tqdm(dataloader, desc='prediction', file=sys.stdout, disable=False) as iterator:
        for i, iter_ in enumerate(iterator):
            x, y = iter_
            x, y = x.to(self.device), y.to(self.device)

            loss, y_pred = self.batch_update(x, y)

            # save prediction
            torch.save('{}/prediction_{}_by_{}_model'.format(args.output_dir, i+1, cfg['name']))

    writer.close()

