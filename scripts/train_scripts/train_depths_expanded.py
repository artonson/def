import os
import torch
import random
import argparse
import json
import glob
import h5py
import time
import sys
sys.path.append('/trinity/home/g.bobrovskih/sharp_features_in_progress/')

import numpy as np
from tqdm import tqdm
from PIL import Image
from tensorboardX import SummaryWriter

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as torch_transforms
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
import torch.nn as nn
import torchvision

from scripts.train_scripts.seg_models_lib.segmentation_models_pytorch.unet import Classifier, Unet, two_stage_unet
from scripts.train_scripts.seg_models_lib.segmentation_models_pytorch.afm_model.deeplabv3plus import DeepLabv3_plus
from scripts.train_scripts.seg_models_lib.segmentation_models_pytorch.utils.losses import BCEDiceLoss
from scripts.train_scripts.seg_models_lib.segmentation_models_pytorch.utils.train import PredictEpoch, ValidEpoch, TrainEpoch
from scripts.train_scripts.seg_models_lib.raster_metrics import precision_recall_fscore_iou_support

from sharpf.data.datasets.hdf5_datasets import Hdf5File, LotsOfHdf5Files

high_res_quantile = 7.4776
med_res_quantile = 69.0811
low_res_quantile = 1.0


class Transform():
    def __init__(self):
        pass

    def forward(self, image, target):
        # Random 90rot
        # if 'numpy' in str(type(image)):

        if random.random() > 0.5:
            image = TF.rotate(Image.fromarray(image.numpy()), 90)
            target = TF.rotate(Image.fromarray(target.numpy()), 90)
        # Random crop
        # i, j, h, w = torch_transforms.RandomCrop.get_params(image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # target = TF.crop(target, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            if 'torch.Tensor' in str(type(image)):
                image = TF.hflip(Image.fromarray(image.numpy()))
                target = TF.hflip(Image.fromarray(target.numpy()))
            else:
                image = TF.hflip(image)
                target = TF.hflip(target)

        # Random vertical flipping
        if random.random() > 0.5:
            if 'torch.Tensor' in str(type(image)):
                image = TF.hflip(Image.fromarray(image.numpy()))
                target = TF.hflip(Image.fromarray(target.numpy()))
            else:
                image = TF.hflip(image)
                target = TF.hflip(target)

        # Transform to tensor

        # image = TF.to_tensor(image)
        # target = TF.to_tensor(target)

        return np.array(image), np.array(target)


class DepthHDF5Dataset(Dataset):
    def __init__(self, task, data_dir, save_mask=False, partition=None, normalisation=None,
                 transform=None, target_transform=None, max_loaded_files=0, quality=None):
        self.task = task
        self.data_dir = data_dir
        self.data_label = 'image'
        self.target_label = 'distances'
        self.quality = self._get_quantity()
        self.save_mask = save_mask
        self.normalisation = normalisation

        if None is not partition:
            data_dir = os.path.join(data_dir, partition)
        filenames = glob.glob(os.path.join(data_dir, '*.hdf5'))
        self.files = []

        for i, filename in enumerate(filenames):
            print('preload', True * bool(i > max_loaded_files))
            self.files.append(Hdf5File(filename, self.data_label, self.target_label,
                                       transform=transform, target_transform=target_transform,
                                       preload=bool(i < max_loaded_files)))

        self.cum_num_items = np.cumsum([len(f) for f in self.files])
        self.current_file_idx = 0
        self.max_loaded_files = max_loaded_files
        self.loaded_file_indexes = 0
        self.previous_loaded_id = 0

    def __len__(self):
        if len(self.cum_num_items) > 0:
            return self.cum_num_items[-1]
        return 0

    def _get_quantity(self):
        data_dir_split = self.data_dir.split('_')
        if 'high' in data_dir_split:
            return 'high'
        elif 'low' in data_dir_split:
            return 'low'
        elif 'med' in data_dir_split:
            return 'med'

    def quantile_normalize(self, data):
        # mask -> min shift -> quantile

        norm_data = np.copy(data)
        mask_obj = np.where(norm_data != 0)
        mask_back = np.where(norm_data == 0)
        norm_data[mask_back] = norm_data.max() + 1.0 # new line
        norm_data -= norm_data[mask_obj].min()

        norm_data /= high_res_quantile

        return norm_data

    def standartize(self, data):
        # zero mean, unit variance

        standart_data = np.copy(data)
        standart_data -= np.mean(standart_data)
        std = np.linalg.norm(standart_data, axis=1).max()
        if std > 0:
            standart_data /= std

        return standart_data

    def _getdata(self, index):

        file_index = np.searchsorted(self.cum_num_items, index, side='right')
        relative_index = index - self.cum_num_items[file_index] if file_index > 0 else index

        data, target = self.files[file_index][relative_index]
        if file_index != self.previous_loaded_id:
            self.loaded_file_indexes += 1

        # loaded_file_indexes = [i for i, f in enumerate(self.files) if f.is_loaded()]
        if self.loaded_file_indexes > self.max_loaded_files:
            print('unloading {}'.format(self.previous_loaded_id))
            self.files[self.previous_loaded_id].unload()
            self.loaded_file_indexes -= 1
        self.previous_loaded_id = file_index

        return data, target

    def __getitem__(self, index):

        data, target = self._getdata(index)

        mask_1 = (np.copy(data) != 0.0).astype(float) # mask for object
        mask_2 = np.where(data == 0) # mask for background

        if 'quantile_normalize' in self.normalisation:
            data = self.quantile_normalize(data)
        if 'standartize' in self.normalisation:
            data = self.standartize(data)

        dist_new = np.copy(target)
        dist_mask = dist_new * mask_1  # select object points
        dist_mask[mask_2] = 1.0  # background points has max distance to sharp features
        close_to_sharp = np.array((dist_mask != np.nan) & (dist_mask < 1.)).astype(float)

        if self.task == 'regression_segmentation' or self.task == 'two-heads':
            # regression + segmentation (or two-head network) has to targets:
            # distance field and segmented close-to-sharp region of the object
            target = torch.cat(
                [torch.FloatTensor(dist_mask).unsqueeze(0), torch.FloatTensor(close_to_sharp).unsqueeze(0)], dim=0)
        if self.task == 'segmentation':
            # dist_new *= mask_1
            # dist1 = np.array((dist_new != np.nan) & (dist_new < 0.15)).astype(float)
            # dist2 = np.array((dist_new != np.nan) & (dist_new > 0.15)).astype(float)
            # target = torch.cat([torch.FloatTensor(dist1).unsqueeze(0), torch.FloatTensor(dist2).unsqueeze(0)], dim=0)
            target = torch.FloatTensor(mask_1)
        elif self.task == 'regression' or self.task == 'two-stages':
            dist_mask = dist_new * mask_1  # * 5.0
            dist_mask[mask_2] = 1.0
            target = torch.FloatTensor(dist_mask)
        elif self.task == 'classification':
            # classify has sharp/ doesn't have sharp
            dist_mask = dist_new*mask_1
            dist1 = np.array((dist_mask != np.nan) & (dist_mask < 0.15)).astype(float)
            target = torch.FloatTensor([np.any(dist1 == 1.).astype(float)])
            # classify has obj / doesn't have obj
            # target = torch.FloatTensor([np.any(dist_new != 1.).astype(float)])

        mask = (dist_new == 1.0).astype(float)
        data = torch.FloatTensor(data).unsqueeze(0)
        if self.save_mask:
            mask = torch.FloatTensor(mask).unsqueeze(0)
            data = torch.cat([data, data, data, torch.FloatTensor(mask)], dim=0)
        else:
            data = torch.cat([data, data, data], dim=0)

        #print('target shape', target.shape)
        return data, target


# regression -> segmentation threshold equal to 1 resolution of dataset
class precision_recall_fscore_iou():
    __name__ = ['prec', 'rec', 'f1', 'iou']

    def forward(self, pred, true):
        per_class_metric = []
        num_classes = pred.shape[0]
        for i_cl in range(num_classes):  # num of classes
            metric_value = precision_recall_fscore_iou_support(true[i_cl], pred[i_cl])
            per_class_metric.append(metric_value)

        return per_class_metric


class abs_error():
    __name__ = 'abs error'

    def forward(self, pred, true):
        return np.abs(true - pred)


class lp_error():
    __name__ = 'rmse'
    def __init__(self, p=1):
        self.p = p

    def forward(self, pred, true):
        metric = (true - pred)**self.p
        return metric


class mse_loss(torch.nn.MSELoss):
    __name__ = 'mse loss'

    def forward(self, input, target, mask=None):
        if mask is None:
            return torch.mean(F.mse_loss(input, target, reduction='none')), \
                   lp_error(p=2).forward(input, target)
        else:
            # mask[mask == 1.0] = 5.0
            # mask[mask == 0.0] = 10.0
            return torch.mean(F.mse_loss(input, target, reduction='none') * mask), \
                   lp_error(p=2).forward(input, target)


class huber(torch.nn.SmoothL1Loss):
    __name__ = 'huber'


class kldivloss(torch.nn.KLDivLoss):
    __name__ = 'kldiv loss'

    def forward(self, prediction, target, mask=None):
        # function to transform model output
        from torch import erf
        def calculate_density(mean, sigma, discretization, a=0, b=1, margin=2):
            # compute how many bins should be (heuristic)
            #             sigma = 1 / discretization

            # allow margins from interval ranges for better accuracy
            a = a - margin * (1 / discretization)
            b = b + margin * (1 / discretization)

            num_bins = discretization + margin * 2

            # calculate bin limits
            tmp_bins = np.tile(np.linspace(a, b, num_bins + 1), (mean.shape[0], mean.shape[1], 1))
            bin_limits = torch.FloatTensor(tmp_bins).to('cuda')

            # normalizing coefficient
            Z = 0.5 * (erf((b - mean) / (np.sqrt(2) * sigma)) - erf((a - mean) / (np.sqrt(2) * sigma)))

            # compute discretized densities
            # c = torch.zeros((mean.shape[0], mean.shape[1], 240))

            # for i, bin in enumerate(bin_limits):
            tmp = erf((bin_limits[:, :, 1:] - mean.unsqueeze(-1)) / (np.sqrt(2) * sigma)) - erf(
                (bin_limits[:, :, :-1] - mean.unsqueeze(-1)) / (np.sqrt(2) * sigma))
            c = tmp / (2 * Z.unsqueeze(-1))

            # return the computed density and set of centers of the bins
            return c.permute(1, 0, 2, ).type(torch.float32), (bin_limits[:, :, :-1] + (1 / discretization / 2)).type(
                torch.float32), num_bins

        # densities = torch.zeros_like(input)
        #         data_resolution = 0.005
        discretization = 240  # heuristic
        sigma = 1 / discretization

        batch_sz = target.shape[0]
        tg_resized = target.reshape(batch_sz, -1)

        density, linspace, num_bins = calculate_density(torch.t(tg_resized), sigma, discretization)

        kldiv = super().forward(prediction.permute(0, 2, 3, 1).reshape(batch_sz, 4096, num_bins), density.to('cuda'))
        estimated_pred = torch.matmul(
            torch.exp(prediction.reshape(batch_sz, num_bins, -1).permute(0, 2, 1)[:, :, None, :].detach().cpu()),
            linspace.detach().cpu().permute(1, 0, 2)[:, :, :, None])
        estimated_pred = estimated_pred.reshape(batch_sz, 1, 64, 64)
        l2_error = lp_error(p=2).forward(estimated_pred, target.detach().cpu())
        return kldiv, l2_error.numpy()


class bce_loss(nn.BCEWithLogitsLoss):
    __name__ = 'bce loss'
    def __init__(self, weights, reduce='mean'):
        self.loss = super().__init__(reduce=reduce, weights=weights)

    def forward(self, prediction, target, mask=None):
        return self.loss.forward(prediction, target)


class FocalLoss(nn.Module):
    __name__='focal loss'
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets, mask=None):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class l1loss(nn.L1Loss):
    __name__ = 'l1 loss'

# two-heads (regression + segmentation) loss function
class field_seg_loss(torch.nn.MSELoss):
    __name__ = 'field_seg_loss'

    def __init__(self, reg_loss, seg_loss, alpha_1, alpha_2, weights=None):
        super().__init__()
        self.reg_loss = LOSS[reg_loss]().to('cuda')
        self.weights = torch.FloatTensor(weights).cuda if weights is not None else weights
        self.weights = None
        self.seg_loss = FocalLoss(logits=True).to('cuda')
        self.alpha_1 = alpha_1

    def to(self, device):
        self.seg_loss.to(device)
        self.seg_loss.requires_grad = True
        self.reg_loss.to(device)

    def L_field(self, pred, target, mask=None):
        return self.reg_loss.forward(pred, target, mask)[0]  # compute regression only for object
                                                       # select [0] cause mse_loss outputs rmse_metric at [1]

    def L_seg(self, pred, target):
        return self.seg_loss.forward(pred, target)  # compute segmentation only for far from sharp pixels

    def forward(self, pred, target, mask=None):
        preds_reg, preds_seg = pred[0], pred[1]
        gt_reg, gt_seg = target[:, 0], target[:, 1]

        ones_tensor = torch.ones(preds_reg.size(), device='cuda')
        preds_reg_masked = preds_reg * preds_seg.round() + (ones_tensor - preds_seg.round())
        self.seg_loss_val = self.L_seg(preds_seg, gt_seg).mean()
        self.reg_loss_val = self.L_field(preds_reg_masked, gt_reg, mask=gt_seg)

        l2_error = lp_error(p=2).forward(preds_reg, gt_reg)  # metric
        return self.alpha_1 * self.seg_loss_val + self.reg_loss_val, l2_error

class two_heads_nn(Unet):
    def __init__(self, encoder_name='vgg16'):
        super().__init__(
            encoder_name=encoder_name,
            decoder_channels=(256, 128, 64, 32, 16),
            encoder_weights=None)

        self.regression_head = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=(1, 1))
        )

        self.freeze_regression = False
        self.freeze_segmentation = False

    def forward(self, x):
        encoded_x = self.encoder(x)
        features = self.decoder(encoded_x)
        reg, seg = None, None
        if not (self.freeze_regression):
            reg = self.regression_head(features).squeeze(1)
        if not (self.freeze_segmentation):
            seg = self.segmentation_head(features).squeeze(1)

        if reg is None:
            return seg
        elif seg is None:
            return reg

        return reg, seg

    def freeze_regression_head(self, mode=True):
        self.freeze_regression = mode
        if mode:
            self.regression_head.train(False)
        else:
            self.regression_head.train(True)
        # done

    def freeze_segmentation_head(self, mode=True):
        self.freeze_segmentation = mode
        if mode:
            self.segmentation_head.train(False)
        else:
            self.segmentation_head.train(True)
        # done

    def freeze_unet(self, mode=True):
        if mode:
            self.encoder.train(False)
            self.decoder.train(False)
        else:
            self.encoder.train(True)
            self.decoder.train(True)
        # done


LOSS = {
    'field_seg_loss': field_seg_loss,
    'seg_bce_dice': BCEDiceLoss,
    'reg_mse_loss': mse_loss,
    'reg_huber_loss': huber,
    'reg_kldiv_loss': kldivloss,
    'classif_ce_loss': bce_loss,
    'focal_loss': FocalLoss,
    'l1loss': l1loss
}

def iterate_epoches(start_epoch, epochs, args, cfg, train_epoch, val_epoch, train_loader, val_loader, name):
    for epoch in range(start_epoch, start_epoch + epochs + 1):
        print('\nEpoch: {}'.format(epoch))

        train_logs, another_loss, loss_values = train_epoch.run(train_loader, epoch)  # [first_batch], epoch)
        if args.save_model:
            torch.save(model.state_dict(), '{}/seg_model_{}_epoch_{}_{}'.format(args.output_dir+'/'+name, cfg['name'], epoch, name))
        with open('{}/train_{}_log_epoch_{}.json'.format(args.logs_dir, cfg['name'], epoch), 'w') as fp:
             json.dump(train_logs, fp)
        if epoch % 50 == 0:
            torch.save(torch.FloatTensor(another_loss), '{}/train_log_l2error_{}_epoch{}'.format(args.output_dir, cfg['name'], epoch))

        torch.save(loss_values, '{}/train_loss_values_{}_epoch{}'.format(args.output_dir, cfg['name'], epoch))
        val_logs, predictions, another_loss, loss_values = val_epoch.run(val_loader, epoch)  # [first_batch], epoch)
        with open('{}/val_{}_log_epoch_{}.json'.format(args.logs_dir, cfg['name'], epoch), 'w') as fp:
            json.dump(val_logs, fp)
        np.save('/{}/predictions_{}_{}.npy'.format(args.logs_dir, cfg['name'], name), predictions)

        if epoch % 50 == 0:
            torch.save(torch.FloatTensor(another_loss),
                       '{}/val_log_l2error_{}_epoch{}'.format(args.output_dir, cfg['name'], epoch))

        if epoch % 50 == 0:
            torch.save(loss_values, '{}/val_loss_values_{}_epoch{}'.format(args.output_dir, cfg['name'], epoch))


METRIC = {
    'precision_recall_fscore_iou': precision_recall_fscore_iou(),
    'l1_error': lp_error(p=1),
    'l2_error': lp_error(p=2),
    'mse': None


}

MAX_LOADED_FILES = 12


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
    parser.add_argument('--lr', type=float, dest='lr', default=0.01)
    parser.add_argument('-aug', '--augmentations', type=bool, dest='aug', default=False)
    parser.add_argument('--shuffle', type=bool, dest='shuffle', default=False)

    parser.add_argument('--model-spec', dest='model_spec_filename', required=True,
                        help='model specification JSON file to use [default: none].')

    parser.add_argument('--writer', dest='writer', type=bool, default=False, help='Tensorboard writer on/off')
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
    writer = SummaryWriter(args.logs_dir) if args.writer else None
    print('tensorboard writer:', writer)

    cfg = json.load(open(args.model_spec_filename))

    if cfg['task'] == 'two-heads':
        model = two_heads_nn(
            encoder_name=cfg['encoder'],
        )
    elif cfg['task'] == 'two-stages-classifier':
        cfg_classif = json.load(open('./sharpf/models/specs/image_based_regression/classification_model.json'))
        classifier = Classifier(
            encoder_name=cfg_classif['encoder'],
            encoder_weights=None,
            decoder_channels=[4096, 2048, 1024],
            classes=len(cfg_classif['classes']),
            activation=cfg_classif['activation'],
            attention=cfg_classif['attention'],
            attention_type=cfg_classif['attention_type']
        )
        print('loading {}'.format(cfg['classifier']))
        classifier.load_state_dict(torch.load('/gpfs/gpfs0/g.bobrovskih/classification/seg_model_classification_vgg19_bn_epoch_150'))
        model = two_stage_unet(
            encoder_name=cfg['encoder'],
            encoder_weights=None,
            classes=cfg['num_classes'],
            activation=cfg['activation'],
            classifier=classifier
        )
    elif cfg['task'] == 'classification':
        model = Classifier(
            encoder_name=cfg['encoder'],
            encoder_weights=None,
            decoder_channels=[512, 128, 16],
            classes=len(cfg['classes']),
            activation=cfg['activation'],
            attention=cfg['attention'],
            attention_type=cfg['attention_type']
        )
    else:  # regression or segmentation
        if cfg['encoder'] == 'atrous-residual':
            model = DeepLabv3_plus(nInputChannels=3, nOutChannels=cfg['num_classes'], os=16, pretrained=False, _print=True)
        else:
            model = Unet(
                encoder_name=cfg['encoder'],
                encoder_weights=None,
                classes=cfg['num_classes'],
                activation=cfg['activation']
            )

    model.to(args.gpu)
    if 'loss_params' in cfg.keys():
        loss = LOSS[cfg['loss_name']](**cfg['loss_params'])
    else:
        loss = LOSS[cfg['loss_name']]()
    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    metrics = []#[cfg['metrics_name']] if cfg['metrics_name'] is not None else []

    transforms = Transform() if args.aug else None
    print('transforms', True if transforms is not None else False)

    train_dataset = DepthHDF5Dataset(
                                     'segmentation',
                                     #cfg['task'],
                                     args.data_dir,
                                     partition='train/batched_16k',
                                     normalisation=cfg['normalisation'],
                                     transform=transforms,
                                     save_mask=False, # if save mask => data channels equals 4 else equals 3
                                     target_transform=None,
                                     max_loaded_files=MAX_LOADED_FILES
                                    )
    val_dataset = DepthHDF5Dataset(
                                   'segmentation',
                                   #cfg['task'],
                                   args.data_dir,
                                   partition='val/batched_16k',
                                   normalisation=cfg['normalisation'],
                                   transform=None,
                                   save_mask=False,
                                   target_transform=None,
                                   max_loaded_files=MAX_LOADED_FILES
                                  )
    print('train set size:', len(train_dataset))
    print('validation set size:', len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size)

    print('datasets created')
    train_epoch = TrainEpoch(
        'segmentation',
        model,
        loss=LOSS['seg_bce_dice'](),
        metrics=metrics,
        optimizer=optimizer,
        writer=writer,
        save_each_batch=1,
        save_predictions=False,
        device=args.gpu,
        visualization=True,
        verbose=True
    )
    val_epoch = ValidEpoch(
        'segmentation',
        model,
        loss=LOSS['seg_bce_dice'](),
        metrics=metrics,
        writer=writer,
        save_each_batch=1,
        save_predictions=True,
        device=args.gpu,
        visualization=True,
        verbose=True
    )

    # train segmentation first

    # model.freeze_regression_head(mode=True)

    PATH = '{}/seg_model_{}_epoch_{}'
    if args.load_model_filename is not None and args.load_model_filename != 'None':
        model.load_state_dict(torch.load(args.load_model_filename))

    name = 'segmentation'
    #model.load_state_dict(torch.load('/gpfs/gpfs0/g.bobrovskih/kldiv/vgg16/seg_model_reg_unet_vgg16_bn_epoch_457'))
    iterate_epoches(0, 200, args, cfg, train_epoch, val_epoch, train_loader, val_loader, name)

    # train regression next
    print('train regression')
    model.freeze_regression_head(mode=False)
    model.freeze_segmentation_head(mode=True)
    model.freeze_unet(mode=True)

    optimizer = torch.optim.Adam(model.parameters(), 0.01)

    train_dataset = DepthHDF5Dataset(
        'regression',
        # cfg['task'],
        args.data_dir,
        partition='train/batched_16k',
        normalisation=cfg['normalisation'],
        transform=transforms,
        save_mask=False,  # if save mask => data channels equals 4 else equals 3
        target_transform=None,
        max_loaded_files=MAX_LOADED_FILES
    )
    val_dataset = DepthHDF5Dataset(
        'regression',
        # cfg['task'],
        args.data_dir,
        partition='val/batched_16k',
        normalisation=cfg['normalisation'],
        transform=None,
        save_mask=False,
        target_transform=None,
        max_loaded_files=MAX_LOADED_FILES
    )

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size)

    train_epoch = TrainEpoch(
        'regression',
        model,
        loss=LOSS['reg_mse_loss'](),
        metrics=metrics,
        optimizer=optimizer,
        writer=writer,
        save_each_batch=1,
        save_predictions=False,
        device=args.gpu,
        visualization=True,
        verbose=True
    )
    val_epoch = ValidEpoch(
        'regression',
        model,
        loss=LOSS['reg_mse_loss'](),
        metrics=metrics,
        writer=writer,
        save_each_batch=1,
        save_predictions=True,
        device=args.gpu,
        visualization=True,
        verbose=True
    )

    name = 'regression'
    iterate_epoches(0, 100, args, cfg, train_epoch, val_epoch, train_loader, val_loader, name)

    # train two-heads
    print('train two-heads')
    model.freeze_regression_head(mode=False)
    model.freeze_segmentation_head(mode=False)
    model.freeze_unet(mode=False)

    train_dataset = DepthHDF5Dataset(
        cfg['task'],
        # cfg['task'],
        args.data_dir,
        partition='train/batched_16k',
        normalisation=cfg['normalisation'],
        transform=transforms,
        save_mask=False,  # if save mask => data channels equals 4 else equals 3
        target_transform=None,
        max_loaded_files=MAX_LOADED_FILES
    )
    val_dataset = DepthHDF5Dataset(
        cfg['task'],
        # cfg['task'],
        args.data_dir,
        partition='val/batched_16k',
        normalisation=cfg['normalisation'],
        transform=None,
        save_mask=False,
        target_transform=None,
        max_loaded_files=MAX_LOADED_FILES
    )

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size)

    train_epoch = TrainEpoch(
        cfg['task'],
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        writer=writer,
        save_each_batch=1,
        save_predictions=False,
        device=args.gpu,
        visualization=True,
        verbose=True
    )
    val_epoch = ValidEpoch(
        cfg['task'],
        model,
        loss=loss,
        metrics=metrics,
        writer=writer,
        save_each_batch=1,
        save_predictions=True,
        device=args.gpu,
        visualization=True,
        verbose=True
    )

    name = 'two-heads'
    iterate_epoches(0, 200, args, cfg, train_epoch, val_epoch, train_loader, val_loader, name)

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
            torch.save(y_pred, '{}/predictions/prediction_{}_by_{}_model'.format(args.output_dir, i+1, cfg['name']))
            if iter_ > 5:
                break

    if writer is not None:
        writer.close()