#!/usr/bin/env python3

import argparse
from collections import defaultdict
from itertools import islice
import os.path
import sys

import numpy as np
import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
)
sys.path[1:1] = [__dir__]

from sharpf.util.logging import create_logger
from sharpf.models import load_model
from sharpf.util.os import require_empty
from sharpf.data.data import ABCData
from sharpf.util.util import bce_loss, smooth_l1_loss, smooth_l1_reg_loss


LOSS = {
    'has_sharp': bce_loss,
    'segment_sharp': bce_loss,
    'regress_sharpdf': smooth_l1_loss,
    'regress_sharpdirf': smooth_l1_reg_loss
}


def make_loaders_fn(options):
    return DataLoader(ABCData(data_path=options.data_root, partition='train',
                              data_label=options.data_label,
                              target_label=options.target_label),
                      num_workers=1,
                      batch_size=options.train_batch_size, shuffle=False, drop_last=False), \
           DataLoader(ABCData(data_path=options.data_root, partition='test',
                              data_label=options.data_label, target_label=options.target_label),
                      batch_size=options.val_batch_size, shuffle=False, drop_last=False), \
           None  # add mini val


def prepare_batch_on_device(batch_data, device):
    data, label = batch_data
    data = data.float().to(device)
    label = label.float().to(device).squeeze()
    return data, label


def main(options):
    name, ext = os.path.splitext(os.path.basename(options.model_spec_filename))
    logs_dir = os.path.join(options.logs_dir, name)
    require_empty(logs_dir, recreate=options.overwrite)

    if None is options.logging_filename:
        options.logging_filename = os.path.join(logs_dir, 'train.log')
    if None is options.tboard_json_logging_file:
        options.tboard_json_logging_file = os.path.join(logs_dir, 'tboard.json')
    if None is options.tboard_dir:
        options.tboard_dir = logs_dir
    if None is options.save_model_filename:
        options.save_model_filename = os.path.join(logs_dir, 'model')

    if len(options.gpu) == 0:
        device = torch.device('cpu')
        prefetch_data = False
    elif len(options.gpu) == 1:
        device = torch.device('cuda:{}'.format(options.gpu[0]))
        prefetch_data = True
    else:
        raise ValueError('currenly only a single GPU is supported')

    # METRIC_PARAMS_AVG = {'average': 'mean', 'binarization': 'median', 'raster_res': RASTER_RES}
    # METRIC_PARAMS = {'binarization': 'median', 'raster_res': RASTER_RES}

    #####################################################
    # all these parameters stay unchanged for all models

    logger = create_logger(options)
    writer = SummaryWriter(options.tboard_dir)

    train_loader, val_loader, val_mini_loader = make_loaders_fn(options)
    if options.end_batch_train is not None:
        end_batch_train = int(options.end_batch_train)
    else:
        end_batch_train = len(train_loader)
    if options.end_batch_val is not None:
        end_batch_val = int(options.end_batch_val)
    else:
        end_batch_val = len(val_loader)

    logger.info('Total number of train patches: ~{}'.format(len(train_loader) * options.train_batch_size))
    logger.info('Total number of val patches: ~{}'.format(len(val_loader) * options.val_batch_size))
    if val_mini_loader is not None:
        logger.info('Total number of mini val patches: ~{}'.format(len(val_mini_loader) * options.val_batch_size))

    model = load_model(options.model_spec_filename).to(device)
    # from model_seg import DGCNN
    # from collections import namedtuple

    # Args = namedtuple('Args', ['k', 'dropout'])
    # args = Args(10, 0.5)
    # model = DGCNN(args)
    logger.info_trainable_params(model)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=options.lr)

    if options.scheduler == 'exp':
        scheduler = ExponentialLR(opt, gamma=0.5)
        optimizer = opt
    else:
        scheduler = None
        optimizer = ScheduledOptimizer(opt)

    if options.init_model_filename:
        checkpoint = torch.load(options.init_model_filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_completed = checkpoint['epoch']
        batches_completed_in_epoch = checkpoint['batch'] + 1
        optimizer.n_current_steps = batches_completed_in_epoch - 1
    else:
        epochs_completed = 0
        batches_completed_in_epoch = 0

    if end_batch_train < batches_completed_in_epoch:
        end_batch_train = batches_completed_in_epoch + 1

    # loss function choose
    criterion = LOSS[options.loss_funct]

    # def set_grad(var):
    #    def hook(grad):
    #        var.grad = grad
    #    return hook

    def validate(loader, log_i, prefix=''):
        val_metrics = defaultdict(list)
        val_loss = []
        # Model to eval
        model.eval()
        for batch_j, batch_data_val in islice(enumerate(loader), 0, end_batch_val):
            logger.info('Validating batch {}'.format(batch_j))
            # Run through validation dataset

            with logger.print_duration('    preparing batch on device'):
                data, label = prepare_batch_on_device(batch_data_val, device)

            with torch.no_grad():
                with logger.print_duration('    forward pass'):
                    preds = model.forward(data)  # currently model returns x, [f1, f2, f3]
    
            val_losses_per_sample = criterion(preds, label)
            val_loss.append(val_losses_per_sample)

            # y_true_vector = vmetrics.batch_numpy_to_vector(label.cpu().numpy(), RASTER_RES)
            # y_pred_vector = vmetrics.batch_numpy_to_vector(preds.cpu().numpy(), RASTER_RES)
            # for metric_name, metric in vmetrics.METRICS_BY_NAME.items():
            #    metric_value = metric(y_true_vector, y_pred_vector, **METRIC_PARAMS)
            #    metric_name = '{}val_{}'.format(prefix, metric_name)
            #    val_metrics[metric_name].append(metric_value)

        # Save stuff to tensorboard for visualization
        val_loss = np.asarray(val_loss)

        class _Loader:
            def __iter__(self):
                for batch_data_val in loader:
                    yield prepare_batch_on_device(batch_data_val, device)

        _loader = _Loader()

        # scores = np.concatenate(val_metrics['{}val_{}'.format(prefix, 'iou_score')])

        mean_val_loss = val_loss.mean()
        writer.add_scalar(('{}val_' + options.loss_funct).format(prefix), mean_val_loss, global_step=log_i)
        logger.info('Validation loss: {:.4f}'.format(mean_val_loss))
        metrics_scalars = {name: np.mean(np.concatenate(values)) for name, values in val_metrics.items()}
        for name, value in metrics_scalars.items():
            writer.add_scalar(name, value, global_step=log_i)
            values_array = np.concatenate(val_metrics[name])
            values_array[np.isnan(values_array)] = 0.  # maybe tensorboard does handle this by itself, dunno
            writer.add_histogram(name + '_hist', values_array, global_step=log_i)
        logger.info_scalars('Computed {key} over {num_items} images: {value:.4f}', metrics_scalars,
                            num_items=end_batch_val)

    for epoch_i in range(epochs_completed, epochs_completed + options.epochs):
        for batch_i, batch_data in islice(enumerate(train_loader), batches_completed_in_epoch, end_batch_train):

            model.train()
            iter_i = epoch_i * len(train_loader) + batch_i

            # Train for one batch
            logger.info('Training batch {}'.format(batch_i))
            with logger.print_duration('    preparing batch on device'):
                data, label = prepare_batch_on_device(batch_data, device)
                batch_size = data.shape[0]
            with logger.print_duration('    forward pass'):
                preds = model.forward(data)  # model returns x, (f1, f2, f3), saving only x
            loss = criterion(preds, label)
            # print(preds, label, loss.item())
            optimizer.zero_grad()

            with logger.print_duration('    backward pass'):
                loss.backward()
                optimizer.step()

            # Output loss for each training step, as it is already computed
            logger.info(
                'Training iteration [{item_idx} / {num_batches}] {percent:.3f}% complete, loss: {loss:.4f}'.format(
                    item_idx=batch_i, num_batches=end_batch_train,
                    percent=100. * batch_i / end_batch_train, loss=loss.item()))
            writer.add_scalar('learning_rate', np.array([param_group['lr'] for param_group in optimizer.param_groups]),
                              global_step=iter_i)
            writer.add_scalar(('train_' + options.loss_funct), loss.item(), global_step=iter_i)

            if batch_i > 0 and batch_i % options.batches_before_val == 0 and val_mini_loader is not None:
                logger.info('Running mini validation with {} batches'.format(len(val_mini_loader)))
                validate(val_mini_loader, iter_i, prefix='mini')
            if batch_i % options.batches_before_save == 0:
                weights_filename = '{prefix}_{batch_id}.weights'.format(
                    prefix=options.save_model_filename, batch_id=iter_i)
                print(weights_filename)
                torch.save({
                    'epoch': epoch_i,
                    'batch': batch_i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, weights_filename)
                logger.debug('Model for batch {} saved to {}'.format(batch_i, weights_filename))

            batches_completed_in_epoch = 0

        logger.info('Running validation on the whole validation dataset with {} batches'.format(len(val_loader)))
        validate(val_loader, (epoch_i + 1) * len(train_loader))

    if options.tboard_json_logging_file:
        writer.export_scalars_to_json(options.tboard_json_logging_file)

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default='',
                        help='GPU to use, can use multiple [default: use CPU].')

    parser.add_argument('-e', '--epochs', type=int, default=1, help='how many epochs to train [default: 1].')
    parser.add_argument('-b', '--train-batch-size', type=int, default=32, dest='train_batch_size',
                        help='train batch size [default: 128].')
    parser.add_argument('-B', '--val-batch-size', type=int, default=32, dest='val_batch_size',
                        help='val batch size [default: 128].')

    parser.add_argument('--batches-before-val', type=int, default=1024, dest='batches_before_val',
                        help='how many batches to train before validation [default: 1024].')
    parser.add_argument('--mini-val-batches-n-per-subset', type=int, default=12, dest='mini_val_batches_n_per_subset',
                        help='how many batches per subset to run for mini validation [default: 12].')
    parser.add_argument('--model-spec', dest='model_spec_filename', required=True,
                        help='model specification JSON file to use [default: none].')
    parser.add_argument('--infer-from-spec', dest='infer_from_spec', action='store_true', default=False,
                        help='if set, --model, --save-model-file, --logging-file, --tboard-json-logging-file,'
                             'and --tboard-dir are formed automatically [default: False].')
    parser.add_argument('--log-dir-prefix', dest='logs_dir', default='/logs',
                        help='path to root of logging location [default: /logs].')
    parser.add_argument('-m', '--init-model-file', dest='init_model_filename',
                        help='Path to initializer model file [default: none].')
    parser.add_argument('-s', '--save-model-file', dest='save_model_filename',
                        help='Path to output vectorization model file [default: none].')
    parser.add_argument('--batches_before_save', type=int, default=1024, dest='batches_before_save',
                        help='how many batches to run before saving the model [default: 1024].')

    parser.add_argument('--data-root', dest='data_root', help='root of the data tree (directory).')
    parser.add_argument('--num-points', type=int, default=1024, dest='num_points')

    parser.add_argument('--loss-funct', required=False, dest='loss_funct',
                        choices=list(LOSS.keys()),
                        help='Choose loss function. Default cross_entropy_loss',
                        default='cross_entropy_loss')
    parser.add_argument('--data-label', dest='data_label', help='data label')
    parser.add_argument('--target-label', dest='target_label', help='target label')

    parser.add_argument('--lr', type=float, required=False, dest='lr', default=0.01)
    parser.add_argument('--scheduler', required=False, dest='scheduler', default='exp')

    parser.add_argument('--end-batch-train', dest='end_batch_train', help='number of the last batch in dataset slice')
    parser.add_argument('--end-batch-val', dest='end_batch_val', help='number of the last batch in dataset slice')

    parser.add_argument('--verbose', action='store_true', default=False, dest='verbose',
                        help='verbose output [default: False].')
    parser.add_argument('-l', '--logging-file', dest='logging_filename',
                        help='Path to output logging text file [default: output to stdout only].')
    parser.add_argument('-tl', '--tboard-json-logging-file', dest='tboard_json_logging_file',
                        help='Path to output logging JSON file with scalars [default: none].')
    parser.add_argument('-x', '--tboard-dir', dest='tboard_dir',
                        help='Path to tensorboard [default: do not log events].')
    parser.add_argument('-w', '--overwrite', action='store_true', default=False,
                        help='If set, overwrite existing logs [default: exit if output dir exists].')

    # parser.add_argument('--data-type', required=True, dest='dataloader_type',
    #                    help='type of the train/val data to use.', choices=dataloading.prepare_loaders.keys())
    # parser.add_argument('--handcrafted-train', required=False, action='append',
    #                    dest='handcrafted_train_paths', help='dirnames of handcrafted datasets used for training '
    #                                                         '(sought for in preprocessed/synthetic_handcrafted).')
    # parser.add_argument('--handcrafted-val', required=False, action='append',
    #                    dest='handcrafted_val_paths', help='dirnames of handcrafted datasets used for validation '
    #                                                       '(sought for in preprocessed/synthetic_handcrafted).')
    # parser.add_argument('--handcrafted-val-part', required=False, type=float, default=.1,
    #                    dest='handcrafted_val_part', help='portion of handcrafted_train used for validation')
    # parser.add_argument('-M', '--memory-constraint', required=True, type=int, dest='memory_constraint',help='maximum RAM usage in bytes.')
    # parser.add_argument('-r', '--render-resolution', dest='render_res', default=64, type=int,
    #                    help='resolution used for rendering.')
    # parser.add_argument('--l2_weight_change', required=False, dest='l2_weight_change',
    #                    help='Weight change for L2. l2_weight = l2_weight + l2_weight_change(default -1e-5)',
    #                    type=float, default=-1e-5)

    # parser.add_argument('--l2_weight_init', required=False, dest='l2_weight_init',
    #                    help='l2_weight initialization(default 1)',
    #                    type=float, default= 1.)

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
