import os
import torch
import torch.nn
import h5py
import numpy as np
import argparse
import glob
import json
# from tqdm import tqdm


def main(options):
    filenames = glob.glob(os.path.join(options.true_path, '*.hdf5'))
    mse_loss = []
    sl1_loss = []
    for true_pathname in filenames:

        true_filename = os.path.basename(true_pathname)
        print('=== Reading file %s ===' % true_filename)

        with h5py.File(true_pathname, 'r') as f:
            true = torch.from_numpy(f[options.target_label][:])

        pred_pathname = os.path.join(options.pred_path, true_filename)
        with h5py.File(pred_pathname, 'r') as f:
            pred = torch.from_numpy(f[options.target_label][:])

        mse_loss_function = torch.nn.SmoothL1Loss(reduction='none')
        sl1_loss_function = torch.nn.MSELoss(reduction='none')

        mse_loss.append(mse_loss_function(true, pred).numpy())
        sl1_loss.append(sl1_loss_function(true, pred).numpy())

    mse_loss = np.concatenate(mse_loss)
    sl1_loss = np.concatenate(sl1_loss)

    print('=== Creating dictionary ===')

    metric_dict = {'mse_per_point': mse_loss.tolist(),
                   'sl1_per_point': sl1_loss.tolist(),
                   'mse_per_patch': mse_loss.mean(-1).tolist(),
                   'sl1_per_patch': sl1_loss.mean(-1).tolist(),
                   'mse_per_dataset': mse_loss.mean(-1).mean().tolist(),
                   'sl1_per_dataset': sl1_loss.mean(-1).mean().tolist()}

    print('=== Dumping to JSON ===')

    with open(options.save_path + "metrics.json", "w") as write_file:
        json.dump(metric_dict, write_file)

    print('=== Finish ===')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--true-path', default='', help='Path to GT')
    parser.add_argument('--pred-path', default='', help='Path to prediction')

    parser.add_argument('--target-label', dest='target_label', help='Target label')

    parser.add_argument('--save-path', default='', help='Path to save to')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
