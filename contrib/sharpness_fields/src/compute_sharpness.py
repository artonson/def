import os
import numpy as np
import torch
import argparse
from dataset_sharp import PointCloudDataset
from glob import glob
from tqdm import tqdm


def run_on_one_file(input_file, output_file, model_path, r_factor):

    np.warnings.filterwarnings('ignore')
    dataset = PointCloudDataset([input_file], training=False, r_factor_range=(r_factor, r_factor))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

    model = torch.jit.load(model_path)

    result = []
    for batch_idx, data_values in tqdm(enumerate(dataloader), total=len(dataloader)):

        data_values = [val.squeeze().cuda() for val in data_values[:-1]]
        pred = model(*data_values)
        result.append(pred.cpu().detach().squeeze().numpy())

    result = np.concatenate(result)
    np.savetxt(output_file, result)

def run(input_dir, output_dir, model_path, r_factor):
    input_files = glob(input_dir)

    for input_file in tqdm(input_files, total=len(input_files)):
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.txt')
        run_on_one_file(input_file, output_file, model_path, r_factor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', nargs='?', help='Input XYZ file containing points and normals', 
                        default='blade_mls.xyz')
    parser.add_argument('output_dir', nargs='?', help='Output text file containing sharpness field values', 
                        default='sharpness_values.txt')
    parser.add_argument('-m', '--model_path', help='Path of pretrained model',
                        default='sharpness_model.pt')
    parser.add_argument('-r', '--r_factor', type=float, help='Neighborhood radius factor',
                         default=5.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run(args.input_dir, args.output_dir, args.model_path, args.r_factor)

