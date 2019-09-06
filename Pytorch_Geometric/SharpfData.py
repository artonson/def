import os
import os.path as osp
import glob
import h5py
import numpy as np

import torch
from torch_geometric.data import Data, InMemoryDataset


"""
Note:
    Sharp features from ABD dataset. The data are structured as HDF5 format which consist of

    ['data'] -> patches of point cloud with shape [num_patches, num_points, 3] where num_points = 1,024
    ['data_noisy'] -> Noise is added to each patches of point cloud
    ['org_data'] -> original patches
    ['label'] -> label of each point whether it is a point on sharp feature with shape [num_patches, num_points, 1]
                  label are 0 or 1
"""

class Sharpf(InMemoryDataset):
    def __init__(self,
                root,
                train=True, 
                transform=None, 
                pre_transform=None,
                pre_filter=None):
        super(Sharpf, self).__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
    
    @property
    def raw_file_names(self):
        return ['abc_05_sharp_1024_patches_normalized_4pps_selected_short_curves_surfaces_train.hdf5',
                'abc_05_sharp_1024_patches_normalized_4pps_selected_short_curves_surfaces_val.hdf5',
                'abc_05_sharp_1024_patches_normalized_4pps_selected_short_curves_surfaces_test.hdf5']
    
    @property
    def processed_file_names(self):
        return ['abc_sharpf_train.pt', 'abc_sharpf_test.pt']

    def _download(self):
        pass
    
    def process(self):
        train_data_list = self.process_raw_path(self.raw_paths[0])
        test_data_list = self.process_raw_path(self.raw_paths[2])

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])
    
    def process_raw_path(self, file_path):
        data_list = []
        # seg_list = []

        pcs, ys = self.load_h5(file_path, ltype=torch.long)
        ys = [y for y in ys]
        lens = [y.size(0) for y in ys]

        y = torch.cat(ys).unique(return_inverse=True)[1]
        # seg_list.append(y.unique())
        ys = y.split(lens)

        for (pos, y) in zip(pcs, ys):
            data = Data(y=y, pos=pos)
            data_list.append(data)

        return data_list

    def load_h5(self, filename, dtype=None, ltype=None):
        """
        RETURN:
        data = tensor [num_patches, num_points, 3]
        label = tensor [num_patches, num_points]
        """
    
        f = h5py.File(filename)
        data = f['data'][:] # return array
        label = f['label'][:] #return array

        assert data.shape == (len(data), 1024, 3), 'Data shape is not (num_patches, 1024,3)'
        assert label.shape == (len(label), 1024, 1), 'Label shape is not (num_patches, 1024,1)'
        
        data = torch.from_numpy(data).squeeze().float()
        print(type(data))
        # label = label.astype('float64')
        label = torch.from_numpy(label).squeeze().long()
        # label = torch.tensor(label, dtype=torch.long)
        print(type(label))
        return (data, label)

        