import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
import h5py
import os 
from torch_geometric.data import Data

class ABCDataset(InMemoryDataset):
    def __init__(self, root, split, transform=None, pre_transform=None):
        self.split = split
        super(ABCDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset_dgcnn'+ str(self.split) + '.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        key = [1, 2]
        for k in key:
            split_root = self.root + '/' + self.split + '/' + str(k)
            for f in os.listdir(split_root):
                split_file = split_root + '/' + f
                dataset = h5py.File(split_file, 'r')
                for x, y in zip(dataset['data'], dataset['label']):
                    data_list.append(Data(x=torch.FloatTensor(x), y=torch.FloatTensor(y)))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


#class ABCDataset():
#    def __init__(self, root, transform=None):
#        self.root = root
#        self.transform = transform
#        
#    def get_data(self, split, key=[1, 2]):
#        data = []
#        
#        for k in key:
#             split_root = self.root + split + '/' + str(k)
#             for f in os.listdir(split_root):
#                 split_file = split_root + '/' + f
#                 dataset = h5py.File(split_file, 'r')
#                 for x, y in zip(dataset['data'], dataset['label']):
#                     data.append(Data(x=torch.FloatTensor(x), y=torch.LongTensor(y)))
                            
#         return data

if __name__ == '__main__':
    ds = ABCDataset('/home/gbobrovskih/DGCNN_geompytorch/data/')
    data = ds.get_data('train')
    ds_loader = DataLoader(data, batch_size=32)
    for d in ds_loader:
        print(d)
        print(d.batch, d.x)
        break
        
