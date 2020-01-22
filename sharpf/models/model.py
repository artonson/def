import torch
import torch.nn as nn
from torch.nn import Sequential, Conv1d, Linear, ReLU, LeakyReLU, ModuleList, BatchNorm1d, Dropout
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import knn_graph

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Sequential(Linear(2 * in_channels, out_channels),
                       BatchNorm1d(out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels),
                       BatchNorm1d(out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        return aggr_out

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False)
        return super(DynamicEdgeConv, self).forward(x, edge_index)

class DGCNN(torch.nn.Module):
    def __init__(self, num_classes=40, k=20):
        super(DGCNN, self).__init__()
        self.num_classes = num_classes
        self.k = k
        self.conv1 = DynamicEdgeConv(in_channels=3, out_channels=64, k=self.k)
        self.conv2 = DynamicEdgeConv(in_channels=64, out_channels=64, k=self.k) 
        self.conv3 = DynamicEdgeConv(in_channels=64, out_channels=64, k=self.k)
        self.conv4 = DynamicEdgeConv(in_channels=64, out_channels=128, k=self.k)
#        self.conv5 = DynamicEdgeConv(in_channels=(64 + 64 + 64 + 128), out_channels=1024, k=self.k)
        self.conv5 = Sequential(Linear(64 + 64 + 64 + 128, 1024),
                                LeakyReLU(),
                                BatchNorm1d(1024))
        self.mlp_list = ModuleList([
                         Linear(1024, 512),
                         LeakyReLU(),
                         BatchNorm1d(512),
                         Dropout(p=0.5),
                         Linear(512, 256),
                         LeakyReLU(),
                         BatchNorm1d(256),
                         Dropout(p=0.5),
                         Linear(256, self.num_classes)
                        ])

    def forward(self, sample):
        x, batch = sample.x, sample.batch
        x1 = self.conv1(x, batch=batch)
        x2 = self.conv2(x1, batch=batch)
        x3 = self.conv3(x2, batch=batch)
        x4 = self.conv4(x3, batch=batch)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_mlp = self.conv5(x_cat)#.max(dim=-1, keepdim=False)[0]
        x_out = [x_mlp]
        for layer in self.mlp_list:
            x_out.append(layer(x_out[-1]))
        return x_out[-1]
    
if __name__ == '__main__':
    num_classes = 2
    model = DGCNN(num_classes)
