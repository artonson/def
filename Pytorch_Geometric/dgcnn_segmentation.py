import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import DynamicEdgeConv
# from torch_geometric.utils import intersection_and_union as i_and_u
from torch_scatter import scatter_add
from pointnet2_classification import MLP
from SharpfData import Sharpf

from datetime import date
import copy

transform = T.Compose([
    T.RandomTranslate(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2)
])
# pre_transform = T.NormalizeScale()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'abc')

train_dataset = Sharpf(path, train=True, transform=transform)
test_dataset = Sharpf(path, train=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                         num_workers=6)


def intersection_and_union(pred, target, num_classes, batch=None):
    r"""Computes intersection and union of predictions.
    Args:
        pred (LongTensor): The predictions.
        target (LongTensor): The targets.
        num_classes (int): The number of classes.
        batch (LongTensor): The assignment vector which maps each pred-target
            pair to an example.
    :rtype: (:class:`LongTensor`, :class:`LongTensor`)
    """
    pred, target = F.one_hot(pred, num_classes), F.one_hot(target, num_classes)

    if batch is None:
        i = (pred & target).sum(dim=0)
        u = (pred | target).sum(dim=0)
    else:
        i = scatter_add(pred & target, batch, dim=0)
        u = scatter_add(pred | target, batch, dim=0)

    return i, u

class Net(torch.nn.Module):
    def __init__(self, out_channels, k=30, aggr='max'):
        super(Net, self).__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.lin1 = MLP([3 * 64, 1024])
        # self.lin1 = MLP([2 * 64, 1024])

        self.mlp = Seq(
            MLP([1024, 256]), Dropout(0.5), MLP([256, 128]), Dropout(0.5),
            Lin(128, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        # out = self.lin1(torch.cat([x1, x2], dim=1))
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# print('num_classes: {}'.format(train_dataset.num_classes))
model = Net(train_dataset.num_classes, k=10).to(device)  # default k=30
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)


def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.max(dim=1)[1].eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print('[{}/{}] Loss: {:.4f}, Train Accuracy: {:.4f}'.format(
                i + 1, len(train_loader), total_loss / 10,
                correct_nodes / total_nodes))
            total_loss = correct_nodes = total_nodes = 0


def test(loader):
    model.eval()

    correct_nodes = total_nodes = 0
    # intersections, unions, categories = [], [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pred = out.max(dim=1)[1]
        correct_nodes += pred.eq(data.y).sum().item()
        total_nodes += data.num_nodes
        # i, u = i_and_u(pred, data.y, test_dataset.num_classes, data.batch)
        # i, u = intersection_and_union(pred, data.y, test_dataset.num_classes, data.batch)
        # intersections.append(i.to(torch.device('cpu')))
        # unions.append(u.to(torch.device('cpu')))
        # categories.append(data.category.to(torch.device('cpu')))

    # category = torch.cat(categories, dim=0)
    # intersection = torch.cat(intersections, dim=0)
    # union = torch.cat(unions, dim=0)

    # ious = [[] for _ in range(loader.dataset.categories)]
    # for j in range(len(loader.dataset)):
    #     i = intersection[j, loader.dataset.y_mask[category[j]]]
    #     u = union[j, loader.dataset.y_mask[category[j]]]
    #     iou = i.to(torch.float) / u.to(torch.float)
    #     iou[torch.isnan(iou)] = 1
    #     ious[category[j]].append(iou.mean().item())

    # for cat in range(len(loader.dataset.categories)):
    #     ious[cat] = torch.tensor(ious[cat]).mean().item()

    return correct_nodes / total_nodes #, torch.tensor(ious).mean().item()

if __name__ == '__main__':

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, 10):
        train()
        epoch_acc = test(test_loader)
        # print('Epoch: {:02d}, Acc: {:.4f}, IoU: {:.4f}'.format(epoch, acc, iou))
        print('Epoch: {:02d}, Acc: {:.4f}'.format(epoch, epoch_acc))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    model.cpu()
    torch.save(model.state_dict(), 'checkpoints/dgcnn-k10-sharpf-%s.pth' %date.today())


