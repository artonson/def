import torch
import torch.nn.functional as F
from sharpf.models.model import DGCNN
from sharpf.data.datasets import ABCDataset
from torch_geometric.data import DataLoader
import datetime
from tensorboardX import SummaryWriter

def exp_lr_scheduler(optimizer, global_step, init_lr, decay_steps, decay_rate, lr_clip, staircase=True):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if staircase:
        lr = init_lr * decay_rate**(global_step // decay_steps)
    else:
        lr = init_lr * decay_rate**(global_step / decay_steps)
    if (global_step // decay_steps) >= 1.0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        print(param_group['lr'])
        param_group['lr'] = lr

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #ds = ABCDataset('/home/gbobrovskih/DGCNN_geompytorch/data/')
    #dataset = ds.get_data('train')
    batch_size = 32
    dataset = ABCDataset(root='/home/gbobrovskih/DGCNN_geompytorch/data', split='train')
    train_loader = DataLoader(dataset, batch_size=32)
    batches_n = len(train_loader)
    model_dgcnn = DGCNN(num_classes=1, k=20) 
    lr = 0.01
    optimizer = torch.optim.Adam(model_dgcnn.parameters(), lr=lr, weight_decay=1e-4)#momentum=0.9, weight_decay=1e-4)
    model_dgcnn.to(device)
    model_dgcnn.train()
    tag = 'dgcnn_batch=32_adam'
    loss_tag = 'bce_loss'
    loss = torch.nn.BCEWithLogitsLoss()
    start = 0
    batch = 0
    if start >= 0:
       path = '/home/gbobrovskih/DGCNN_geompytorch/model/{}/checkpoint_{}'.format(tag, start)
       checkpoint = torch.load(path)
       model_dgcnn.load_state_dict(checkpoint['model_state_dict'])
       batch = start * batches_n
    tb_writer = SummaryWriter('/home/gbobrovskih/data/.logs/{}/{}/{}'.format(tag, loss_tag, datetime.datetime.today()))
    
    epochs = 15
    decay_steps=39100
    for epoch in range(1+start, epochs):
        loss_avg = 0
        for batch_idx, sample_idx in enumerate(train_loader):
            batch += 1
            print('%.1f '%(batch * batch_size / decay_steps * 100) + '% / 100% till loss change...')
            sample_idx = sample_idx.to(device)
            optimizer.zero_grad()
            output = model_dgcnn(sample_idx)
            train_loss = loss(output, sample_idx.y.float())
            train_loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print('current lr={}'.format(current_lr))
            exp_lr_scheduler(optimizer=optimizer, global_step=batch*batch_size, init_lr=lr, decay_steps=decay_steps, decay_rate=0.1, lr_clip=0.00001, staircase=True)
            optimizer.step()
            if batch_idx % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx*len(sample_idx), len(train_loader.dataset),
                        100. * batch_idx / batches_n, train_loss.item()))
            iteration = epoch * batches_n + batch_idx
            tb_writer.add_scalar('Train_loss', train_loss.item(), iteration)
            loss_avg += train_loss.item()
            tb_writer.add_scalar('Train_loss_Loss_avg', loss_avg/(batch_idx+1), iteration)
        torch.save({'epoch': epoch,
            'model_state_dict': model_dgcnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, '/home/gbobrovskih/DGCNN_geompytorch/model/{}/checkpoint_{}'.format(tag, epoch))
