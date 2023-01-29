import torch
import numpy as np
from ae import aeNet, aeDataset
from torch.utils.data import DataLoader
import param
import time
import os
from torch.utils.tensorboard import SummaryWriter


files = os.listdir("airplaneData")
allfile = []
for id, file in enumerate(files):
    datapath = os.path.join("airplaneData", file)
    allfile.append(datapath)
x = np.array(allfile)
train_set, test_set = np.split(x, [int(len(x)*0.8)])  #60%训练集、30%测试集、10%验证集

trainset = aeDataset(train_set)
trainloader = DataLoader(trainset, batch_size=param.batch_size, shuffle=True)
testset = aeDataset(test_set)
testloader = DataLoader(testset, batch_size=param.batch_size, shuffle=True)
net = aeNet().to(param.device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6, betas=param.beta)


def loss(gt, fake):
    return torch.mean((gt - fake) ** 2)


def saveM(net, opt, PATH):
    torch.save({
        'G_state_dict': net.en.state_dict(),
        'D_state_dict': net.de.state_dict(),
        'D_opt_state_dict': opt.state_dict(),
    }, PATH)


writer = SummaryWriter('tensorboard_aesave')
print("start training!")
for epoch in range(param.epochs):
    epoch_start_time = time.time()
    train_loss = 0.0
    val_loss = 0.0

    net.train()
    for i, data in enumerate(trainloader):
        data = data.to(param.device)
        optimizer.zero_grad()
        out = net(data)
        batch_loss = loss(out, data)
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss.item()

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(trainloader):
            data = data.to(param.device)
            optimizer.zero_grad()
            out = net(data)
            batch_loss = loss(out, data)

            val_loss += batch_loss.item()

    writer.add_scalars('Loss', {'Train': train_loss / len(trainset)}, epoch)
    writer.add_scalars('Loss', {'Test': val_loss / len(testset)}, epoch)

    if epoch % 10 == 0:
        saveM(net, optimizer, "AE_M/"+str(epoch).zfill(3)+".pth")
    print('[%03d/%03d] %2.2f sec(s) Train Loss: %3.6f | Val loss: %3.6f' % \
          (epoch + 1, param.epochs, time.time() - epoch_start_time, train_loss / len(trainset), val_loss / len(testset)))























