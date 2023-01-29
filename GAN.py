import torch
import numpy as np
from GAN3Dnets import gen, dis
import param
from torch.utils.data import Dataset
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision


class data3d(Dataset):
    def __init__(self, path):
        files = os.listdir(path)
        allfile = []
        for id, file in enumerate(files):
            datapath = os.path.join(path, file)
            allfile.append(datapath)
        self.files = allfile

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        v = torch.from_numpy(np.load(self.files[index]))
        return v.float()


class trainer():
    def __init__(self, dataload):
        self.dataloader = dataload
        self.D = dis().float().to(param.device)
        self.G = gen().float().to(param.device)
        self.g_opt = torch.optim.RMSprop(self.G.parameters(), lr=param.g_lr)
        self.d_opt = torch.optim.RMSprop(self.D.parameters(), lr=param.d_lr)
        self.writer = SummaryWriter('tensorboard_save')
        self.step = 0
        self.epoch = 0
        if param.load == True:
            self.loadM(param.modelPath)

    def generateZ(self, batch):
        # batch = param.batch_size
        if param.z_dis == "norm":
            Z = torch.Tensor(batch, param.z_dim).normal_(param.z_mean, param.z_std).to(param.device)
        elif param.z_dis == "uni":
            Z = torch.randn(batch, param.z_dim).to(param.device)
        else:
            Z = torch.Tensor(batch, param.z_dim).normal_(param.z_mean, param.z_std).to(param.device)
            print("z_dist is not normal or uniform, use normal as default")
        return Z

    def train_dis(self, truedata):
        z = self.generateZ(param.batch_size)
        z_gen = self.G(z)
        real = self.D(truedata)
        fake = self.D(z_gen)
        lossD = -(torch.mean(real) - torch.mean(fake))
        self.d_opt.zero_grad()
        lossD.backward(retain_graph=True)
        self.d_opt.step()

        # clip critic weights between -c, c
        for p in self.D.parameters():
            p.data.clamp_(-param.weightClip, param.weightClip)
        return lossD, torch.mean(real), torch.mean(fake)

    def train_gen(self):
        z = self.generateZ(param.batch_size)
        z_gen = self.G(z)
        lossG = -torch.mean(self.D(z_gen))
        self.g_opt.zero_grad()
        lossG.backward()
        self.g_opt.step()
        return lossG

    def training(self):
        self.G.train()
        self.D.train()
        # torch.autograd.set_detect_anomaly(True)
        while True:
            if self.epoch > param.epochs:
                break
            self.epoch += 1

            for i, data in enumerate(self.dataloader):
                lossD, tl, fl = self.train_dis(data.to(param.device))
                if i % param.n_critic:
                    lossG = self.train_gen()

                self.step += 1
                if self.step % 200 == 0:
                    self.record(self.epoch, i, lossD, lossG, tl, fl)
                # if self.step % 1000 == 0:
                    self.saveM(self.epoch, f"checkpoint/checkpoint_"+str(self.step).zfill(6)+"_lossD_{lossD}_lossG_{lossG}.pth")

    def record(self, epoch, batch, ld, lg, tl, fl):
        # Print losses occasionally and print to tensorboard
        print(
            f"Epoch [{epoch}/{param.epochs}] Batch {batch}/{len(self.dataloader)} \
              Loss D: {ld:.8f}, loss G: {lg:.8f}, True: {tl}, Fake:{fl}"
        )
        self.writer.add_scalar("D", ld, global_step=self.step)
        self.writer.add_scalar("G", lg, global_step=self.step)
        self.output3d(f"output/output_"+str(self.step).zfill(6)+"_lossD_{ld}_lossG_{lg}.npy")
        self.step += 1

    def output3d(self, path):
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            fake = self.G(self.generateZ(1))
            np.save(path, fake.cpu().numpy())
        #     # take out (up to) 32 examples
        #     img_grid_real = torchvision.utils.make_grid(
        #         data[:32], normalize=True
        #     )
        #     img_grid_fake = torchvision.utils.make_grid(
        #         fake[:32], normalize=True
        #     )
        #     self.writer.add_image("Real", img_grid_real, global_step=self.step)
        #     self.writer.add_image("Fake", img_grid_fake, global_step=self.step)
        self.G.train()
        self.D.train()

    def saveM(self, epoch, PATH):
        torch.save({
            'epoch': epoch,
            'G_state_dict': self.G.state_dict(),
            'G_opt_state_dict': self.g_opt.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'D_opt_state_dict': self.d_opt.state_dict(),
        }, PATH)

    def loadM(self, path):
        checkpoint = torch.load(path)
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.g_opt.load_state_dict(checkpoint['G_opt_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.d_opt.load_state_dict(checkpoint['D_opt_state_dict'])
        self.epoch = checkpoint['epoch']



