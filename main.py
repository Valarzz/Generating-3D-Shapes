import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from GAN import data3d, trainer
from GAN3Dnets import gen, dis
import param
import numpy as np
import random
import shutil
import os
import time


def prepareOut(p):
    if param.TrainNew == True:
        if os.path.exists(p):
            shutil.rmtree(p)
        os.mkdir(p)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    prepareOut("output")  # for airplane
    prepareOut("checkpoint")  # for weights
    prepareOut("tensorboard_save")  # for loss

    trainset = data3d("airplaneData")
    trainloader = DataLoader(trainset, batch_size=param.batch_size, shuffle=True)
    T = trainer(trainloader)

    setup_seed(20)
    print("start training!")
    start = time.time()
    T.training()
    end = time.time() - start
    print(f"Training time: {end}")

