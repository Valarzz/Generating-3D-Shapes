import torch

# training
epochs = 200
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TrainNew = True
load = True
modelPath = "checkpoint/checkpoint_027601_lossD_{lossD}_lossG_{lossG}" + ".pth"
# device = torch.device('cpu')

# noise generate
z_dim = 200
z_dis = "norm"
z_mean = 0.
z_std = 0.33

# G
g_lr = 5e-6
weightClip = 0.01
n_critic = 5

# D
d_lr = 1e-6
leak_value = 0.2

soft_label = False
adv_weight = 0
d_thresh = 0.8
beta = (0.5, 0.999)

# data & network
cube_len = 32
bias = False

# output
model_save_step = 1
data_dir = '../volumetric_data/'
model_dir = 'chair/'  # change it to train on other data models
output_dir = '../outputs'
# images_dir = '../test_outputs'


def print_params():
    l = 16
    print(l * '*' + 'hyper-parameters' + l * '*')

    print('epochs =', epochs)
    print('batch_size =', batch_size)
    print('soft_labels =', soft_label)
    print('adv_weight =', adv_weight)
    print('d_thresh =', d_thresh)
    print('z_dim =', z_dim)
    print('z_dis =', z_dis)
    print('model_images_save_step =', model_save_step)
    print('data =', model_dir)
    print('device =', device)
    print('g_lr =', g_lr)
    print('d_lr =', d_lr)
    print('cube_len =', cube_len)
    print('leak_value =', leak_value)
    print('bias =', bias)

    print(l * '*' + 'hyper-parameters' + l * '*')
