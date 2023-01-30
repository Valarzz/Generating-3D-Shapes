import torch
import torch.nn as nn
import param


class gen(nn.Module):  # in: [z_dim] --> out: []
    def __init__(self):
        super(gen, self).__init__()
        self.cube_len = param.cube_len
        self.bias = param.bias
        self.z_dim = param.z_dim
        self.f_dim = 32

        self.layer1 = self.conv_layer(self.z_dim    , self.f_dim * 8, bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim * 8, self.f_dim * 4, bias=self.bias)
        self.layer3 = self.conv_layer(self.f_dim * 4, self.f_dim * 2, bias=self.bias)
        self.layer4 = self.conv_layer(self.f_dim * 2, self.f_dim    , bias=self.bias)

        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.f_dim, 1, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
            # torch.nn.Tanh()
        )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1, 1, 1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias,
                                     padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.ReLU(True)
            # torch.nn.LeakyReLU(self.leak_value, True)
        )
        return layer

    def forward(self, x):
        out = x.view(-1, self.z_dim, 1, 1, 1)
        # print(out.size())  # torch.Size([32, 200, 1, 1, 1])
        out = self.layer1(out)
        # print(out.size())  # torch.Size([32, 256, 2, 2, 2])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([32, 128, 4, 4, 4])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([32, 64, 8, 8, 8])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([32, 32, 16, 16, 16])
        out = self.layer5(out)
        # print(out.size())  # torch.Size([32, 1, 32, 32, 32])
        out = torch.squeeze(out)
        return out


class dis(nn.Module):  # in:[32, 32, 32];  out:1/0
    def __init__(self):
        super(dis, self).__init__()
        self.cube_len = param.cube_len
        self.leak_value = param.leak_value
        self.bias = param.bias

        padd = (0, 0, 0)
        if self.cube_len == 32:
            padd = (1, 1, 1)

        self.f_dim = 32

        self.layer1 = self.conv_layer(1, self.f_dim, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim, self.f_dim * 2, kernel_size=4, stride=2, padding=(1, 1, 1),
                                      bias=self.bias)
        self.layer3 = self.conv_layer(self.f_dim * 2, self.f_dim * 4, kernel_size=4, stride=2, padding=(1, 1, 1),
                                      bias=self.bias)
        self.layer4 = self.conv_layer(self.f_dim * 4, self.f_dim * 8, kernel_size=4, stride=2, padding=(1, 1, 1),
                                      bias=self.bias)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(self.f_dim * 8, 1, kernel_size=4, stride=2, bias=self.bias, padding=padd),
            # torch.nn.Sigmoid()
        )

        # self.layer5 = torch.nn.Sequential(
        #     torch.nn.Linear(256*2*2*2, 1),
        #     torch.nn.Sigmoid()
        # )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1, 1, 1), bias=False):
        layer = torch.nn.Sequential(
            torch.nn.Conv3d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            torch.nn.BatchNorm3d(output_dim),
            torch.nn.LeakyReLU(self.leak_value, inplace=True)
        )
        return layer

    def forward(self, x):
        # out = torch.unsqueeze(x, dim=1)
        out = x.view(-1, 1, self.cube_len, self.cube_len, self.cube_len)
        # print(out.size())  # torch.Size([32, 1, 32, 32, 32])
        out = self.layer1(out)
        # print(out.size())  # torch.Size([32, 32, 16, 16, 16])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([32, 64, 8, 8, 8])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([32, 128, 4, 4, 4])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([32, 256, 2, 2, 2])
        out = self.layer5(out)
        # print(out.size(), out)  # torch.Size([32, 1, 1, 1, 1])
        out = torch.squeeze(out)
        return out



class latentG(nn.Module):  # in: [z_dim] --> out: []
    def __init__(self):
        super(latentG, self).__init__()
        self.cube_len = param.cube_len
        self.bias = param.bias
        self.z_dim = param.z_dim
        self.f_dim = 200

        self.layer1 = self.fc_layer(self.z_dim    , self.f_dim * 8, bias=self.bias)
        self.layer2 = self.fc_layer(self.f_dim * 8, self.f_dim * 4, bias=self.bias)
        self.layer3 = self.fc_layer(self.f_dim * 4, self.f_dim * 2, bias=self.bias)
        self.layer4 = self.fc_layer(self.f_dim * 2, self.f_dim    , bias=self.bias)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(self.f_dim, self.z_dim, bias=self.bias),
            torch.nn.Sigmoid()
            # torch.nn.Tanh()
        )

    def fc_layer(self, input_dim, output_dim, bias=False):
        layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim, bias=bias),
            torch.nn.ReLU(True)
            # torch.nn.LeakyReLU(self.leak_value, True)
        )
        return layer

    def forward(self, x):
        # out = x.view(-1, self.z_dim, 1, 1, 1)
        out = x
        # print(out.size())  # torch.Size([32, 200, 1, 1, 1])
        out = self.layer1(out)
        # print(out.size())  # torch.Size([32, 256, 2, 2, 2])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([32, 128, 4, 4, 4])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([32, 64, 8, 8, 8])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([32, 32, 16, 16, 16])
        out = self.layer5(out)
        # print(out.size())  # torch.Size([32, 1, 32, 32, 32])
        out = torch.squeeze(out)
        return out


class latentD(nn.Module):  # in:[32, 32, 32];  out:1/0
    def __init__(self):
        super(latentD, self).__init__()
        self.cube_len = param.cube_len
        self.leak_value = param.leak_value
        self.bias = param.bias
        self.f_dim = 200

        self.layer1 = self.fc_layer(param.z_dim, self.f_dim, bias=self.bias)
        self.layer2 = self.fc_layer(self.f_dim, self.f_dim * 2, bias=self.bias)
        self.layer3 = self.fc_layer(self.f_dim * 2, self.f_dim * 4, bias=self.bias)
        self.layer4 = self.fc_layer(self.f_dim * 4, self.f_dim * 8, bias=self.bias)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(self.f_dim * 8, 1, bias=self.bias),
            # torch.nn.Sigmoid()
        )

    def fc_layer(self, input_dim, output_dim, bias=False):
        layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim, bias=bias),
            torch.nn.LeakyReLU(self.leak_value, inplace=True)
        )
        return layer

    def forward(self, x):
        # out = torch.unsqueeze(x, dim=1)
        # out = x.view(-1, 1, self.cube_len, self.cube_len, self.cube_len)
        out = x
        # print(out.size())  # torch.Size([32, 1, 32, 32, 32])
        out = self.layer1(out)
        # print(out.size())  # torch.Size([32, 32, 16, 16, 16])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([32, 64, 8, 8, 8])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([32, 128, 4, 4, 4])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([32, 256, 2, 2, 2])
        out = self.layer5(out)
        # print(out.size(), out)  # torch.Size([32, 1, 1, 1, 1])
        out = torch.squeeze(out)
        return out

