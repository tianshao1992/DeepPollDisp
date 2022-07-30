import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import h5py

def cal_fields_error(true, pred):
    y_error = torch.abs(true - pred)

    L1_error = torch.mean(y_error, dim=(2, 3))
    L2_error = torch.mean(y_error ** 2, dim=(2, 3))
    Li_error = y_error.max(dim=2)[0].max(dim=2)[0]

    L_error = np.hstack((L1_error.cpu().numpy(), L2_error.cpu().numpy(), Li_error.cpu().numpy()))
    return L_error

class generator(nn.Module):
    def __init__(self, nz, num_layers=5, num_filters=32, num_convs=3, outputs=(1, 41, 4), activate=nn.GELU(),
                 mode="train"):
        super(generator, self).__init__()
        self.x0_shape = [num_filters, ]
        self.filters = num_filters
        self.convs = num_convs
        self.outputs = outputs
        self.activate = activate
        self.nz = nz
        self.num_layers = num_layers

        self.outputs = outputs
        outputs_ = (outputs[0], 2**int(math.log2(outputs[1])),  2**int(math.log2(outputs[2])))
        self.x0_shape.append(int(outputs_[1] / 2 ** (num_layers - 1)))
        self.x0_shape.append(int(outputs_[2] / 2 ** (num_layers - 1)))

        self.linear = nn.Linear(self.nz,int(np.prod(self.x0_shape)))
        self.conv = nn.Conv2d(self.filters, self.outputs[0], kernel_size=3, padding=1, stride=1)

        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(self._make_layer())
        self.layers = nn.Sequential(*self.layers)

        if mode == "train":
            self.apply(initialize_weights)
        else:
            for p in self.parameters():
                p.requires_grad = False


    def _make_layer(self):
        layers = []
        for id in range(self.convs):
            conv = nn.Conv2d(self.filters, self.filters, kernel_size=3, padding=1, stride=1)
            layers.append(nn.Sequential(conv, self.activate))
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x0 = x.reshape(x.size(0), self.x0_shape[0], self.x0_shape[1], self.x0_shape[2])
        for i in range(self.num_layers-1):
            x = self.layers[i](x0)
            x += x0
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x0 = x

        x = self.layers[-1](x)
        x_ = self.conv(x)
        x = F.interpolate(x_, size=self.outputs[1:], mode='bilinear', align_corners=False)
        return x # x_



def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            # m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data)
            # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.zero_()


if __name__ == '__main__':
    ge = generator(nz=4, num_layers=8, num_filters=32, num_convs=4, outputs=(3, 512, 512)).cuda()

    x = torch.rand(16, 4).cuda()
    y = ge(x)
    print(y.shape)

    # dc = discrimitor(nz=2, inputs=(6, 64, 256))
    # z = dc(y)
    # print(z.shape)

