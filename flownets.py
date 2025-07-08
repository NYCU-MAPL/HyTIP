import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util.functional as FE
from util.sampler import shift, warp


class SPyBlock(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(8, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 2, kernel_size=7, padding=3)
        )

    def forward(self, flow_course, im1, im2):
        flow = FE.interpolate_as(flow_course, im2, mode='bilinear') * 2.0
        res = super().forward(torch.cat([im1, warp(im2, flow), flow], dim=1))
        flow_fine = res + flow

        return flow_fine


class SPyNet(nn.Module):
    """SPyNet"""

    def __init__(self, level=4, path=None, trainable=False):
        super(SPyNet, self).__init__()
        self.level = level
        self.Blocks = nn.ModuleList([SPyBlock() for _ in range(level+1)])

        if path is not None:
            data = torch.load(path, map_location='cpu')
            if 'state_dict' in data.keys():
                self.load_state_dict(data['state_dict'])
            else:
                self.load_state_dict(data)

        self.register_buffer('mean', torch.Tensor(
            [.406, .456, .485]).view(-1, 1, 1))
        self.register_buffer('std', torch.Tensor(
            [.225, .224, .229]).view(-1, 1, 1))

        if not trainable:
            self.requires_grad_(False)

    def norm(self, input):
        return input.sub(self.mean).div(self.std)

    def forward(self, im2, im1):  # backwarp
        # B, 6, H, W
        volume = [torch.cat([self.norm(im1), self.norm(im2)], dim=1)]
        for _ in range(self.level):
            volume.append(F.avg_pool2d(volume[-1], kernel_size=2))

        flows = [torch.zeros_like(volume[-1][:, :2])]  # B, 2, H//16, W//16
        for l, layer in enumerate(self.Blocks):
            flows.append(layer(flows[-1], *volume[self.level-l].chunk(2, 1)))

        return flows[-1]


def norm(input, mean=[.406, .456, .485], std=[.225, .224, .229]):
    return input.sub(input.new_tensor(mean).view(-1, 1, 1)).div(input.new_tensor(std).view(-1, 1, 1))


class SPyNet0(nn.Module):
    """SPyNet"""

    def __init__(self, level=4, path='./models/spy_net-sintel-final.pytorch', trainable=False):
        super(SPyNet0, self).__init__()
        self.level = level
        self.Blocks = nn.ModuleList([SPyBlock() for _ in range(level+1)])

        if path is not None:
            data = torch.load(path, map_location='cpu')
            if 'state_dict' in data.keys():
                self.load_state_dict(data['state_dict'])
            else:
                self.load_state_dict(data)

        if not trainable:
            self.requires_grad_(False)

    def forward(self, im2, im1):  # backwarp
        volume = [torch.cat([norm(im1), norm(im2)], dim=1)]  # B, 6, H, W
        for _ in range(self.level):
            volume.append(F.avg_pool2d(volume[-1], kernel_size=2))

        flows = [torch.zeros_like(volume[-1][:, :2])]  # B, 2, H//16, W//16
        for l, layer in enumerate(self.Blocks):
            flows.append(layer(flows[-1], *volume[self.level-l].chunk(2, 1)))

        return flows[-1]


def extend_2dflow_to_3d(m):
    assert m.out_channels == 2
    old_weight_data = m.weight.data
    new_weight = nn.Parameter(torch.cat([m.weight, m.weight[:1]]))
    # print(new_weight)
    # print(m.weight)
    m.weight = new_weight

    if m.bias is not None:

        old_bias_data = m.bias.data
        new_bias = nn.Parameter(torch.cat([m.bias, m.bias[:1]]))
        # print(new_bias)
        # print(m.bias)
        m.bias = new_bias

    m.out_channels = 3
    m.reset_parameters()

    # print(m.weight)
    m.weight.data[:2].copy_(old_weight_data)
    # print(m.weight)

    if m.bias is not None:
        # print(m.bias)
        m.bias.data[:2].copy_(old_bias_data)
        # print(m.bias)

    # print(m)