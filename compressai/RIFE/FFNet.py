import torch.nn as nn


def UpsampleSBlock(in_channels, out_channels, stride=1):
	return nn.Sequential(
		nn.PReLU(),
		nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.Upsample(scale_factor=2, mode='bilinear')
	)
