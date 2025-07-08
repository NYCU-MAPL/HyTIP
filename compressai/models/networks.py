import torch
import torch.nn as nn
from compressai.models.utils import conv, deconv, bilinearupsacling
from compressai.layers.layers import subpel_conv3x3
from loguru import logger

class DownsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__(
            conv(in_channels, out_channels, kernel_size, stride),
            nn.LeakyReLU(0.1, inplace=True)
        )

class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__(
            conv(in_channels, out_channels, kernel_size, stride),
            nn.LeakyReLU(0.1, inplace=True)
        )

class ResidualBlock(nn.Sequential):

    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__(
            conv(num_filters, num_filters//2, 1, stride=1),
            nn.ReLU(inplace=True),
            conv(num_filters//2, num_filters//2, 3, stride=1),
            nn.ReLU(inplace=True),
            conv(num_filters//2, num_filters, 1, stride=1)
        )

    def forward(self, input):
        return input + super().forward(input)

class TopDown_extractor(nn.Module):
    def __init__(self, dims, kernel_size, scale_list):
        super().__init__()
        assert isinstance(dims, list) and isinstance(kernel_size, list) and isinstance(scale_list, list)

        self.depth = len(kernel_size)
        for i in range(self.depth):
            self.add_module('down'+str(i), DownsampleBlock(dims[i], dims[i+1], kernel_size[i], scale_list[i]))

    def forward(self, input):
        features = []
        features_size = []

        for i in range(self.depth):
            input = self._modules['down'+str(i)](input)
            features.append(input)
            features_size.append(input.shape[2:4])

        return features, features_size

# New for conditional motion coder
class TopDown_extractor_Fusion(TopDown_extractor):
    def __init__(self, dims, kernel_size, scale_list, implicit_channel=64):
        super().__init__(dims, kernel_size, scale_list)

        self.fusion = DownsampleBlock(dims[1]+implicit_channel, dims[1], 3, 1)

    def forward(self, input, implicit_feature):
        features = []
        features_size = []

        for i in range(self.depth):
            input = self._modules['down'+str(i)](input)
            if i == 0:
                input = self.fusion(torch.cat([input, implicit_feature], dim=1))
            
            features.append(input)
            features_size.append(input.shape[2:4])

        return features, features_size
    

#################################### P-frame ####################################

class ResBlock_Inter(nn.Module):
    def __init__(self, channel, slope=0.01, end_with_relu=False,
                 bottleneck=False, inplace=False):
        super().__init__()
        in_channel = channel // 2 if bottleneck else channel
        self.first_layer = nn.LeakyReLU(negative_slope=slope, inplace=False)
        self.conv1 = nn.Conv2d(channel, in_channel, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)
        self.conv2 = nn.Conv2d(in_channel, channel, 3, padding=1)
        self.last_layer = self.relu if end_with_relu else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return identity + out


class FeatureExtractor_Inter(nn.Module):
    def __init__(self, inplace=False, g_ch_1x=48, g_ch_2x=64, g_ch_4x=96):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x, g_ch_1x, 3, stride=1, padding=1)
        self.res_block1 = ResBlock_Inter(g_ch_1x, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_1x, g_ch_2x, 3, stride=2, padding=1)
        self.res_block2 = ResBlock_Inter(g_ch_2x, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_2x, g_ch_4x, 3, stride=2, padding=1)
        self.res_block3 = ResBlock_Inter(g_ch_4x, inplace=inplace)

    def forward(self, feature, quant_step):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)
        
        feature = feature * quant_step

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, inplace=False, g_ch_1x=48, g_ch_2x=64, g_ch_4x=96):
        super().__init__()
        self.conv3_up = subpel_conv3x3(g_ch_4x, g_ch_2x, 2)
        self.res_block3_up = ResBlock_Inter(g_ch_2x, inplace=inplace)
        self.conv3_out = nn.Conv2d(g_ch_4x, g_ch_4x, 3, padding=1)
        self.res_block3_out = ResBlock_Inter(g_ch_4x, inplace=inplace)

        self.conv2_up = subpel_conv3x3(g_ch_2x * 2, g_ch_1x, 2)
        self.res_block2_up = ResBlock_Inter(g_ch_1x, inplace=inplace)
        self.conv2_out = nn.Conv2d(g_ch_2x * 2, g_ch_2x, 3, padding=1)
        self.res_block2_out = ResBlock_Inter(g_ch_2x, inplace=inplace)

        self.conv1_out = nn.Conv2d(g_ch_1x * 2, g_ch_1x, 3, padding=1)
        self.res_block1_out = ResBlock_Inter(g_ch_1x, inplace=inplace)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)

        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)

        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out

        return context1, context2, context3
    
