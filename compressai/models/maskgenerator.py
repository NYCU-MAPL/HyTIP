import torch
import torch.nn as nn

from .utils import conv, deconv

# correct the architecture of the MaskGenerator
class MaskGenerator_Correct(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()

        self.down_1 = nn.Sequential(
            conv(in_channels, 16, 3, 1), 
            nn.LeakyReLU(),
            conv(16, 32, 3, 2), 
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.down_2 = nn.Sequential(
            conv(32, 32, 3, 1), 
            nn.LeakyReLU(),
            conv(32, 64, 3, 2),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.conv = nn.Sequential(
            conv(64, 64, 3, 1), 
            nn.LeakyReLU(),
            conv(64, 64, 3, 1), 
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.up_2 = nn.Sequential(
            deconv(128, 64, 3, 2), 
            nn.LeakyReLU(),
            #nn.ReLU(), 
            conv(64, 32, 3, 1), 
            #nn.ReLU()
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.up_1 = nn.Sequential(
            deconv(64, 32, 3, 2), 
            nn.LeakyReLU(),
            #nn.ReLU(), 
            conv(32, 16, 3, 1), 
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.1),
            conv(16, out_channels, 3, 1), 
            #nn.ReLU()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        down_map_1 = self.down_1(x)
        down_map_2 = self.down_2(down_map_1)

        feature = self.conv(down_map_2)

        up_map_2 = self.up_2(torch.cat([feature, down_map_2], dim=1))
        up_map_1 = self.up_1(torch.cat([up_map_2, down_map_1], dim=1))

        return self.sigmoid(up_map_1)
    

