import torch
import torch.nn as nn

from torch import Tensor
from functools import partial
from ..entropy_models.entropy_models import GaussianConditional

class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1, slope=0.01, inplace=False):
        super().__init__()
        dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, dw_ch, 1, stride=stride),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(dw_ch, dw_ch, depth_kernel, padding=depth_kernel // 2,
                                    groups=dw_ch)
        self.conv2 = nn.Conv2d(dw_ch, out_ch, 1)

        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        elif in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity
    
class ConvFFN(nn.Module):
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        internal_ch = max(min(in_ch * 4, 1024), in_ch * 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, internal_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
            nn.Conv2d(internal_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )

    def forward(self, x):
        identity = x
        return identity + self.conv(x)

class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1,
                 slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel, stride, slope=slope_depth_conv, inplace=inplace),
            ConvFFN(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)
    
class FusionBlock(nn.Module):

    def __init__(self, incs):
        super(FusionBlock, self).__init__()
        self.conv1 = DepthConvBlock(incs[0], incs[1], inplace=True)
        self.conv2 = DepthConvBlock(incs[1], incs[2], inplace=True)
        self.conv3 = DepthConvBlock(incs[2], incs[3], inplace=True)
            
    def forward(self, x):
        feature = self.conv1(x)
        
        feature = self.conv2(feature)

        feature = self.conv3(feature)
        return feature

class Checkerboard(nn.Module):
    def __init__(self, num_features, quant_mode="noise", inplace=False):
        super().__init__()

        self.y_spatial_prior = DepthConvBlock(num_features * 3, num_features * 2, inplace=inplace)
        
        self.masks = {}
        self.gaussian_conditional = GaussianConditional(None, quant_mode=quant_mode)
        
    def ste_round(self, x: Tensor) -> Tensor: 
        return torch.round(x) - x.detach() + x
    
    def process_with_mask(self, y, scales, means, mask):
        mask = mask.to(y.device)
        scales_hat = scales * mask
        means_hat = means * mask
        
        y_res = (y - means_hat) * mask
        y_q = self.ste_round(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat, means_hat
    
    def get_mask_four_parts(self, height, width, dtype, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.masks:
            micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
            mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)

            micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
            mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
            mask_1 = mask_1[:height, :width]
            mask_1 = torch.unsqueeze(mask_1, 0)
            mask_1 = torch.unsqueeze(mask_1, 0)

            micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
            mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
            mask_2 = mask_2[:height, :width]
            mask_2 = torch.unsqueeze(mask_2, 0)
            mask_2 = torch.unsqueeze(mask_2, 0)

            micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
            mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
            mask_3 = mask_3[:height, :width]
            mask_3 = torch.unsqueeze(mask_3, 0)
            mask_3 = torch.unsqueeze(mask_3, 0)
            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]
    
    def update(self, scale_table, force=False):
        self.gaussian_conditional.update_scale_table(scale_table=scale_table, force=force)

    def forward(self, y, prior_param):
        # if 3-Prior, q_step handles at the outside of the context model
        H, W = y.shape[-2:]
        scales, means = prior_param.chunk(2, 1)
        
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, y.dtype, y.device)
        
        y_res_0, y_q_0, y_hat_0, scales_hat_0, means_hat_0 = \
            self.process_with_mask(y, scales, means, mask_0)
        
        y_hat_so_far = y_hat_0
        scales_hat_so_far = scales_hat_0
        means_hat_so_far = means_hat_0
        params = torch.cat((y_hat_so_far, scales, means), dim=1)
        scales_0, means_0 = self.y_spatial_prior(params).chunk(2, 1)
        
        
        y_res_1, y_q_1, y_hat_1, scales_hat_1, means_hat_1 = \
            self.process_with_mask(y, scales_0, means_0, mask_3)

        y_hat_so_far = y_hat_so_far + y_hat_1
        scales_hat_so_far = scales_hat_so_far + scales_hat_1
        means_hat_so_far = means_hat_so_far + means_hat_1
        params = torch.cat((y_hat_so_far, scales, means), dim=1)
        scales_1, means_1 = self.y_spatial_prior(params).chunk(2, 1)


        y_res_2, y_q_2, y_hat_2, scales_hat_2, means_hat_2 = \
            self.process_with_mask(y, scales_1, means_1, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        scales_hat_so_far = scales_hat_so_far + scales_hat_2
        means_hat_so_far = means_hat_so_far + means_hat_2
        params = torch.cat((y_hat_so_far, scales, means), dim=1)
        scales_2, means_2 = self.y_spatial_prior(params).chunk(2, 1)


        y_res_3, y_q_3, y_hat_3, scales_hat_3, means_hat_3 = \
            self.process_with_mask(y, scales_2, means_2, mask_1)
            
        y_hat_so_far = y_hat_so_far + y_hat_3
        scales_hat_so_far = scales_hat_so_far + scales_hat_3
        means_hat_so_far = means_hat_so_far + means_hat_3
        
        y_hat_noise, y_likelihoods = self.gaussian_conditional(y, scales_hat_so_far, means=means_hat_so_far)

        return y_hat_so_far, y_likelihoods, {'mean': means_hat_so_far, 'scale': scales_hat_so_far}

    def compress(self, y, prior_param):
        H, W = y.shape[-2:]
        scales, means = prior_param.chunk(2, 1)
        
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, y.dtype, y.device)
        
        y_res_0, y_q_0, y_hat_0, scales_hat_0, means_hat_0 = \
            self.process_with_mask(y, scales, means, mask_0)
        
        y_hat_so_far = y_hat_0
        scales_hat_so_far = scales_hat_0
        means_hat_so_far = means_hat_0
        params = torch.cat((y_hat_so_far, scales, means), dim=1)
        scales_0, means_0 = self.y_spatial_prior(params).chunk(2, 1)
        
        
        y_res_1, y_q_1, y_hat_1, scales_hat_1, means_hat_1 = \
            self.process_with_mask(y, scales_0, means_0, mask_3)

        y_hat_so_far = y_hat_so_far + y_hat_1
        scales_hat_so_far = scales_hat_so_far + scales_hat_1
        means_hat_so_far = means_hat_so_far + means_hat_1
        params = torch.cat((y_hat_so_far, scales, means), dim=1)
        scales_1, means_1 = self.y_spatial_prior(params).chunk(2, 1)


        y_res_2, y_q_2, y_hat_2, scales_hat_2, means_hat_2 = \
            self.process_with_mask(y, scales_1, means_1, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        scales_hat_so_far = scales_hat_so_far + scales_hat_2
        means_hat_so_far = means_hat_so_far + means_hat_2
        params = torch.cat((y_hat_so_far, scales, means), dim=1)
        scales_2, means_2 = self.y_spatial_prior(params).chunk(2, 1)


        y_res_3, y_q_3, y_hat_3, scales_hat_3, means_hat_3 = \
            self.process_with_mask(y, scales_2, means_2, mask_1)
        
        y_hat_so_far = y_hat_so_far + y_hat_3
        scales_hat_so_far = scales_hat_so_far + scales_hat_3
        means_hat_so_far = means_hat_so_far + means_hat_3
            
        y_strings = []

        indexes = self.gaussian_conditional.build_indexes(scales_hat_0)
        y_strings.append(self.gaussian_conditional.compress(y_q_0, indexes))

        indexes = self.gaussian_conditional.build_indexes(scales_hat_1)
        y_strings.append(self.gaussian_conditional.compress(y_q_1, indexes))

        indexes = self.gaussian_conditional.build_indexes(scales_hat_2)
        y_strings.append(self.gaussian_conditional.compress(y_q_2, indexes))

        indexes = self.gaussian_conditional.build_indexes(scales_hat_3)
        y_strings.append(self.gaussian_conditional.compress(y_q_3, indexes))

        return y_hat_so_far, y_strings, {'mean': means_hat_so_far, 'scale': scales_hat_so_far}
    
    def decompress(self, strings, prior_param):
        assert len(strings) == 4

        H, W = prior_param.shape[-2:]
        scales, means = prior_param.chunk(2, 1)

        dtype = means.dtype
        device = means.device
        
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)
        
        indexes = self.gaussian_conditional.build_indexes(scales * mask_0)
        y_q_r = self.gaussian_conditional.decompress(strings[0], indexes)
        y_hat_0 = (y_q_r + means) * mask_0
        y_hat_so_far = y_hat_0
        
        params = torch.cat((y_hat_so_far, scales, means), dim=1)
        scales_0, means_0 = self.y_spatial_prior(params).chunk(2, 1)
        
        indexes = self.gaussian_conditional.build_indexes(scales_0 * mask_3)
        y_q_r = self.gaussian_conditional.decompress(strings[1], indexes)
        y_hat_1 = (y_q_r + means_0) * mask_3
        y_hat_so_far = y_hat_so_far + y_hat_1
        
        params = torch.cat((y_hat_so_far, scales, means), dim=1)
        scales_1, means_1 = self.y_spatial_prior(params).chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_1 * mask_2)
        y_q_r = self.gaussian_conditional.decompress(strings[2], indexes)
        y_hat_2 = (y_q_r + means_1) * mask_2
        y_hat_so_far = y_hat_so_far + y_hat_2
        
        params = torch.cat((y_hat_so_far, scales, means), dim=1)
        scales_2, means_2 = self.y_spatial_prior(params).chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_2 * mask_1)
        y_q_r = self.gaussian_conditional.decompress(strings[3], indexes)
        y_hat_3 = (y_q_r + means_2) * mask_1
        y_hat_so_far = y_hat_so_far + y_hat_3

        return y_hat_so_far
    
__CONTEXT_TYPES__ = {
    "Checkerboard": Checkerboard,
}