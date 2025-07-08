# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numbers
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers.gdn import GDN
from einops import rearrange
from torch import Tensor
from torch.autograd import Function

__all__ = [
    "AttentionBlock",
    "MaskedConv2d",
    "MultistageMaskedConv2d",
    "ResidualBlock",
    "ResidualBlock_Inter",
    "ResidualBlockUpsample",
    "ResidualBlockWithStride",
    "conv3x3",
    "subpel_conv3x3",
    "subpel_conv1x1",
    "QReLU",
    "CTM_CAB",
    "ResidualBlockWithStride_Intra",
    "DepthConvBlock",
    "DepthConvBlock_Inter",
    "DepthConvBlock4",
    "ResidualBlockUpsample_Intra",
    "UNet",
    "UNet2"
]

class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)

class MultistageMaskedConv2d(nn.Conv2d):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == 'A':
            self.mask[:, :, 0::2, 0::2] = 1
        elif mask_type == 'B':
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        elif mask_type == 'C':
            self.mask[:, :, :, :] = 1
            self.mask[:, :, 1:2, 1:2] = 0
        else:
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    def forward(self, x: Tensor) -> Tensor:
        # TODO: weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out

class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out

class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out
    

class ResidualBlock_Inter(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch, leaky_relu_slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=inplace)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = conv1x1(in_ch, out_ch)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        out = out + identity
        return out

class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out

class QReLU(Function):
    """QReLU

    Clamping input with given bit-depth range.
    Suppose that input data presents integer through an integer network
    otherwise any precision of input will simply clamp without rounding
    operation.

    Pre-computed scale with gamma function is used for backward computation.

    More details can be found in
    `"Integer networks for data compression with latent-variable models"
    <https://openreview.net/pdf?id=S1zz2i0cY7>`_,
    by Johannes Ballé, Nick Johnston and David Minnen, ICLR in 2019

    Args:
        input: a tensor data
        bit_depth: source bit-depth (used for clamping)
        beta: a parameter for modeling the gradient during backward computation
    """

    @staticmethod
    def forward(ctx, input, bit_depth, beta):
        # TODO(choih): allow to use adaptive scale instead of
        # pre-computed scale with gamma function
        ctx.alpha = 0.9943258522851727
        ctx.beta = beta
        ctx.max_value = 2**bit_depth - 1
        ctx.save_for_backward(input)

        return input.clamp(min=0, max=ctx.max_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        (input,) = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_sub = (
            torch.exp(
                (-ctx.alpha**ctx.beta)
                * torch.abs(2.0 * input / ctx.max_value - 1) ** ctx.beta
            )
            * grad_output.clone()
        )

        grad_input[input < 0] = grad_sub[input < 0]
        grad_input[input > ctx.max_value] = grad_sub[input > ctx.max_value]

        return grad_input, None, None


############################### CTM ###############################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm_From_CTM(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm_From_CTM, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        # return self.patch_unembed(self.body(self.patch_embed(x)), x_size)
        return to_4d(self.body(to_3d(x)), h, w)
    
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, FA=False, num_frame_type=4):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.FA = FA
        self.num_frame_type = num_frame_type

        if FA:
            self.affine = nn.Sequential(
                nn.Linear(num_frame_type, 16),
                nn.Sigmoid(),
                nn.Linear(16, dim * 2, bias=False)
            )

    def forward(self, x, frame_type=None):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        if self.FA: 
            assert frame_type is not None
            affine_params = self.affine(frame_type)
            scale, bias = torch.chunk(affine_params, 2, -1)
            scale = rearrange(scale.unsqueeze(2), 'b (head c) t -> b head c t', head=self.num_heads)
            bias = rearrange(bias.unsqueeze(2), 'b (head c) t -> b head c t', head=self.num_heads)

            v = F.softplus(scale) * v + bias

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out
    
    def flops(self, h, w):
        print(f"================================== {h, w} ==================================")
        # exit()
        
        flops = 0
        # Linear Projection for q, k, v
        flops += (h*w) * self.dim * 3 * self.dim
        
        # qkv_dwConv
        flops += (3 * 3) * h * w * (self.dim * 3)
        
        # Channel attention
        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        flops += self.dim * (h*w) * (self.dim // self.num_heads) # temperature should not be counted ?
        
        #  out = (attn @ v)
        flops += self.dim * (h*w) * (self.dim // self.num_heads)
        
        # out = self.project_out(out)
        flops += (h*w) * self.dim * self.dim

        if self.FA: 
            flops += 1 * self.num_frame_type * 16
            flops += 16 * self.dim * 2
        
        return flops

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_features, bias, act_layer, drop=0.):
        super(FeedForward, self).__init__()

        self.dim = dim
        self.hidden_features = hidden_features
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.act_layer = act_layer()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.project_in(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.project_out(x)
        x = self.drop(x)
        return x
    
    def flops(self, h, w):
    
        flops = 0
        
        # x = self.project_in(x)
        flops += (h*w) * self.dim * self.hidden_features
        # self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        flops += (h*w) * self.dim * self.hidden_features
        
        return flops
    
class CTMBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., 
                 act_layer=nn.GELU, CA_bias=False, FA=False, num_frame_type=4,
                 ):
        super().__init__()

        self.norm1 = LayerNorm_From_CTM(dim)
        self.attn = ChannelAttention(dim, num_heads, CA_bias, FA=FA, num_frame_type=num_frame_type)
        
        self.norm2 = LayerNorm_From_CTM(dim)
        self.mlp = FeedForward(dim=dim, hidden_features=int(dim * mlp_ratio), bias=CA_bias, act_layer=act_layer, drop=drop)

    def forward(self, x, frame_type=None):
        x = x + self.attn(self.norm1(x), frame_type)
        x = x + self.mlp(self.norm2(x))
        return x 
    
    def flops(self, H, W):
        flops = 0
        
        # norm1
        flops += self.dim * (H * W + H * W)
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        if self.withMLP:
            # mlp
            flops += self.mlp.flops(H, W)
        # norm2
        flops += self.dim * (H * W + H * W)
        return flops
    
class CTMBasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., drop=0., 
                 norm_layer=nn.LayerNorm, CA_bias=False, FA=False, num_frame_type=4,
                 ):
        super().__init__()
        
        # build blocks
        self.blocks = nn.ModuleList([
                        CTMBlock(dim=dim, 
                                       num_heads=num_heads,
                                       mlp_ratio=mlp_ratio,
                                       drop=drop, 
                                       CA_bias=CA_bias,
                                       FA=FA,
                                       num_frame_type=num_frame_type,
                                       )
            for i in range(depth)])

    def forward(self, x, frame_type=None):
        for i, blk in enumerate(self.blocks):
            x = blk(x, frame_type)
        
        return {'output': x}

    def flops(self, H, W):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops
    
class CTM_CAB(nn.Module):
    def __init__(self, dim, depth, num_heads, input_resolution=(256, 256),
                 mlp_ratio=4., drop=0., norm_layer=nn.LayerNorm, CA_bias=False, 
                 FA=False, num_frame_type=4):

        super(CTM_CAB, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.restor_layer = CTMBasicLayer(dim=dim,
                                                depth=depth,
                                                num_heads=num_heads,
                                                mlp_ratio=mlp_ratio,
                                                drop=drop,
                                                norm_layer=norm_layer,
                                                CA_bias=CA_bias,
                                                FA=FA,
                                                num_frame_type=num_frame_type,
                                                )

    def forward(self, x, frame_type=None):

        out_dir1 = self.restor_layer(x, frame_type)
        output = out_dir1['output']
        output = output + x

        return {'output': output}

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.restor_layer.flops(H, W)
        return flops

############################### Intra ###############################
class ResidualBlockWithStride_Intra(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride=2, inplace=False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        if stride != 1:
            self.downsample = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out


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

class DepthConv2(nn.Module):
    def __init__(self, in_ch, out_ch, slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch)
        )
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1)
        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.out_conv(x1 * x2)
        return identity + x


class DepthConv_Intra(nn.Module):
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


class DepthConv_Inter(nn.Module):
    def __init__(self, in_ch, out_ch, slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)

        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity

class ConvFFN_Intra(nn.Module):
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

class ConvFFN_Inter(nn.Module):
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

class ConvFFN(nn.Module): ### class ConvFFN2
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        expansion_factor = 2
        slope = 0.1
        internal_ch = in_ch * expansion_factor
        self.conv = nn.Conv2d(in_ch, internal_ch * 2, 1)
        self.conv_out = nn.Conv2d(internal_ch, in_ch, 1)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

    def forward(self, x):
        identity = x
        x1, x2 = self.conv(x).chunk(2, 1)
        out = x1 * self.relu(x2)
        return identity + self.conv_out(out)
    
class ConvFFN3(nn.Module):
    def __init__(self, in_ch, inplace=False):
        super().__init__()
        expansion_factor = 2
        internal_ch = in_ch * expansion_factor
        self.conv = nn.Conv2d(in_ch, internal_ch * 2, 1)
        self.conv_out = nn.Conv2d(internal_ch, in_ch, 1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=inplace)

    def forward(self, x):
        identity = x
        x1, x2 = self.conv(x).chunk(2, 1)
        out = self.relu1(x1) + self.relu2(x2)
        return identity + self.conv_out(out)

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


class DepthConvBlock_Inter(nn.Module):
    def __init__(self, in_ch, out_ch, slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv_Inter(in_ch, out_ch, slope=slope_depth_conv, inplace=inplace),
            ConvFFN_Inter(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


class DepthConvBlock4(nn.Module):
    def __init__(self, in_ch, out_ch, slope_depth_conv=0.01, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv_Inter(in_ch, out_ch, slope=slope_depth_conv, inplace=inplace),
            # ConvFFN3(out_ch, inplace=inplace),
            ConvFFN3(out_ch, inplace=False if self.training else inplace),
        )

    def forward(self, x):
        return self.block(x)

def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), 
        nn.PixelShuffle(r)
    )

class ResidualBlockUpsample_Intra(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2, inplace=False):
        super().__init__()
        self.subpel_conv = subpel_conv1x1(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu2(out)
        identity = self.upsample(x)
        out = out + identity
        return out

class UNet(nn.Module):  # UNet2
    def __init__(self, in_ch=64, out_ch=64, inplace=False):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DepthConvBlock(in_ch, 32, inplace=inplace)
        self.conv2 = DepthConvBlock(32, 64, inplace=inplace)
        self.conv3 = DepthConvBlock(64, 128, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = DepthConvBlock(128, 64, inplace=inplace)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = DepthConvBlock(64, out_ch, inplace=inplace)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


class UNet2(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, inplace=False, 
                #  Pretrain=False
                 ):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DepthConvBlock4(in_ch, 32, inplace=inplace)
        self.conv2 = DepthConvBlock4(32, 64, inplace=inplace)
        self.conv3 = DepthConvBlock4(64, 128, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock4(128, 128, inplace=inplace),
            DepthConvBlock4(128, 128, inplace=inplace),
            DepthConvBlock4(128, 128, inplace=inplace),
            DepthConvBlock4(128, 128, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = DepthConvBlock4(128, 64, inplace=inplace)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = DepthConvBlock4(64, out_ch, inplace=inplace)


    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        return d2
