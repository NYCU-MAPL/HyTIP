import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import compressai.models.transform as ts

from compressai.models.utils import bilineardownsacling
from compressai.RIFE.FFNet import UpsampleSBlock

from ..entropy_models import GaussianConditional, CompressionModel, CompressionModel_Inter, EntropyCoder_Inter
from ..entropy_models.context_model import Checkerboard
from ..layers import (
    CTM_CAB, 
    ResidualBlockWithStride_Intra, 
    DepthConvBlock, DepthConvBlock_Inter, 
    DepthConvBlock4,
    ResidualBlockUpsample_Intra,
    UNet, 
    UNet2, ResidualBlock_Inter,
    subpel_conv1x1, subpel_conv3x3,
)
from .networks import (
    TopDown_extractor, TopDown_extractor_Fusion,
    FeatureExtractor_Inter, MultiScaleContextFusion
)
from .block_mc import block_mc_func
from .utils import get_padding_size, get_downsampled_shape
from util.stream_helper_SPS import flatten_strings_list

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

    
#################################### Intra ####################################
class IntraEncoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride_Intra(3, 128, stride=2, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
        )
        self.enc_2 = nn.Sequential(
            ResidualBlockWithStride_Intra(128, 192, stride=2, inplace=inplace),
            DepthConvBlock(192, 192, inplace=inplace),
            ResidualBlockWithStride_Intra(192, N, stride=2, inplace=inplace),
            DepthConvBlock(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = self.enc_1(x)
        out = out * quant_step
        return self.enc_2(out)

class IntraDecoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.dec_1 = nn.Sequential(
            DepthConvBlock(N, N, inplace=inplace),
            ResidualBlockUpsample_Intra(N, N, 2, inplace=inplace),
            DepthConvBlock(N, N, inplace=inplace),
            ResidualBlockUpsample_Intra(N, 192, 2, inplace=inplace),
            DepthConvBlock(192, 192, inplace=inplace),
            ResidualBlockUpsample_Intra(192, 128, 2, inplace=inplace),
        )
        self.dec_2 = nn.Sequential(
            DepthConvBlock(128, 128, inplace=inplace),
            ResidualBlockUpsample_Intra(128, 16, 2, inplace=inplace),
        )

    def forward(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        return self.dec_2(out)

class Intra_NoAR(CompressionModel):
    def __init__(self, N=256, anchor_num=4, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(y_distribution='gaussian', z_channel=N,
                         ec_thread=ec_thread, stream_part=stream_part)

        self.enc = IntraEncoder(N, inplace)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )
        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample_Intra(N, N, 2, inplace=inplace),
            ResidualBlockUpsample_Intra(N, N, 2, inplace=inplace),
            DepthConvBlock(N, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 2, inplace=inplace),
            DepthConvBlock(N * 2, N * 3, inplace=inplace),
        )

        self.y_spatial_prior_adaptor_1 = nn.Conv2d(N * 4, N * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(N * 4, N * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(N * 4, N * 3, 1)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(N * 3, N * 3, inplace=inplace),
            DepthConvBlock(N * 3, N * 2, inplace=inplace),
            DepthConvBlock(N * 2, N * 2, inplace=inplace),
        )

        self.dec = IntraDecoder(N, inplace)
        self.refine = nn.Sequential(
            UNet(16, 16, inplace=inplace),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )

        self.q_basic_enc = nn.Parameter(torch.ones((1, 128, 1, 1)))
        self.q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.q_scale_enc_fine = None
        self.q_basic_dec = nn.Parameter(torch.ones((1, 128, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.q_scale_dec_fine = None

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
        self.interpolate()

    def interpolate(self):

        with torch.no_grad():
            q_scale_enc_fine = np.linspace(np.log(self.q_scale_enc[0, 0, 0, 0]),
                                            np.log(self.q_scale_enc[3, 0, 0, 0]), 64)
            self.q_scale_enc_fine = torch.from_numpy(np.exp(q_scale_enc_fine)).to(dtype=torch.float32)
            q_scale_dec_fine = np.linspace(np.log(self.q_scale_dec[0, 0, 0, 0]),
                                            np.log(self.q_scale_dec[3, 0, 0, 0]), 64)
            self.q_scale_dec_fine = torch.from_numpy(np.exp(q_scale_dec_fine)).to(dtype=torch.float32)

    def get_curr_q(self, q_scale, q_basic, q_index=None):
        q_scale = q_scale.to(q_basic.device)
        q_scale = q_scale[q_index]
        return q_basic * q_scale.view(-1, 1, 1, 1)

    def get_q_for_inference(self, q_in_ckpt, q_index):
        q_scale_enc = self.q_scale_enc[:, 0, 0, 0] if q_in_ckpt else self.q_scale_enc_fine
        curr_q_enc = self.get_curr_q(q_scale_enc, self.q_basic_enc, q_index=q_index)
        q_scale_dec = self.q_scale_dec[:, 0, 0, 0] if q_in_ckpt else self.q_scale_dec_fine
        curr_q_dec = self.get_curr_q(q_scale_dec, self.q_basic_dec, q_index=q_index)
        return curr_q_enc, curr_q_dec

    def forward(self, x, q_in_ckpt=False, q_index=None):
        curr_q_enc, curr_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        y = self.enc(x, curr_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat = self.quant(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        _, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        x_hat = self.dec(y_hat, curr_q_dec)
        x_hat = self.refine(x_hat)

        y_for_bit = y_q
        z_for_bit = z_hat
        bits_y, likelihood_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z, likelihood_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        _, _, H, W = x.size()
        pixel_num = H * W
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bits = torch.sum(bpp_y + bpp_z) * pixel_num
        bpp = bpp_y + bpp_z

        return {
            "x_hat": x_hat,
            "bit": bits,
            "bpp": bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            "likelihoods":{"y":likelihood_y, "z":likelihood_z}
        }
    
    def compress(self, x, q_in_ckpt, q_index):
        curr_q_enc, curr_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        y = self.enc(x, curr_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat = torch.round(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        self.entropy_coder.reset()
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        self.entropy_coder.flush()

        x_hat = self.refine(self.dec(y_hat, curr_q_dec)).clamp_(0, 1)
        bit_stream = self.entropy_coder.get_encoded_stream()

        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
        }
        return result
    
    def decompress(self, bit_stream, height, width, q_in_ckpt, q_index):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        _, curr_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        self.entropy_coder.set_stream(bit_stream)
        z_size = get_downsampled_shape(height, width, 64)
        y_height, y_width = get_downsampled_shape(height, width, 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)

        x_hat = self.refine(self.dec(y_hat, curr_q_dec)).clamp_(0, 1)
        return {"x_hat": x_hat}
    
    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(
            y,
            (padding_l, padding_r, padding_t, padding_b),
            mode="replicate",
        )
        return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)
    
    def quant(self, x):
        return (torch.round(x) - x).detach() + x


########## advanced_model: Mask Generation ##########
from .maskgenerator import MaskGenerator_Correct
from numpy import ceil

def split_k(input, size: int, dim: int = 0):
    """reshape input to original batch size"""
    if dim < 0:
        dim = input.dim() + dim
    split_size = list(input.size())
    split_size[dim] = size
    split_size.insert(dim+1, -1)
    return input.view(split_size)

def cat_k(input):
    """concat second dimesion to batch"""
    return input.flatten(0, 1)

class Alignment(torch.nn.Module):
    """Image Alignment for model downsample requirement"""

    def __init__(self, divisor=64., mode='pad', padding_mode='replicate'):
        super().__init__()
        self.divisor = float(divisor)
        self.mode = mode
        self.padding_mode = padding_mode
        self._tmp_shape = None

    def extra_repr(self):
        s = 'divisor={divisor}, mode={mode}'
        if self.mode == 'pad':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    @staticmethod
    def _resize(input, size):
        return F.interpolate(input, size, mode='bilinear', align_corners=False)

    def _align(self, input):
        H, W = input.size()[-2:]
        H_ = int(ceil(H / self.divisor) * self.divisor)
        W_ = int(ceil(W / self.divisor) * self.divisor)
        pad_H, pad_W = H_-H, W_-W
        if pad_H == pad_W == 0:
            self._tmp_shape = None
            return input

        self._tmp_shape = input.size()
        if self.mode == 'pad':
            return F.pad(input, (0, pad_W, 0, pad_H), mode=self.padding_mode)
        elif self.mode == 'resize':
            return self._resize(input, size=(H_, W_))

    def _resume(self, input, shape=None):
        if shape is not None:
            self._tmp_shape = shape
        if self._tmp_shape is None:
            return input

        if self.mode == 'pad':
            output = input[..., :self._tmp_shape[-2], :self._tmp_shape[-1]]
        elif self.mode == 'resize':
            output = self._resize(input, size=self._tmp_shape[-2:])

        return output

    def align(self, input):
        """align"""
        if input.dim() == 4:
            return self._align(input)
        elif input.dim() == 5:
            return split_k(self._align(cat_k(input)), input.size(0))

    def align_to(self, input, shape):
        H, W = input.size()[-2:]
        assert (shape[0] >= H and shape[1] >= W), f'(H, W) = {(H, W)}, shape={shape}'
        pad_H, pad_W = shape[0] - H , shape[1] - W
        
        if self.mode == 'pad':
            return F.pad(input, (0, pad_W, 0, pad_H), mode=self.padding_mode)
        elif self.mode == 'resize':
            return self._resize(input, size=(shape[0], shape[1]))

    def resume(self, input, shape=None):
        """resume"""
        if input.dim() == 4:
            return self._resume(input, shape)
        elif input.dim() == 5:
            return split_k(self._resume(cat_k(input), shape), input.size(0))

    def forward(self, func, *args, **kwargs):
        pass

class MaskGeneration(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, align_factor=4):
        super().__init__()

        self.net = MaskGenerator_Correct(in_channels, out_channels)
        
        self.align = Alignment(align_factor)

    def forward(self, flow):
        flow = self.align.align(flow)

        mask = self.net(flow)
        mask = self.align.resume(mask)

        return mask

#################################### P-frame Codec ####################################

class MvEnc(nn.Module):
    def __init__(self, input_channel, channel, inplace=False, cond_source='ref_mv_feature', cond_dims=[64]):
        super().__init__()

        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride_Intra(input_channel, channel, stride=2, inplace=inplace),
            DepthConvBlock4(channel, channel, inplace=inplace),
        )
        self.enc_2 = ResidualBlockWithStride_Intra(channel, channel, stride=2, inplace=inplace)

        self.adaptor_0 = DepthConvBlock4(channel, channel, inplace=inplace)
        self.adaptor_1 = DepthConvBlock4(channel + cond_dims[0], channel, inplace=inplace)
        self.enc_3 = nn.Sequential(
            ResidualBlockWithStride_Intra(channel, channel, stride=2, inplace=inplace),
            DepthConvBlock4(channel, channel, inplace=inplace),
            nn.Conv2d(channel, channel, 3, stride=2, padding=1),
        )

        self.adaptor_3_1 = DepthConvBlock4(channel + cond_dims[1], channel, inplace=inplace)

    def forward(self, x, context, quant_step):
        out = self.enc_1(x)
        if quant_step is not None:
            out = out * quant_step
        
        out = self.enc_2(out)

        if context is None:
            out = self.adaptor_0(out)
        else:
            out = self.adaptor_1(torch.cat((out, context[0]), dim=1))
        
        if context is None:
            out = self.enc_3(out)
        else:
            out = self.enc_3[0](out)
            out = self.adaptor_3_1(torch.cat((out, context[1]), dim=1))
            out = self.enc_3[2](out)
            
        return out


class MvDec(nn.Module):
    def __init__(self, output_channel, channel, inplace=False, cond_source='ref_mv_feature', cond_dims=[]):
        super().__init__()

        self.dec_1 = nn.Sequential(
            DepthConvBlock4(channel, channel, inplace=inplace),
            ResidualBlockUpsample_Intra(channel, channel, 2, inplace=inplace),
            DepthConvBlock4(channel, channel, inplace=inplace),
            ResidualBlockUpsample_Intra(channel, channel, 2, inplace=inplace),
            DepthConvBlock4(channel, channel, inplace=inplace)
        )
        self.dec_2 = ResidualBlockUpsample_Intra(channel, channel, 2, inplace=inplace)
        self.dec_3 = nn.Sequential(
            DepthConvBlock4(channel, channel, inplace=inplace),
            subpel_conv1x1(channel, output_channel, 2),
        )

        self.adaptor_0_1 = DepthConvBlock4(channel + cond_dims[0], channel, inplace=inplace)
        self.adaptor_1_1 = DepthConvBlock4(channel + cond_dims[1], channel, inplace=inplace)

    def forward(self, x, quant_step, context=None):
        if context is None:
            feature = self.dec_1(x)
        else:
            assert len(context) == 2
            x = self.dec_1[:2](x)
            x = self.adaptor_0_1(torch.cat((x, context[0]), dim=1))
            x = self.dec_1[3](x)
            feature = self.adaptor_1_1(torch.cat((x, context[1]), dim=1))
        
        out = self.dec_2(feature)
        if quant_step is not None:
            out = out * quant_step
        
        mv = self.dec_3(out)

        return mv, feature

class Motion(CompressionModel_Inter):
    def __init__(self, ec_thread=False, stream_part=1, inplace=False, g_ch_z=64, channel_mv=64, channel_N=64, 
                 cond_source='flow_hat+ref_mv_feature', cond_input_dim=[2], enc_cond_dims=[64, 64], dec_cond_dims=[64, 64], cond_kernel_size=[5, 3], cond_scale_list=[4, 2], predprior_input_dim=5, temp_source='ref_intensity+ref_mv_y', 
                 variable_rate=True, quant_mode='estUN_outR', implicit_channel=2, add_fg=True):
        super().__init__(y_distribution='laplace', z_channel=g_ch_z, mv_z_channel=64,
                         ec_thread=ec_thread, stream_part=stream_part)
        
        self.__delattr__('bit_estimator_z')
        
        self.variable_rate = variable_rate
        self.cond_source = cond_source
        self.cond_scale_list = cond_scale_list
        self.temp_source = temp_source
        self.channel_mv = channel_mv
        
        
        if len(cond_scale_list) > 0:
            if self.cond_source in ['flow_hat']:
                self.feature_extractor = TopDown_extractor(cond_input_dim+enc_cond_dims, cond_kernel_size, cond_scale_list)
            elif self.cond_source in ['flow_hat+ref_mv_feature']:
                self.feature_extractor = TopDown_extractor_Fusion(cond_input_dim+enc_cond_dims, cond_kernel_size, cond_scale_list, implicit_channel)
        
        self.mv_encoder = MvEnc(2, channel_mv, inplace=inplace, cond_source=cond_source, cond_dims=enc_cond_dims)

        self.mv_hyper_prior_encoder = nn.Sequential(
            DepthConvBlock4(channel_mv, channel_N, inplace=inplace),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )
        self.mv_hyper_prior_decoder = nn.Sequential(
            ResidualBlockUpsample_Intra(channel_N, channel_N, 2, inplace=inplace),
            ResidualBlockUpsample_Intra(channel_N, channel_N, 2, inplace=inplace),
            DepthConvBlock4(channel_N, channel_mv),
        )

        if self.temp_source in ['ref_intensity']:
            self.pred_prior = ts.GoogleAnalysisTransform(predprior_input_dim, channel_mv, 128, 3)
            
            self.mv_y_prior_fusion_adaptor_temp_0 = DepthConvBlock_Inter(channel_mv * 1, channel_mv * 2, inplace=inplace)
            self.mv_y_prior_fusion_adaptor_temp_1 = DepthConvBlock_Inter(channel_mv * 2, channel_mv * 2, inplace=inplace)
            
            self.PA = nn.Sequential(
                DepthConvBlock_Inter(channel_mv * 2, channel_mv * 3, inplace=inplace),
                DepthConvBlock_Inter(channel_mv * 3, channel_mv * 3, inplace=inplace),
            )
        elif self.temp_source in ['ref_intensity+ref_mv_y']:
            self.temp_prior_fusion_adaptor_0 = DepthConvBlock_Inter(channel_mv * 1, channel_mv * 2, inplace=inplace)
            self.temp_prior_fusion_adaptor_1 = DepthConvBlock_Inter(channel_mv * 3, channel_mv * 2, inplace=inplace)
            
            self.pred_prior = ts.GoogleAnalysisTransform(predprior_input_dim, channel_mv, 128, 3)
            self.prior_fusion = nn.Sequential(
                DepthConvBlock_Inter(channel_mv * 2, channel_mv * 3, inplace=inplace),
                DepthConvBlock_Inter(channel_mv * 3, channel_mv * 3, inplace=inplace),
            )

        self.gaussian_conditional = GaussianConditional(None, quant_mode=quant_mode)

        self.mv_decoder = MvDec(2, channel_mv, inplace=inplace, cond_source=cond_source, cond_dims=dec_cond_dims)

        
        if ('ref_mv_feature' in self.cond_source) and (add_fg or (channel_mv != implicit_channel)):
            self.feature_generator = nn.Conv2d(channel_mv, implicit_channel, 3, padding=1)
        
        
        if self.variable_rate:
            self.mv_q_encoder = nn.Parameter(torch.ones((64, channel_mv, 1, 1)))
            self.mv_q_decoder = nn.Parameter(torch.ones((64, channel_mv, 1, 1)))

    def mv_prior_param_decoder(self, mv_z_hat, dpb, slice_shape=None):
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        mv_params = self.slice_to_y(mv_params, slice_shape)
        
        if self.temp_source in ['ref_intensity']:
            if dpb['temporal_input'] is None:
                mv_params = self.mv_y_prior_fusion_adaptor_temp_0(mv_params)
            else:
                temporal_params = self.pred_prior(dpb['temporal_input'])
                mv_params = torch.cat([mv_params, temporal_params], dim=1)
                mv_params = self.mv_y_prior_fusion_adaptor_temp_1(mv_params)
            mv_params = self.PA(mv_params)
        
        elif self.temp_source in ['ref_intensity+ref_mv_y']:
            if dpb["ref_mv_y"] is None and dpb["temporal_input"] is None:
                mv_params = self.temp_prior_fusion_adaptor_0(mv_params)
            else:
                assert dpb["ref_mv_y"] is not None and dpb["temporal_input"] is not None
                temporal_params = self.pred_prior(dpb["temporal_input"])
                mv_params = torch.cat((mv_params, dpb["ref_mv_y"], temporal_params), dim=1)
                mv_params = self.temp_prior_fusion_adaptor_1(mv_params)
            mv_params = self.prior_fusion(mv_params)

        return mv_params
        
    def quant(self, x):
        if self.training:
            n = torch.round(x) - x
            n = n.clone().detach()
            return x + n
        else:
            return torch.round(x)
    
    def set_noise_level(self, noise_level):
        self.noise_level = noise_level

    def add_noise(self, x):
        noise = torch.nn.init.uniform_(torch.zeros_like(x), -self.noise_level, self.noise_level)
        noise = noise.clone().detach()
        return x + noise

    def forward(self, est_mv, aux_buf={}):

        if self.variable_rate:
        
            q_index = aux_buf['index_list']
            mv_y_q_enc = self.mv_q_encoder[q_index]
            mv_y_q_dec = self.mv_q_decoder[q_index]
        else:
            mv_y_q_enc, mv_y_q_dec = None, None

        index = self.get_index_tensor(0, est_mv.device)
        
        
        if len(self.cond_scale_list) > 0:

            if self.cond_source in ['flow_hat']:
                conds, _ = self.feature_extractor(aux_buf["cond_input"]) if aux_buf["cond_input"] is not None else (None, None)
            elif self.cond_source in ['flow_hat+ref_mv_feature']:
                conds, _ = self.feature_extractor(aux_buf["cond_input"], aux_buf["ref_mv_feature"]) if aux_buf["cond_input"] is not None else (None, None)

        if "compute_decode_macs" in aux_buf and aux_buf["compute_decode_macs"]:
            mv_y = torch.zeros([1, self.channel_mv, est_mv.shape[-2] // 16, est_mv.shape[-1] // 16]).to(est_mv.device)
            mv_y_pad, slice_shape = self.pad_for_y(mv_y)
            mv_z_hat = torch.zeros([1, self.channel_mv, est_mv.shape[-2] // 64, est_mv.shape[-1] // 64]).to(est_mv.device)
        else:
            mv_y = self.mv_encoder(est_mv, conds, mv_y_q_enc)
            
            mv_y_pad, slice_shape = self.pad_for_y(mv_y)
            mv_z = self.mv_hyper_prior_encoder(mv_y_pad)
            mv_z_hat = self.quant(mv_z)     # Output for decoder
            
        mv_params = self.mv_prior_param_decoder(mv_z_hat, aux_buf, slice_shape)

        mv_q_enc, mv_q_dec, mv_scales_hat, mv_means_hat = self.separate_prior(mv_params, is_video=True)
        mv_y = mv_y * mv_q_enc
        mv_y_hat, mv_y_likelihoods = self.gaussian_conditional(mv_y, mv_scales_hat, mv_means_hat)
        mv_y_hat = mv_y_hat * mv_q_dec
        
        mv_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec, conds[::-1] if conds is not None else None)

        
        if hasattr(self, "feature_generator"):
            mv_feature = self.feature_generator(mv_feature)
        
        if self.training:
            mv_z_for_bit = self.add_noise(mv_z)
        else:
            mv_z_for_bit = mv_z_hat

        _, mv_z_likelihoods = self.get_z_bits(mv_z_for_bit, self.bit_estimator_z_mv, index)

        data = {
            "ref_mv_feature": mv_feature,
            "ref_mv_y": mv_y_hat,
        }
        
        return mv_hat, (mv_y_likelihoods, mv_z_likelihoods), data
    
    def update(self, force=False):
        self.entropy_coder = EntropyCoder_Inter(self.ec_thread, self.stream_part)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z_mv.update(force=force, entropy_coder=self.entropy_coder)
        scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table=scale_table, force=force)

    def compress(self, est_mv, aux_buf={}):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        q_index = aux_buf['index_list']
        mv_y_q_enc = self.mv_q_encoder[q_index]
        mv_y_q_dec = self.mv_q_decoder[q_index]
        
        conds, _ = self.feature_extractor(aux_buf["cond_input"], aux_buf["ref_mv_feature"]) if aux_buf["cond_input"] is not None else (None, None)
        
        mv_y = self.mv_encoder(est_mv, conds, mv_y_q_enc)
        
        mv_y_pad, slice_shape = self.pad_for_y(mv_y)
        mv_z = self.mv_hyper_prior_encoder(mv_y_pad)
        mv_z_hat = self.quant(mv_z)     # Output for decoder
        
        mv_params = self.mv_prior_param_decoder(mv_z_hat, aux_buf, slice_shape)

        mv_q_enc, mv_q_dec, mv_scales_hat, mv_means_hat = self.separate_prior(mv_params, is_video=True)
        mv_y = mv_y * mv_q_enc

        indexes = self.gaussian_conditional.build_indexes(mv_scales_hat)
        mv_y_strings = self.gaussian_conditional.compress(mv_y, indexes, means=mv_means_hat)
        mv_y_hat = self.gaussian_conditional.quantize(mv_y, "dequantize", means=mv_means_hat)

        mv_y_hat = mv_y_hat * mv_q_dec

        self.entropy_coder.reset()
        self.bit_estimator_z_mv.encode(mv_z_hat, 0)
        self.entropy_coder.flush()

        mv_z_strings = self.entropy_coder.get_encoded_stream()

        bit_stream = mv_y_strings + [mv_z_strings]
        
        mv_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec, conds[::-1] if conds is not None else None)

        mv_feature = self.feature_generator(mv_feature)

        data = {
            "ref_mv_feature": mv_feature,
            "ref_mv_y": mv_y_hat,
            "strings": bit_stream,
        }
        
        return mv_hat, data

    def decompress(self, strings, aux_buf={}):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        
        q_index = aux_buf['index_list']
        _ = self.mv_q_encoder[q_index]
        mv_y_q_dec = self.mv_q_decoder[q_index]

        sps = aux_buf['sps']
        
        conds, _ = self.feature_extractor(aux_buf["cond_input"], aux_buf["ref_mv_feature"]) if aux_buf["cond_input"] is not None else (None, None)
        
        if strings is not None:
            self.entropy_coder.set_stream(strings[-1])
        z_size = get_downsampled_shape(sps['height'], sps['width'], 64)
        y_height, y_width = get_downsampled_shape(sps['height'], sps['width'], 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        mv_z_hat = self.bit_estimator_z_mv.decode_stream(z_size, dtype, device, 0)
        
        mv_params = self.mv_prior_param_decoder(mv_z_hat, aux_buf, slice_shape)
        
        _, mv_q_dec, mv_scales_hat, mv_means_hat = self.separate_prior(mv_params, is_video=True)

        indexes = self.gaussian_conditional.build_indexes(mv_scales_hat)
        mv_y_hat = self.gaussian_conditional.decompress(strings[:-1], indexes, means=mv_means_hat)
        mv_y_hat = mv_y_hat * mv_q_dec
        
        mv_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec, conds[::-1] if conds is not None else None)
        
        mv_feature = self.feature_generator(mv_feature)

        data = {
            "ref_mv_feature": mv_feature,
            "ref_mv_y": mv_y_hat,
        }

        return mv_hat, data


class ContextualEncoder(nn.Module):
    def __init__(self, inplace=False, g_ch_1x=48, g_ch_2x=64, g_ch_4x=96, g_ch_8x=96, g_ch_16x=128,
                 Add_CTM=False, CTM_depth=2, CTM_head=8, CA_bias=False, 
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x + 3, g_ch_2x, 3, stride=2, padding=1)
        self.res1 = DepthConvBlock4(g_ch_2x * 2, g_ch_2x * 2, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_2x * 2, g_ch_4x, 3, stride=2, padding=1)
        self.res2 = DepthConvBlock4(g_ch_4x * 2, g_ch_4x * 2, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_4x * 2, g_ch_8x, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1)   # conv_for_CTM
        
        self.Add_CTM = Add_CTM
        if self.Add_CTM:
            self.CTM = CTM_CAB(dim=g_ch_16x,
                               depth=CTM_depth,
                               num_heads=CTM_head,
                               CA_bias=CA_bias,
                               )

    def forward(self, x, context1, context2, context3, quant_step):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        if quant_step is not None:
            feature = feature * quant_step
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        
        if self.Add_CTM:
        
            feature = self.CTM(feature)['output']
            
        return feature
    
class ContextualDecoder(nn.Module):
    def __init__(self, inplace=False, g_ch_2x=64, g_ch_4x=96, g_ch_8x=96, g_ch_16x=128,
                 Add_CTM=False, CTM_depth=2, CTM_head=8, CA_bias=False, 
                 ):
        super().__init__()
        self.Add_CTM = Add_CTM
        if self.Add_CTM:
            self.CTM = CTM_CAB(dim=g_ch_16x,
                               depth=CTM_depth,
                               num_heads=CTM_head,
                               CA_bias=CA_bias,
                               )
            
        self.up1 = subpel_conv3x3(g_ch_16x, g_ch_8x, 2)   # conv_for_CTM
        self.up2 = subpel_conv3x3(g_ch_8x, g_ch_4x, 2)
        self.res1 = DepthConvBlock4(g_ch_4x * 2, g_ch_4x * 2, inplace=inplace)
        self.up3 = subpel_conv3x3(g_ch_4x * 2, g_ch_2x, 2)
        self.res2 = DepthConvBlock4(g_ch_2x * 2, g_ch_2x * 2, inplace=inplace)
        self.up4 = subpel_conv3x3(g_ch_2x * 2, 32, 2)

    def forward(self, x, context2, context3, quant_step):
        if self.Add_CTM:
            x = self.CTM(x)['output']
            
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        if quant_step is not None:
            feature = feature * quant_step
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        
        return feature
    
class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=48, res_channel=32, inplace=False, g_ch_1x=48): # ctx_channel=g_ch_1x
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, g_ch_1x, 3, stride=1, padding=1)
        self.unet_1 = UNet2(g_ch_1x, g_ch_1x, inplace=inplace)
        self.unet_2 = UNet2(g_ch_1x, g_ch_1x, inplace=inplace)
        self.recon_conv = nn.Conv2d(g_ch_1x, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res, quant_step):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        
        if quant_step is not None:
            recon = self.recon_conv(feature * quant_step)
        else:
            recon = self.recon_conv(feature)
            
        return feature, recon
    
class Feature_Generator(nn.Module):
    def __init__(self, in_channel=48, out_channel=48):
        super().__init__()
        self.recon_conv = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)

    def forward(self, feature):
        feature = self.recon_conv(feature)

        return feature

class Inter(CompressionModel_Inter):
    def __init__(self, ec_thread=False, stream_part=1, inplace=False, g_ch_1x=48, g_ch_2x=64, g_ch_4x=96, g_ch_8x=96, g_ch_16x=128, g_ch_z=64, quant_mode='estUN_outR', 
                Pretrain=False, Add_Mask=True, no_implicit=False, buffering_type='hybrid', feature_channel=2, CTM_Params={}, no_context=False, variable_rate=True):
        super().__init__(y_distribution='laplace', z_channel=g_ch_z, mv_z_channel=64,
                         ec_thread=ec_thread, stream_part=stream_part)
        
        self.__delattr__('bit_estimator_z_mv')

        self.no_context = no_context
        self.Add_Mask = Add_Mask
        self.no_implicit = no_implicit
        self.buffering_type = buffering_type
        self.variable_rate = variable_rate
        self.Pretrain = Pretrain
        self.g_ch_16x = g_ch_16x
        self.g_ch_z = g_ch_z

        self.mc_generator = ResidualBlock_Inter(g_ch_1x, 3, inplace=inplace)

        if self.Pretrain:
            self.mc_generate_1 = nn.Conv2d(g_ch_1x, 3, 3, 1, 1)
            self.mc_generate_2 = UpsampleSBlock(g_ch_2x, 3)
            self.mc_generate_3 = UpsampleSBlock(g_ch_4x, g_ch_2x)
            self.mc_generate_3_2 = UpsampleSBlock(g_ch_2x, 3)
            
        if self.Add_Mask:
            self.MaskGenerator = MaskGeneration(in_channels=5, out_channels=1)
        

        self.feature_adaptor_I = nn.Conv2d(3, g_ch_1x, 3, stride=1, padding=1)
        
        if self.buffering_type == 'hybrid':
            self.feature_adaptor = nn.ModuleList([nn.Conv2d(feature_channel+3, g_ch_1x, 1) for _ in range(3)])
        
        self.feature_extractor = FeatureExtractor_Inter(inplace, g_ch_1x, g_ch_2x, g_ch_4x)
        self.context_fusion_net = MultiScaleContextFusion(inplace, g_ch_1x, g_ch_2x, g_ch_4x)
            
        self.contextual_encoder = ContextualEncoder(inplace, g_ch_1x, g_ch_2x, g_ch_4x, g_ch_8x, g_ch_16x, 
                                                    **CTM_Params)

        self.contextual_hyper_prior_encoder = nn.Sequential(
            DepthConvBlock4(g_ch_16x, g_ch_z, inplace=inplace),
            nn.Conv2d(g_ch_z, g_ch_z, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(g_ch_z, g_ch_z, 3, stride=2, padding=1),
        )
        self.contextual_hyper_prior_decoder = nn.Sequential(
            ResidualBlockUpsample_Intra(g_ch_z, g_ch_z, 2, inplace=inplace),
            ResidualBlockUpsample_Intra(g_ch_z, g_ch_z, 2, inplace=inplace),
            DepthConvBlock4(g_ch_z, g_ch_16x),
        )

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(g_ch_4x, g_ch_8x, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1),
        )

        self.y_prior_fusion_adaptor_0 = DepthConvBlock_Inter(g_ch_16x * 2, g_ch_16x * 3,
                                                              inplace=inplace)
        if not self.no_implicit:
            self.y_prior_fusion_adaptor_1 = DepthConvBlock_Inter(g_ch_16x * 3, g_ch_16x * 3,
                                                              inplace=inplace)

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock_Inter(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock_Inter(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
        )

        if self.no_context:
            self.gaussian_conditional = GaussianConditional(None, quant_mode=quant_mode)
        else:
            self.context_model = Checkerboard(g_ch_16x, quant_mode=quant_mode)

        self.contextual_decoder = ContextualDecoder(inplace, g_ch_2x, g_ch_4x, g_ch_8x, g_ch_16x,
                                                    **CTM_Params)
        self.recon_generation_net = ReconGeneration(ctx_channel=g_ch_1x, inplace=inplace, g_ch_1x=g_ch_1x)

        if self.buffering_type == 'hybrid':
            self.frame_feature_generator = Feature_Generator(g_ch_1x, out_channel=feature_channel)

        if self.variable_rate:
            self.q_encoder = nn.Parameter(torch.ones((64, g_ch_2x * 2, 1, 1)))
            self.q_decoder = nn.Parameter(torch.ones((64, g_ch_2x, 1, 1)))
            self.q_feature = nn.Parameter(torch.ones((64, g_ch_1x, 1, 1)))
            self.q_recon = nn.Parameter(torch.ones((64, g_ch_1x, 1, 1)))
    
    def multi_scale_feature_extractor(self, dpb, fa_idx):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(dpb["ref_frame"])
        
        else:
            feature = self.feature_adaptor[fa_idx](torch.cat((dpb["ref_feature"], dpb["ref_frame"]), dim=1))
            
        if self.variable_rate:
            feature = self.feature_extractor(feature, dpb["q_feature"])
        else:
            feature = self.feature_extractor(feature)

        return feature
    
    def motion_compensation(self, dpb, mv, fa_idx, Pretrain=False):
        warpframe = block_mc_func(dpb["ref_frame"], mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2

        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb, fa_idx)
        
        context1 = block_mc_func(ref_feature1, mv)
        context2 = block_mc_func(ref_feature2, mv2)
        context3 = block_mc_func(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)

        mc_frame = self.mc_generator(context1)
        if Pretrain:
            return self.mc_generate_1(context1), self.mc_generate_2(context2), self.mc_generate_3_2(self.mc_generate_3(context3)), warpframe, mc_frame
        else:
            return context1, context2, context3, warpframe, mc_frame
    
    def contextual_prior_param_decoder(self, z_hat, dpb, context3, slice_shape=None):
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        hierarchical_params = self.slice_to_y(hierarchical_params, slice_shape)
        temporal_params = self.temporal_prior_encoder(context3)
        ref_y = dpb["ref_y"]
        if self.no_implicit or ref_y is None:
            params = torch.cat((temporal_params, hierarchical_params), dim=1)
            params = self.y_prior_fusion_adaptor_0(params)
        else:
            params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)
            params = self.y_prior_fusion_adaptor_1(params)
        params = self.y_prior_fusion(params)
        return params
    
    def get_recon_and_feature(self, y_hat, context1, context2, context3, y_q_dec, q_recon):
        recon_image_feature = self.contextual_decoder(y_hat, context2, context3, y_q_dec)
        feature, x_hat = self.recon_generation_net(recon_image_feature, context1, q_recon)
        x_hat = x_hat.clamp_(0, 1)
        return x_hat, feature
    
    def quant(self, x):
        if self.training:
            n = torch.round(x) - x
            n = n.clone().detach()
            return x + n
        else:
            return torch.round(x)

    def set_noise_level(self, noise_level):
        self.noise_level = noise_level

    def add_noise(self, x):
        noise = torch.nn.init.uniform_(torch.zeros_like(x), -self.noise_level, self.noise_level)
        noise = noise.clone().detach()
        return x + noise
    
    def forward(self, x, aux_buf={}):
        if not self.variable_rate:
            y_q_enc = y_q_dec = aux_buf['q_feature'] = q_recon = None
        else:
            q_index = aux_buf['index_list']
            y_q_enc = self.q_encoder[q_index]
            y_q_dec = self.q_decoder[q_index]
            q_feature = self.q_feature[q_index]
            q_recon = self.q_recon[q_index]
            aux_buf['q_feature'] = q_feature
        
        if self.no_implicit:
            fa_idx = 0
            aux_buf["ref_y"] = None
            aux_buf["ref_feature"] = None
        else:
            fa_idx = aux_buf['fa_idx'] # frame_type_id
        
        index = self.get_index_tensor(0, x.device)

        context1, context2, context3, _, mc_frame = self.motion_compensation(aux_buf, aux_buf["mv_hat"], fa_idx)
        
        if not self.Add_Mask:
            prediction_signal = mc_frame
        else:
            mask = self.MaskGenerator(torch.cat([aux_buf["mv_hat"], mc_frame], dim=1))
            prediction_signal = mask * mc_frame
        
        res = x - prediction_signal

        if "compute_decode_macs" in aux_buf and aux_buf["compute_decode_macs"]:
            y_shape = [1, self.g_ch_16x, x.shape[-2] // 16, x.shape[-1] // 16]
            z_shape = [1, self.g_ch_z, x.shape[-2] // 64, x.shape[-1] // 64]
            y = torch.zeros(y_shape).to(x.device)
            y_pad, slice_shape = self.pad_for_y(y)
            z_hat = torch.zeros(z_shape).to(x.device)
        else:

            y = self.contextual_encoder(res, context1, context2, context3, y_q_enc)
            y_pad, slice_shape = self.pad_for_y(y)
            z = self.contextual_hyper_prior_encoder(y_pad)
            z_hat = self.quant(z)

        # hyperprior decoding
        params = self.contextual_prior_param_decoder(z_hat, aux_buf, context3, slice_shape)
        
        if self.no_context:
            q_enc, q_dec, scales_hat, means_hat = self.separate_prior(params, True)
            y = y * q_enc
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)
            y_hat = y_hat * q_dec

            # y_q = self.quant(y-means_hat)
            # y_for_bit = y_q
            # bits_y, _ = self.get_y_laplace_bits(y_for_bit, scales_hat)

        else:
            q_enc, q_dec, scales, means = self.separate_prior(params, True)
            y = y * q_enc
            
            y_hat, y_likelihoods, entropy_info = self.context_model(y, torch.cat((scales, means), dim=1))
            means_hat, scales_hat = entropy_info['mean'], entropy_info['scale']
            
            y_hat = y_hat * q_dec

        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec, q_recon)

        if self.buffering_type == 'hybrid':
            frame_feature = self.frame_feature_generator(feature)

        rec_frame = x_hat + prediction_signal

        if self.training:
            z_for_bit = self.add_noise(z)
        else:
            z_for_bit = z_hat

        _, z_likelihoods = self.get_z_bits(z_for_bit, self.bit_estimator_z, index)


        data = {
            "mean": means_hat,
            "scale": scales_hat, 
            "res": res,
            "res_hat": x_hat,
            "mc_frame": mc_frame
        }

        if not self.no_implicit:
            data.update({
                "ref_y": y_hat,
                "ref_feature": frame_feature
            })
        if self.Add_Mask:
            data.update({
                "mask": mask
            })

        return rec_frame, (y_likelihoods, z_likelihoods), data
    
    def update(self, force=False):
        self.entropy_coder = EntropyCoder_Inter(self.ec_thread, self.stream_part)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)
        scale_table = get_scale_table()
        self.context_model.update(scale_table=scale_table, force=force)

    def compress(self, x, aux_buf={}):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x

        q_index = aux_buf['index_list']
        y_q_enc = self.q_encoder[q_index]
        y_q_dec = self.q_decoder[q_index]
        q_feature = self.q_feature[q_index]
        q_recon = self.q_recon[q_index]
        aux_buf['q_feature'] = q_feature
        
        fa_idx = aux_buf['fa_idx'] # frame_type_id

        context1, context2, context3, _, mc_frame = self.motion_compensation(aux_buf, aux_buf["mv_hat"], fa_idx)
        
        mask = self.MaskGenerator(torch.cat([aux_buf["mv_hat"], mc_frame], dim=1))
        prediction_signal = mask * mc_frame
        
        res = x - prediction_signal
        
        y = self.contextual_encoder(res, context1, context2, context3, y_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.contextual_hyper_prior_encoder(y_pad)
        z_hat = self.quant(z)
        
        # hyperprior decoding
        params = self.contextual_prior_param_decoder(z_hat, aux_buf, context3, slice_shape)

        q_enc, q_dec, scales, means = self.separate_prior(params, True)
        y = y * q_enc
        
        y_hat, y_strings, entropy_info = self.context_model.compress(y, torch.cat((scales, means), dim=1))
        means_hat, scales_hat = entropy_info['mean'], entropy_info['scale']

        y_hat = y_hat * q_dec

        self.entropy_coder.reset()
        self.bit_estimator_z.encode(z_hat, 0)
        self.entropy_coder.flush()

        y_strings = list(flatten_strings_list(y_strings))
        z_strings = self.entropy_coder.get_encoded_stream()

        bit_stream = y_strings + [z_strings]
            
        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec, q_recon)

        frame_feature = self.frame_feature_generator(feature)

        rec_frame = x_hat + prediction_signal
        
        data = {
            "strings": bit_stream,
            "ref_y": y_hat,
            "ref_feature": frame_feature,
            "res": res,
            "res_hat": x_hat,
            "mc_frame": mc_frame,
            "mask": mask
        }

        return rec_frame, data

    def decompress(self, strings, aux_buf={}):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        q_index = aux_buf['index_list']
        _= self.q_encoder[q_index]
        y_q_dec = self.q_decoder[q_index]
        q_feature = self.q_feature[q_index]
        q_recon = self.q_recon[q_index]
        aux_buf['q_feature'] = q_feature

        fa_idx = aux_buf['fa_idx'] # frame_type_id

        sps = aux_buf['sps']

        context1, context2, context3, _, mc_frame = self.motion_compensation(aux_buf, aux_buf["mv_hat"], fa_idx)

        mask = self.MaskGenerator(torch.cat([aux_buf["mv_hat"], mc_frame], dim=1))
        prediction_signal = mask * mc_frame
        
        if strings is not None:
            self.entropy_coder.set_stream(strings[-1])
        z_size = get_downsampled_shape(sps['height'], sps['width'], 64)
        y_height, y_width = get_downsampled_shape(sps['height'], sps['width'], 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device, 0)

        params = self.contextual_prior_param_decoder(z_hat, aux_buf, context3, slice_shape)

        _, q_dec, scales, means = self.separate_prior(params, True)

        y_strings = []
        for i in range(len(strings) - 1):
            y_strings.append([strings[i]])

        y_hat = self.context_model.decompress(y_strings, torch.cat((scales, means), dim=1))
        y_hat = y_hat * q_dec
        
        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec, q_recon)

        frame_feature = self.frame_feature_generator(feature)

        rec_frame = x_hat + prediction_signal

        data = {
            "ref_y": y_hat,
            "ref_feature": frame_feature,
            "res_hat": x_hat,
            "mc_frame": mc_frame,
            "mask": mask
        }
        
        return rec_frame, data


__CODER_TYPES__ = {'Motion'                          : Motion,
                   'Inter'                           : Inter,
                   }
