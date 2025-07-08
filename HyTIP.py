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

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms

from advance_model import *
from compressai.entropy_models import EntropyBottleneck
from compressai.models import __CODER_TYPES__
from compressai.models.utils import get_padding_size
from dataloader import VideoTestData, SEQUENCES
from trainer import Trainer
from util.estimate_bpp import estimate_bpp
from util.psnr import mse2psnr
from util.sampler import Resampler
from util.stream_helper_SPS import NalType, write_sps, read_header, read_sps_remaining, write_ip_segment, read_ip_remaining_segment
from util.seed import seed_everything
from pytorch_msssim import MS_SSIM as MS_SSIM_PyTorch
from util.vision import PlotFlow, PlotHeatMap, save_image
from util.YUV_Transformation import RGB2YUV420_v2, rgb2ycbcr420
from util.bit_depth import write_yuv_args
from loguru import logger

plot_flow = PlotFlow().cuda()
plot_bitalloc = PlotHeatMap("RB").cuda()

lmda = {1: 0.0018, 2: 0.0035, 3: 0.0067, 4: 0.0130, 
        5: 0.0250, 6: 0.0483, 7: 0.0932, 8: 0.1800}

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

class CompressesModel(nn.Module):
    """Basic Compress Model"""

    def __init__(self):
        super(CompressesModel, self).__init__()

    def named_main_parameters(self, prefix='', include_module_name=None, exclude_module_name=None):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' not in name:
                if include_module_name is not None:
                    assert isinstance(include_module_name, str) or isinstance(include_module_name, list), ValueError

                    if isinstance(include_module_name, str) and (include_module_name in name):
                        yield (name, param)
                    elif isinstance(include_module_name, list) and any([_n in name for _n in include_module_name]):
                        yield (name, param)
                elif exclude_module_name is not None:
                    assert isinstance(exclude_module_name, str) or isinstance(exclude_module_name, list), ValueError
                    
                    if isinstance(exclude_module_name, str) and not (exclude_module_name in name):
                        yield (name, param)
                    elif isinstance(exclude_module_name, list) and not any([_n in name for _n in exclude_module_name]):
                        yield (name, param)
                else:
                    yield (name, param)

    def include_main_parameters(self, include_module_name=None):
        for _, param in self.named_main_parameters(include_module_name=include_module_name):
            yield param

    def exclude_main_parameters(self, exclude_module_name=None):
        for _, param in self.named_main_parameters(exclude_module_name=exclude_module_name):
            yield param

    def main_parameters(self):
        for _, param in self.named_main_parameters():
            yield param

    def named_aux_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' in name:
                yield (name, param)

    def aux_parameters(self):
        for _, param in self.named_aux_parameters():
            yield param

    def aux_loss(self):
        aux_loss = []
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                aux_loss.append(m.loss())

        return torch.stack(aux_loss).mean()

class Pframe(CompressesModel):
    def __init__(self, args, cond_mo_coder, res_coder):
        super(Pframe, self).__init__()
        self.args = args
        if self.args.ssim:
            self.criterion = MS_SSIM_PyTorch(data_range=1., size_average=False).cuda()
        else:
            self.criterion = nn.MSELoss(reduction='none')

        if not self.args.compute_model_size:
            self.if_model = Iframe_Coder(args.if_coder, args.iframe_quality, args.ssim)
            self.if_model.eval()

        self.MENet = MENet(args.MENet)
        self.CondMotion = MotionCoder(cond_mo_coder)
        self.Resampler = Resampler()
        self.Residual = InterCoder(res_coder)

        self.ref_flow = None
        self.implicit_buffer = {
            "ref_feature": None,
            "ref_y": None,
            "ref_mv_feature": None,
            "ref_mv_y": None,
        }

    def update(self):
        self.if_model.update()
        self.CondMotion.update()
        self.Residual.update()

    def UP_Sample(self, tensor, factor=2, mode="nearest"):
        return F.interpolate(tensor, scale_factor=factor, mode=mode)

    @torch.no_grad()
    def RGB2YUV444(self, batch, up_mode='nearest'):
        assert len(batch.shape) == 5, "This may not be a video sequence !!!"
        YUV444 = []
        for i in range(batch.shape[1]):
            if self.args.color_transform == 'BT709':
                y, uv = rgb2ycbcr420(batch[:,i])
            elif self.args.color_transform == 'BT601':
                y, uv = RGB2YUV420_v2(batch[:,i])
            yuv = torch.cat([y, self.UP_Sample(uv, factor=2, mode=up_mode)], dim=1)
            YUV444.append(yuv)
        batch = torch.stack(YUV444).permute(1, 0, 2, 3, 4)
        return batch
                
    def motion_forward(self, ref_frame, coding_frame, predict=False, RNN=True, index_list_tensor=None):
        if not self.args.compute_decode_macs:
            flow = self.MENet(ref_frame, coding_frame)
        else:
            flow = torch.zeros_like(ref_frame[:,0:2,:,:])

        aux_buf = {
            "index_list": index_list_tensor,
            'cond_input': self.ref_flow if predict else None,
            'temporal_input': torch.cat([ref_frame, self.ref_flow], dim=1) if predict else None,
            'ref_mv_feature': self.implicit_buffer["ref_mv_feature"],
            'ref_mv_y': self.implicit_buffer["ref_mv_y"],
            'compute_decode_macs': self.args.compute_decode_macs,
        }
        flow_hat, likelihood_m, data = self.CondMotion(flow, aux_buf=aux_buf)
        
        self.implicit_buffer["ref_mv_feature"] = data["ref_mv_feature"]
        self.implicit_buffer["ref_mv_y"] = data["ref_mv_y"]
        del data
        
        self.ref_flow = flow_hat # if RNN else flow_hat.detach()

        warped_frame = self.Resampler(ref_frame, flow_hat)
        
        m_info = {'likelihood_m': likelihood_m, 
                  'flow': flow, 'flow_hat': flow_hat,
                  'warped_frame': warped_frame}

        return likelihood_m, m_info

    def forward(self, ref_frame, coding_frame, predict, RNN=True, frame_idx=None, index_list=None):
        
        index_list_tensor = torch.tensor(index_list)

        fa_idx = self.args.index_map[frame_idx % self.args.rate_gop_size]
        likelihood_m, m_info = self.motion_forward(ref_frame, coding_frame, predict, RNN, index_list_tensor=index_list_tensor)

        aux_buf = {
            "index_list": index_list_tensor,
            "fa_idx": fa_idx,
            "mv_hat": m_info['flow_hat'],
            "ref_frame": ref_frame,
            "ref_feature": self.implicit_buffer["ref_feature"],
            "ref_y": self.implicit_buffer["ref_y"],
        }
        if self.args.compute_decode_macs:
            aux_buf.update({'compute_decode_macs': True})
        
        rec_frame, likelihood_r, info = self.Residual(coding_frame, aux_buf=aux_buf)

        self.implicit_buffer["ref_feature"] = info["ref_feature"]
        self.implicit_buffer["ref_y"] = info["ref_y"]
        
        likelihoods = likelihood_m + likelihood_r

        r_info = {
            'res'     : info['res'],
            'res_hat' : info['res_hat'],
            'mc_frame': info['mc_frame'],
            'mask'    : info['mask']
        }

        
        return rec_frame, likelihoods, m_info, r_info
    
    @torch.no_grad()
    def compress(self, ref_frame, coding_frame, predict, frame_idx=None, index_list=None):
        # device = next(self.parameters()).device
        index_list_tensor = torch.tensor(index_list)

        flow = self.MENet(ref_frame, coding_frame)

        aux_buf = {
            "index_list": index_list_tensor,
            'cond_input': self.ref_flow if predict else None,
            'temporal_input': torch.cat([ref_frame, self.ref_flow], dim=1) if predict else None,
            'ref_mv_feature': self.implicit_buffer["ref_mv_feature"],
            'ref_mv_y': self.implicit_buffer["ref_mv_y"]
        }

        flow_hat, m_info = self.CondMotion.compress(flow, aux_buf=aux_buf)

        m_strings = m_info['strings']

        self.implicit_buffer["ref_mv_feature"] = m_info["ref_mv_feature"]
        self.implicit_buffer["ref_mv_y"] = m_info["ref_mv_y"]
        del m_info

        self.ref_flow = flow_hat

        warped_frame = self.Resampler(ref_frame, flow_hat)

        fa_idx = self.args.index_map[frame_idx % self.args.rate_gop_size]

        aux_buf = {
            "index_list": index_list_tensor,
            "fa_idx": fa_idx,
            "mv_hat": flow_hat,
            "ref_frame": ref_frame,
            "ref_feature": self.implicit_buffer["ref_feature"],
            "ref_y": self.implicit_buffer["ref_y"],
        }

        rec_frame, r_info = self.Residual.compress(coding_frame, aux_buf=aux_buf)

        self.implicit_buffer["ref_feature"] = r_info["ref_feature"]
        self.implicit_buffer["ref_y"] = r_info["ref_y"]

        r_strings = r_info['strings']

        sps = {
            'sps_id': -1,
            'height': coding_frame.size(2),
            'width': coding_frame.size(3),
            'qp': self.args.QP,
            'fa_idx': fa_idx,
        }

        info = {
            'flow_hat': flow_hat,
            'warped_frame': warped_frame,
            'mc_frame': r_info['mc_frame'],
            'sps': sps,
        }
        
        del r_info

        return rec_frame, m_strings + r_strings, info
    
    @torch.no_grad()
    def decompress(self, ref_frame, strings, predict, frame_idx=None, sps=None, index_list=None):
        device = next(self.parameters()).device
        m_strings = strings[:2]
        r_strings = strings[2:]
        index_list_tensor = torch.tensor(index_list)

        aux_buf = {
            "index_list": index_list_tensor,
            'cond_input': self.ref_flow if predict else None,
            'temporal_input': torch.cat([ref_frame, self.ref_flow], dim=1) if predict else None,
            'ref_mv_feature': self.implicit_buffer["ref_mv_feature"],
            'ref_mv_y': self.implicit_buffer["ref_mv_y"],
            'sps': sps
        }
        
        flow_hat, m_info = self.CondMotion.decompress(m_strings, aux_buf=aux_buf)

        self.implicit_buffer["ref_mv_feature"] = m_info["ref_mv_feature"]
        self.implicit_buffer["ref_mv_y"] = m_info["ref_mv_y"]
        del m_info

        self.ref_flow = flow_hat

        warped_frame = self.Resampler(ref_frame, flow_hat)

        fa_idx = sps['fa_idx']

        aux_buf = {
            "index_list": index_list_tensor,
            "fa_idx": fa_idx,
            "mv_hat": flow_hat,
            "ref_frame": ref_frame,
            "ref_feature": self.implicit_buffer["ref_feature"],
            "ref_y": self.implicit_buffer["ref_y"],
            'sps': sps
        }

        rec_frame, r_info = self.Residual.decompress(r_strings, aux_buf=aux_buf)

        self.implicit_buffer["ref_feature"] = r_info["ref_feature"]
        self.implicit_buffer["ref_y"] = r_info["ref_y"]

        info = {
            'flow_hat': flow_hat,
            'warped_frame': warped_frame,
            'mc_frame': r_info['mc_frame'],
        }
        
        del r_info

        return rec_frame, info

    @torch.no_grad()
    def test_step(self, batch):
        if self.args.ssim: 
            dist_metric = 'MS-SSIM'
        else: 
            dist_metric = 'PSNR'

        metrics_name = [dist_metric, 'Rate', 'Mo_Rate', f'MC-{dist_metric}', f'Warped-{dist_metric}', 'Mo_Ratio', 'Res_Ratio']
        metrics = {}
        for m in metrics_name:
            metrics[m] = []

        log_list = []

        dataset_name, seq_name, batch, frame_id_start = batch
        device = next(self.parameters()).device
        batch = batch.to(device)
        if self.args.test_crop:
            H, W = batch.shape[-2:]
            batch = transforms.CenterCrop((self.args.crop_factor * (H // self.args.crop_factor), self.args.crop_factor * (W // self.args.crop_factor)))(batch)

        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch.size(1)

        TO_VISUALIZE_Flag = self.args.visualize
        if not self.args.compute_macs and TO_VISUALIZE_Flag:
            # os.makedirs(self.args.save_dir + f'/{seq_name}/flow_hat', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/warped_frame', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/motion_mean', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/motion_scale', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/residual_mean', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/residual_scale', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/likelihood_r', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/likelihood_m', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/mask', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/res', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/res_hat', exist_ok=True)

        for frame_idx in range(gop_size):
            TO_VISUALIZE = TO_VISUALIZE_Flag and frame_id_start == 1 and frame_idx < 8 and seq_name in ['BasketballDrive', 'Kimono1', 'HoneyBee', 'Jockey']
            SCENE_CUT = {
                'Kimono1': [141], 'Kimono1_10b_rgb': [141],
                'videoSRC20': [43, 81, 106], 'videoSRC21': [49, 64, 90], 'videoSRC25': [18, 84], 'videoSRC26': [62], 'videoSRC27': [40], 'videoSRC29': [71],
                "Fallout4": [560]
            }

            coding_frame = batch[:, frame_idx]

            # pad if necessary
            height, width = coding_frame.size(2), coding_frame.size(3)
            padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, self.args.align_factor)
            coding_frame_pad = F.pad(coding_frame, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

            if not self.training and self.args.reset_interval > 0 and frame_idx % self.args.reset_interval == 1:
                self.implicit_buffer["ref_feature"] = None

            # I frame
            if frame_idx == 0 or (self.args.remove_scene_cut and seq_name in SCENE_CUT and frame_id_start + frame_idx in SCENE_CUT[seq_name]):

                self.implicit_buffer = {
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_feature": None,
                    "ref_mv_y": None,
                }

                rec_frame, likelihoods = self.if_model(coding_frame_pad, torch.tensor([self.args.iframe_quality]))
                rec_frame = rec_frame.clamp(0, 1)
                x_hat = F.pad(rec_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

                r_y = estimate_bpp(likelihoods[0], input=coding_frame).mean().item()
                r_z = estimate_bpp(likelihoods[1], input=coding_frame).mean().item()
                rate = r_y + r_z

                if self.args.ssim: 
                    distortion = self.criterion(x_hat, coding_frame).mean().item()
                else: 
                    mse = self.criterion(x_hat, coding_frame).mean().item()
                    distortion = mse2psnr(mse)

                log_list.append({f'{dist_metric}': distortion, 'Rate': rate})
            
            # P frame
            else:
                if self.args.compute_macs and frame_idx != 1:
                    def dummy_inputs(shape):
                        inputs = torch.ones(shape).cuda()
                        return {
                            'ref_frame'     : inputs, 
                            'coding_frame'  : inputs, 
                            'predict'       : True,
                            'RNN'           : False,
                            'frame_idx'     : 2,
                            'index_list'    : [self.args.QP],
                        }
                    
                    macs, _ = get_model_complexity_info(self, tuple(ref_frame.shape), as_strings=False, input_constructor=dummy_inputs)
                    # print(macs)
                    print("========================================")
                    print("MACs:", macs)
                    print(macs / 1e9, "GMac")
                    print(macs / ref_frame.shape[2] / ref_frame.shape[3] / 1e3, "kMACs/pixel")
                    print("ref_frame shape:", ref_frame.shape)
                    exit()

                predict = False if (self.args.remove_scene_cut and (seq_name in SCENE_CUT) and ((frame_id_start + frame_idx - 1) in SCENE_CUT[seq_name])) else (frame_idx!=1)
                rec_frame, likelihoods, m_info, r_info = self(ref_frame, coding_frame_pad, predict, RNN=False, frame_idx=frame_idx,
                                                              index_list=[self.args.QP])
                rec_frame = rec_frame.clamp(0, 1)
                x_hat = F.pad(rec_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

                mc_frame = r_info['mc_frame'].clamp(0, 1)
                mc_frame = F.pad(mc_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
                warped_frame = m_info['warped_frame']
                warped_frame = F.pad(warped_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
                
                m_y = estimate_bpp(likelihoods[0], input=coding_frame).cpu().item()
                m_z = estimate_bpp(likelihoods[1], input=coding_frame).cpu().item()
                r_y = estimate_bpp(likelihoods[2], input=coding_frame).cpu().item()
                r_z = estimate_bpp(likelihoods[3], input=coding_frame).cpu().item()

                m_rate = m_y + m_z
                r_rate = r_y + r_z
                rate = m_rate + r_rate

                if self.args.ssim:
                    distortion = self.criterion(x_hat, coding_frame).mean().item()
                    mc_distortion = self.criterion(mc_frame, coding_frame).mean().item()
                    warped_distortion = self.criterion(warped_frame, coding_frame).mean().item()
                else: 
                    mse = self.criterion(x_hat, coding_frame).mean().item()
                    distortion = mse2psnr(mse)

                    mc_mse = self.criterion(mc_frame, coding_frame).mean().item()
                    mc_distortion = mse2psnr(mc_mse)

                    warped_mse = self.criterion(warped_frame, coding_frame).mean().item()
                    warped_distortion = mse2psnr(warped_mse)

                if TO_VISUALIZE:
                    # flow_map = plot_flow(m_info['flow_hat'])
                    # save_image(flow_map,
                    #            self.args.save_dir + f'/{seq_name}/flow_hat/'
                    #                                 f'frame_{int(frame_id_start + frame_idx)}_flow.png', nrow=1)
                    save_image(coding_frame[0], 
                               self.args.save_dir + f'/{seq_name}/gt_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    # save_image(mc_frame[0], 
                    #            self.args.save_dir + f'/{seq_name}/mc_frame/'
                    #                                 f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(warped_frame[0], 
                               self.args.save_dir + f'/{seq_name}/warped_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(rec_frame[0], 
                               self.args.save_dir + f'/{seq_name}/rec_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')

                    # if frame_idx != 1:
                    #     motion_mean = make_grid(torch.transpose(m_info['mean'], 0, 1), nrow=8)
                    #     save_image(motion_mean, self.args.save_dir + f'/{seq_name}/motion_mean/frame_{int(frame_id_start + frame_idx)}.png')

                    #     motion_scale = make_grid(torch.transpose(m_info['scale'], 0, 1), nrow=8)
                    #     save_image(motion_scale, self.args.save_dir + f'/{seq_name}/motion_scale/frame_{int(frame_id_start + frame_idx)}.png')

                    # residual_mean = make_grid(torch.transpose(r_info['mean'], 0, 1), nrow=8)
                    # save_image(residual_mean, self.args.save_dir + f'/{seq_name}/residual_mean/frame_{int(frame_id_start + frame_idx)}.png')

                    # residual_scale = make_grid(torch.transpose(r_info['scale'], 0, 1), nrow=8)
                    # save_image(residual_scale, self.args.save_dir + f'/{seq_name}/residual_scale/frame_{int(frame_id_start + frame_idx)}.png')

                    # cm = plt.get_cmap('hot')
                    # lll = lower_bound(likelihoods[0], 1e-9).log() / -np.log(2.)
                    # rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                    # plt.imshow(rate_map)
                    # plt.savefig(self.args.save_dir + f'/{seq_name}/likelihood_m/frame_{int(frame_id_start + frame_idx)}_y.png')
                    # plt.close()

                    # cm = plt.get_cmap('hot')
                    # lll = lower_bound(likelihoods[1], 1e-9).log() / -np.log(2.)
                    # rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                    # plt.imshow(rate_map)
                    # plt.savefig(self.args.save_dir + f'/{seq_name}/likelihood_m/frame_{int(frame_id_start + frame_idx)}_z.png')
                    # plt.close()

                    # cm = plt.get_cmap('hot')
                    # lll = lower_bound(likelihoods[2], 1e-9).log() / -np.log(2.)
                    # rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                    # plt.imshow(rate_map)
                    # plt.savefig(self.args.save_dir + f'/{seq_name}/likelihood_r/frame_{int(frame_id_start + frame_idx)}_y.png')
                    # plt.close()
                    
                    # cm = plt.get_cmap('hot')
                    # lll = lower_bound(likelihoods[3], 1e-9).log() / -np.log(2.)
                    # rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                    # plt.imshow(rate_map)
                    # plt.savefig(self.args.save_dir + f'/{seq_name}/likelihood_r/frame_{int(frame_id_start + frame_idx)}_z.png')
                    # plt.close()

                    # m_lower = np.quantile(r_info['mask'].cpu().numpy()[0].squeeze(0), 0.00005)
                    # m_upper = np.quantile(r_info['mask'].cpu().numpy()[0].squeeze(0), 0.99995)
                    # plt.imshow(r_info['mask'].cpu().numpy()[0].squeeze(0), cmap='Oranges_r', vmin=m_lower, vmax=m_upper)
                    # plt.axis('off')
                    # plt.colorbar(shrink=0.55, pad=0.01)
                    # plt.savefig(self.args.save_dir + f'/{seq_name}/mask/'
                    #             f'frame_{int(frame_id_start + frame_idx)}.png',
                    #             bbox_inches='tight', pad_inches=0.01)
                    # plt.close()

                    # save_image(r_info['res'][0], 
                    #            self.args.save_dir + f'/{seq_name}/res/'
                    #                                 f'frame_{int(frame_id_start + frame_idx)}.png')
                    
                    # save_image(r_info['res_hat'][0], 
                    #            self.args.save_dir + f'/{seq_name}/res_hat/'
                    #                                 f'frame_{int(frame_id_start + frame_idx)}.png')
                    
                    # del flow_map, rate_map, lll, cm

                del r_info, m_info, mc_frame, warped_frame, coding_frame
                log_list.append({dist_metric: distortion, 'Rate': rate, f'MC-{dist_metric}': mc_distortion, f'Warped-{dist_metric}': warped_distortion,
                                 'my': m_y, 
                                 'mz': m_z,
                                 'ry': r_y, 
                                 'rz': r_z})

                metrics['Mo_Rate'].append(m_rate)
                metrics[f'MC-{dist_metric}'].append(mc_distortion)
                metrics[f'Warped-{dist_metric}'].append(warped_distortion)
                metrics['Mo_Ratio'].append(m_rate/rate)
                metrics['Res_Ratio'].append(r_rate/rate)

            del likelihoods
            metrics[f'{dist_metric}'].append(distortion)
            metrics['Rate'].append(rate)

            ref_frame = rec_frame

        del rec_frame, ref_frame
        for m in metrics_name:
            metrics[m] = np.mean(metrics[m])

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 'metrics': metrics, 'log_list': log_list}
        return logs

    @torch.no_grad()
    def test_epoch_end(self, outputs):
        metrics_name = list(outputs[0]['metrics'].keys())  # Get all metrics' names

        rd_dict = {}

        single_seq_logs = {}
        for metrics in metrics_name:
            single_seq_logs[metrics] = {}

        single_seq_logs['LOG'] = {}
        single_seq_logs['GOP'] = {}  # Will not be printed currently
        single_seq_logs['Seq_Names'] = []

        for logs in outputs:
            dataset_name = logs['dataset_name']
            seq_name = logs['seq_name']

            if not (dataset_name in rd_dict.keys()):
                rd_dict[dataset_name] = {}
                
                for metrics in metrics_name:
                    rd_dict[dataset_name][metrics] = []

            for metrics in logs['metrics'].keys():
                rd_dict[dataset_name][metrics].append(logs['metrics'][metrics])

            # Initialize
            if seq_name not in single_seq_logs['Seq_Names']:
                single_seq_logs['Seq_Names'].append(seq_name)
                for metrics in metrics_name:
                    single_seq_logs[metrics][seq_name] = []
                single_seq_logs['LOG'][seq_name] = []
                single_seq_logs['GOP'][seq_name] = []

            # Collect metrics logs
            for metrics in metrics_name:
                single_seq_logs[metrics][seq_name].append(logs['metrics'][metrics])
            single_seq_logs['LOG'][seq_name].extend(logs['log_list'])
            single_seq_logs['GOP'][seq_name] = len(logs['log_list'])

        os.makedirs(self.args.save_dir + f'/report', exist_ok=True)

        for seq_name, log_list in single_seq_logs['LOG'].items():
            seq_report_name = f'{seq_name}'
            if self.args.compress:
                seq_report_name = f'{seq_report_name}_enc'
            elif self.args.decompress:
                seq_report_name = f'{seq_report_name}_dec'

            if self.args.full_seq:
                seq_report_name = f'{seq_report_name}_FullSeq'
            
            csv_report_path = self.args.save_dir + f'/report/{seq_report_name}.csv'
            
            with open(csv_report_path, 'w', newline='') as report:
                writer = csv.writer(report, delimiter=',')
                columns = ['frame'] + list(log_list[1].keys())
                writer.writerow(columns)

                for idx in range(len(log_list)):
                    writer.writerow([f'frame_{idx + 1}'] + list(log_list[idx].values()))

        # Summary
        logs = {}
        print_log = '{:>16} '.format('Sequence_Name')
        for metrics in metrics_name:
            print_log += '{:>12}'.format(metrics)
        print_log += '\n'

        for seq_name in single_seq_logs['Seq_Names']:
            print_log += '{:>16} '.format(seq_name[:11])

            for metrics in metrics_name:
                print_log += '{:12.4f}'.format(np.mean(single_seq_logs[metrics][seq_name]))

            print_log += '\n'
        print_log += '================================================\n'
        for dataset_name, rd in rd_dict.items():
            print_log += '{:>16} '.format(dataset_name)

            for metrics in metrics_name:
                logs['test/' + dataset_name + ' ' + metrics] = np.mean(rd[metrics])
                print_log += '{:12.4f}'.format(np.mean(rd[metrics]))

            print_log += '\n'

        logger.info(f"\n{print_log}")

        summary_txt_name = f'brief_summary_{"&".join(self.args.test_dataset)}'
        if self.args.color_transform == 'BT709':
            summary_txt_name += '_BT709'
        elif self.args.color_transform == 'BT601_Bilinear':
            summary_txt_name += '_BT601'
        
        if self.args.compress:
            summary_txt_name += '_enc'
        elif self.args.decompress:
            summary_txt_name += '_dec'

        if self.args.full_seq:
            summary_txt_name += '_FullSeq'

        with open(self.args.save_dir + f'/{summary_txt_name}.txt', 'w', newline='') as report:
            report.write(print_log)

    @torch.no_grad()
    def compress_step(self, batch, first=False):
        if self.args.ssim: 
            dist_metric = 'MS-SSIM'
        else: 
            dist_metric = 'PSNR'

        metrics_name = [dist_metric, 'Rate', 'Mo_Rate', f'MC-{dist_metric}', f'Warped-{dist_metric}', 'Mo_Ratio', 'Res_Ratio']

        metrics = {}
        for m in metrics_name:
            metrics[m] = []

        log_list = []

        dataset_name, seq_name, batch, frame_id_start = batch
        device = next(self.parameters()).device
        batch = batch.to(device)

        if self.args.test_crop:
            H, W = batch.shape[-2:]
            batch = transforms.CenterCrop((self.args.crop_factor * (H // self.args.crop_factor), self.args.crop_factor * (W // self.args.crop_factor)))(batch)

        seq_name = seq_name[0]
        dataset_name = dataset_name[0]
        self.gop_size = batch.size(1)

        TO_VISUALIZE_Flag = (False and self.args.quality_level == 5) or self.args.visualize
        if TO_VISUALIZE_Flag:
            # os.makedirs(self.args.save_dir + f'/{seq_name}/flow_hat', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/warped_frame', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/enc_frame', exist_ok=True)


        SCENE_CUT = {
            'Kimono1': [141], 'Kimono1_10b_rgb': [141],
            'videoSRC20': [43, 81, 106], 'videoSRC21': [49, 64, 90], 'videoSRC25': [18, 84], 'videoSRC26': [62], 'videoSRC27': [40], 'videoSRC29': [71],
            "Fallout4": [560]
        }

        self.num_frames = self.args.test_seq_len
        self.shape = batch.shape[-2:]

        outstanding_sps_bytes = 0
        
        rec_list = []

        for frame_idx in range(self.gop_size):
            torch.cuda.empty_cache()
            logger.debug(f'===== {seq_name} frame={frame_id_start.item() + frame_idx} =====')
            TO_VISUALIZE = TO_VISUALIZE_Flag and frame_id_start == 1 and frame_idx < 8 and seq_name in ['BasketballDrive', 'Kimono1', 'HoneyBee', 'Jockey']

            coding_frame = batch[:, frame_idx]
            # pad if necessary
            height, width = coding_frame.size(2), coding_frame.size(3)
            padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, self.args.align_factor)
            coding_frame_pad = F.pad(coding_frame, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

            if self.args.reset_interval > 0 and frame_idx % self.args.reset_interval == 1:
                self.implicit_buffer["ref_feature"] = None
                fa_idx = 3
            
            # I frame
            if frame_idx == 0 or (self.args.remove_scene_cut and seq_name in SCENE_CUT and frame_id_start + frame_idx in SCENE_CUT[seq_name]):
                
                self.implicit_buffer = {
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_feature": None,
                    "ref_mv_y": None,
                }

                sps = {
                    'sps_id': -1,
                    'height': coding_frame_pad.size(2),
                    'width': coding_frame_pad.size(3),
                    'qp': self.args.iframe_quality,
                    'fa_idx': 0,
                }

                rec_frame, strings, info = self.if_model.compress(coding_frame_pad, torch.tensor([self.args.iframe_quality]))
                del coding_frame_pad

                sps_id, sps_new = self.sps_helper.get_sps_id(sps)
                sps['sps_id'] = sps_id
                if sps_new:
                    outstanding_sps_bytes += write_sps(self.output_file, sps)
                    # print("new sps", sps)

                strings = [strings]
                total_strlen, strlen_list = write_ip_segment(self.output_file, True, sps_id, strings)
                del strings
                
                self.bits.append((total_strlen + outstanding_sps_bytes) * 8)

                rate = strlen_list[1] * 8 / coding_frame.shape[-2] / coding_frame.shape[-1]

                rec_frame = rec_frame.clamp(0, 1)
                x_hat = F.pad(rec_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

                if self.args.ssim:
                    distortion = self.criterion(x_hat, coding_frame).mean().item()
                else:
                    mse = self.criterion(x_hat, coding_frame).mean().item()
                    distortion = mse2psnr(mse)

                log_list.append({f'{dist_metric}': distortion, 'Rate': rate, 
                                 f'MC-{dist_metric}': 0, f'Warped-{dist_metric}': 0,
                                 'my': 0, 
                                 'mz': 0,
                                 'ry': 0, 
                                 'rz': 0})

                outstanding_sps_bytes = 0
                del info
            
            # P frame
            else:
                predict = False if (self.args.remove_scene_cut and (seq_name in SCENE_CUT) and ((frame_id_start + frame_idx - 1) in SCENE_CUT[seq_name])) else (frame_idx!=1)

                rec_frame, strings, info = self.compress(ref_frame, coding_frame_pad, predict, frame_idx=frame_idx,
                                                         index_list=[self.args.QP])

                del coding_frame_pad

                sps_id, sps_new = self.sps_helper.get_sps_id(info['sps'])
                info['sps']['sps_id'] = sps_id
                if sps_new:
                    outstanding_sps_bytes += write_sps(self.output_file, info['sps'])
                    # print("new sps", info['sps'])

                total_strlen, strlen_list = write_ip_segment(self.output_file, False, sps_id, strings)
                del strings

                self.bits.append((total_strlen + outstanding_sps_bytes) * 8)

                rec_frame = rec_frame.clamp(0, 1)
                x_hat = F.pad(rec_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

                mc_frame = info['mc_frame'].clamp(0, 1)
                mc_frame = F.pad(mc_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
                warped_frame = info['warped_frame']
                warped_frame = F.pad(warped_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

                my = strlen_list[1] * 8 / coding_frame.shape[-2] / coding_frame.shape[-1]
                mz = strlen_list[2] * 8 / coding_frame.shape[-2] / coding_frame.shape[-1]

                ry_strlen = sum(strlen_list[3:7])
                rz_strlen = strlen_list[7]

                ry = ry_strlen * 8 / coding_frame.shape[-2] / coding_frame.shape[-1]
                rz = rz_strlen * 8 / coding_frame.shape[-2] / coding_frame.shape[-1]

                m_rate = my + mz
                r_rate = ry + rz
                rate = m_rate + r_rate

                if self.args.ssim:
                    distortion = self.criterion(x_hat, coding_frame).mean().item()
                    mc_distortion = self.criterion(mc_frame, coding_frame).mean().item()
                    warped_distortion = self.criterion(warped_frame, coding_frame).mean().item()
                    
                else:
                    mse = self.criterion(x_hat, coding_frame).mean().item()
                    distortion = mse2psnr(mse)

                    mc_mse = self.criterion(mc_frame, coding_frame).mean().item()
                    mc_distortion = mse2psnr(mc_mse)

                    warped_mse = self.criterion(warped_frame, coding_frame).mean().item()
                    warped_distortion = mse2psnr(warped_mse)

                if TO_VISUALIZE:
                    # flow_map = plot_flow(info['flow_hat'])
                    # save_image(flow_map,
                    #            self.args.save_dir + f'/{seq_name}/flow_hat/'
                    #                                 f'frame_{int(frame_id_start + frame_idx)}_flow.png', nrow=1)
                    # save_image(coding_frame[0], 
                    #            self.args.save_dir + f'/{seq_name}/gt_frame/'
                    #                                 f'frame_{int(frame_id_start + frame_idx)}.png')
                    # save_image(mc_frame[0], 
                    #            self.args.save_dir + f'/{seq_name}/mc_frame/'
                    #                                 f'frame_{int(frame_id_start + frame_idx)}.png')
                    # save_image(warped_frame[0], 
                    #            self.args.save_dir + f'/{seq_name}/warped_frame/'
                    #                                 f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(rec_frame[0], 
                               self.args.save_dir + f'/{seq_name}/enc_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')

                log_list.append({f'{dist_metric}': distortion, 'Rate': rate, 
                                 f'MC-{dist_metric}': mc_distortion, f'Warped-{dist_metric}': warped_distortion,
                                 'my': my, 
                                 'mz': mz,
                                 'ry': ry, 
                                 'rz': rz})

                metrics['Mo_Rate'].append(m_rate)
                metrics[f'MC-{dist_metric}'].append(mc_distortion)
                metrics[f'Warped-{dist_metric}'].append(warped_distortion)
                metrics['Mo_Ratio'].append(m_rate/rate)
                metrics['Res_Ratio'].append(r_rate/rate)

                outstanding_sps_bytes = 0
                del info, mc_frame, warped_frame

            metrics[f'{dist_metric}'].append(distortion)
            metrics['Rate'].append(rate)

            ref_frame = rec_frame

            if self.args.YUV_FILE:
                rec_yuv = self.RGB2YUV444(x_hat.unsqueeze(0))
                rec_list.extend(rec_yuv)
            if self.args.PNG_FILE:
                os.makedirs(self.args.save_dir + f'/{seq_name}/enc_frame', exist_ok=True)
                save_image(x_hat[0],
                           self.args.save_dir + f'/{seq_name}/enc_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
            del x_hat

        if self.args.YUV_FILE:
            dtype, max_val = write_yuv_args(8, device)

            for i in range(self.gop_size):
                Y, U, V = rec_list[i][0].chunk(3, dim=0)

                U = U[:, ::2, ::2]
                V = V[:, ::2, ::2]
                
                Y = Y * max_val
                U = U * max_val
                V = V * max_val
                
                Y = np.asarray(Y.cpu().numpy()).astype(dtype).tobytes()
                U = np.asarray(U.cpu().numpy()).astype(dtype).tobytes()
                V = np.asarray(V.cpu().numpy()).astype(dtype).tobytes()
                self.yuv_file.write(Y)
                self.yuv_file.write(U)
                self.yuv_file.write(V)

        for m in metrics_name:
            metrics[m] = np.mean(metrics[m])

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 'metrics': metrics, 'log_list': log_list}
        return logs

    @torch.no_grad()
    def decompress_step(self, batch, first=False):

        if self.args.ssim: 
            dist_metric = 'MS-SSIM'
        else: 
            dist_metric = 'PSNR'
        
        metrics_name = [dist_metric, 'Rate', 'Mo_Rate', f'MC-{dist_metric}', f'Warped-{dist_metric}', 'Mo_Ratio', 'Res_Ratio']
        
        metrics = {}
        for m in metrics_name:
            metrics[m] = []

        log_list = []

        dataset_name, seq_name, batch, frame_id_start = batch
        device = next(self.parameters()).device
        batch = batch.to(device)

        if self.args.test_crop:
            H, W = batch.shape[-2:]
            batch = transforms.CenterCrop((self.args.crop_factor * (H // self.args.crop_factor), self.args.crop_factor * (W // self.args.crop_factor)))(batch)

        seq_name = seq_name[0]
        dataset_name = dataset_name[0]
        self.gop_size = batch.size(1)

        self.num_frames = self.args.test_seq_len
        self.gop_num = self.num_frames // self.gop_size
        self.shape = batch.shape[-2:]
        
        TO_VISUALIZE_Flag = (False and self.args.quality_level == 5) or self.args.visualize
        if TO_VISUALIZE_Flag:
            # os.makedirs(self.args.save_dir + f'/{seq_name}/flow_hat', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/warped_frame', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/dec_frame', exist_ok=True)

        rec_list = []
        pending_frame_spss = []
        frame_idx = 0
        I_frame_Flag = False

        while frame_idx < self.gop_size:
            logger.debug(f"frame_idx: {(frame_id_start.item() + frame_idx)}")
            
            new_stream = False
            if len(pending_frame_spss) == 0:
                header = read_header(self.input_file)
                if header['nal_type'] == NalType.NAL_SPS:
                    sps = read_sps_remaining(self.input_file, header['sps_id'])
                    self.sps_helper.add_sps_by_id(sps)
                    # print("new sps", sps)
                    continue
                if header['nal_type'] == NalType.NAL_Ps:
                    pending_frame_spss = header['sps_ids'][1:]
                    sps_id = header['sps_ids'][0]
                else:
                    sps_id = header['sps_id']
                new_stream = True
            else:
                sps_id = pending_frame_spss[0]
                pending_frame_spss.pop(0)
            sps = self.sps_helper.get_sps_by_id(sps_id)

            # torch.cuda.empty_cache()
            TO_VISUALIZE = TO_VISUALIZE_Flag and frame_id_start == 1 and frame_idx < 8 and seq_name in ['BasketballDrive', 'Kimono1', 'HoneyBee', 'Jockey']

            coding_frame = batch[:, frame_idx]
            # pad if necessary
            height, width = coding_frame.size(2), coding_frame.size(3)
            padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, self.args.align_factor)
            # coding_frame_pad = F.pad(coding_frame, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

            if self.args.reset_interval > 0 and frame_idx % self.args.reset_interval == 1:
                self.implicit_buffer["ref_feature"] = None
            
            # I frame
            if frame_idx == 0 or header['nal_type'] == NalType.NAL_I:
                self.implicit_buffer = {
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_feature": None,
                    "ref_mv_y": None,
                }

                strings_list, num_bytes_list = [], []

                if new_stream:
                    strings_list, num_bytes_list = read_ip_remaining_segment(self.input_file, read_byte_segment=1)
                else:
                    raise NotImplementedError

                rate = (len(strings_list[0]) + num_bytes_list[0]) * 8 / self.shape[-2] / self.shape[-1]

                rec_frame = self.if_model.decompress(strings_list[0], shape=torch.Size([1, 3, sps['height'], sps['width']]),
                                                     q_index=torch.tensor(sps['qp']), sps=sps)
                
                rec_frame = rec_frame.clamp(0, 1)
                x_hat = F.pad(rec_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

                if self.args.ssim:
                    distortion = self.criterion(x_hat, coding_frame).mean().item()
                else:
                    mse = self.criterion(x_hat, coding_frame).mean().item()
                    distortion = mse2psnr(mse)

                log_list.append({f'{dist_metric}': distortion, 'Rate': rate, 
                                 f'MC-{dist_metric}': 0, f'Warped-{dist_metric}': 0,
                                 'my': 0, 
                                 'mz': 0,
                                 'ry': 0, 
                                 'rz': 0})

                I_frame_Flag = True

            # P frame
            elif header['nal_type'] == NalType.NAL_P or header['nal_type'] == NalType.NAL_Ps:

                if sps['fa_idx'] == 3:
                    self.implicit_buffer["ref_feature"] = None
                
                strings_list, num_bytes_list = [], []
                rate = 0

                if new_stream:
                    strings_list, num_bytes_list = read_ip_remaining_segment(self.input_file, read_byte_segment=7)
                else:
                    raise NotImplementedError

                my = (len(strings_list[0]) + num_bytes_list[0]) * 8 / self.shape[-2] / self.shape[-1]
                mz = (len(strings_list[1]) + num_bytes_list[1]) * 8 / self.shape[-2] / self.shape[-1]

                ry_strlen = len(strings_list[2]) + len(strings_list[3]) + len(strings_list[4]) + len(strings_list[5]) + sum(num_bytes_list[2:6])

                ry = ry_strlen * 8 / self.shape[-2] / self.shape[-1]
                rz = (len(strings_list[6]) + num_bytes_list[6]) * 8 / self.shape[-2] / self.shape[-1]

                m_rate = my + mz
                r_rate = ry + rz
                rate = m_rate + r_rate
                
                if I_frame_Flag:
                    predict = False
                    I_frame_Flag = False
                else:
                    predict = (frame_idx!=1)

                rec_frame, info = self.decompress(ref_frame, strings_list, predict,
                                                  frame_idx, sps, index_list=[sps['qp']])
                
                rec_frame = rec_frame.clamp(0, 1)
                x_hat = F.pad(rec_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

                mc_frame = info['mc_frame'].clamp(0, 1)
                mc_frame = F.pad(mc_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
                warped_frame = info['warped_frame']
                warped_frame = F.pad(warped_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

                if self.args.ssim:
                    distortion = self.criterion(x_hat, coding_frame).mean().item()
                    mc_distortion = self.criterion(mc_frame, coding_frame).mean().item()
                    warped_distortion = self.criterion(warped_frame, coding_frame).mean().item()
                    
                else:
                    mse = self.criterion(x_hat, coding_frame).mean().item()
                    distortion = mse2psnr(mse)

                    mc_mse = self.criterion(mc_frame, coding_frame).mean().item()
                    mc_distortion = mse2psnr(mc_mse)

                    warped_mse = self.criterion(warped_frame, coding_frame).mean().item()
                    warped_distortion = mse2psnr(warped_mse)

                
                if TO_VISUALIZE:
                    # flow_map = plot_flow(info['flow_hat'])
                    # save_image(flow_map,
                    #            self.args.save_dir + f'/{seq_name}/flow_hat/'
                    #                                 f'frame_{int(frame_id_start + frame_idx)}_flow.png', nrow=1)
                    # save_image(coding_frame[0], 
                    #            self.args.save_dir + f'/{seq_name}/gt_frame/'
                    #                                 f'frame_{int(frame_id_start + frame_idx)}.png')
                    # save_image(mc_frame[0], 
                    #            self.args.save_dir + f'/{seq_name}/mc_frame/'
                    #                                 f'frame_{int(frame_id_start + frame_idx)}.png')
                    # save_image(warped_frame[0], 
                    #            self.args.save_dir + f'/{seq_name}/warped_frame/'
                    #                                 f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(rec_frame[0], 
                               self.args.save_dir + f'/{seq_name}/dec_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')

                log_list.append({f'{dist_metric}': distortion, 'Rate': rate, 
                                 f'MC-{dist_metric}': mc_distortion, f'Warped-{dist_metric}': warped_distortion,
                                 'my': my, 
                                 'mz': mz,
                                 'ry': ry, 
                                 'rz': rz})

                metrics['Mo_Rate'].append(m_rate)
                metrics[f'MC-{dist_metric}'].append(mc_distortion)
                metrics[f'Warped-{dist_metric}'].append(warped_distortion)
                metrics['Mo_Ratio'].append(m_rate/rate)
                metrics['Res_Ratio'].append(r_rate/rate)

            metrics[f'{dist_metric}'].append(distortion)
            metrics['Rate'].append(rate)

            ref_frame = rec_frame

            if self.args.YUV_FILE:
                rec_yuv = self.RGB2YUV444(x_hat.unsqueeze(0))
                rec_list.extend(rec_yuv[0])
            if self.args.PNG_FILE:
                os.makedirs(self.args.save_dir + f'/{seq_name}/dec_frame', exist_ok=True)
                save_image(x_hat[0],
                           self.args.save_dir + f'/{seq_name}/dec_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')

            if (self.gop_num == self.gop_count) and (frame_idx == self.last_gop_size - 1):
                break

            frame_idx += 1

        if self.args.YUV_FILE:
            dtype, max_val = write_yuv_args(8, device)

            for i in range(len(rec_list)):
                Y, U, V = rec_list[i].chunk(3, dim=0)

                U = U[:, ::2, ::2]
                V = V[:, ::2, ::2]
                
                Y = Y * max_val
                U = U * max_val
                V = V * max_val
                
                Y = np.asarray(Y.cpu().numpy()).astype(dtype).tobytes()
                U = np.asarray(U.cpu().numpy()).astype(dtype).tobytes()
                V = np.asarray(V.cpu().numpy()).astype(dtype).tobytes()
                self.yuv_file.write(Y)
                self.yuv_file.write(U)
                self.yuv_file.write(V)

        for m in metrics_name:
            metrics[m] = np.mean(metrics[m])

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 'metrics': metrics, 'log_list': log_list}
        return logs
    
    def setup(self, stage):

        if stage == 'test':
            self.test_dataset = VideoTestData(self.args.dataset_root, sequence=self.args.test_dataset, GOP=self.args.gop, test_seq_len=self.args.test_seq_len, use_seqs=self.args.test_seqs, full_seq=self.args.full_seq, color_transform=self.args.color_transform)

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=1,
                                 num_workers=self.args.num_workers,
                                 shuffle=False)
        return test_loader

    def parallel(self, device_ids):
        self.if_model      = CustomDataParallel(self.if_model, device_ids=device_ids)
        self.MENet         = CustomDataParallel(self.MENet, device_ids=device_ids)
        self.CondMotion    = CustomDataParallel(self.CondMotion, device_ids=device_ids)
        self.Residual      = CustomDataParallel(self.Residual, device_ids=device_ids)

def parse_args(argv):
    parser = argparse.ArgumentParser()

    # training specific
    parser.add_argument('--MENet', type=str, choices=['PWC', 'SPy'], default='SPy')
    parser.add_argument('--if_coder', type=str, default='Intra')
    parser.add_argument('--cond_motion_coder_conf', type=str, default=None)
    parser.add_argument('--residual_coder_conf', type=str, default=None)
    parser.add_argument("-n", "--num_workers", type=int, default=4, help="Dataloaders threads (default: %(default)s)")
    parser.add_argument("-i", "--iframe_quality", type=int, default=None, help="Quality level of i frame coder (default: %(default)s)")
    parser.add_argument("--QP", type=int, default=63)
    parser.add_argument('--ssim', action='store_true', help="Optimize for MS-SSIM")
    
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPU (default: %(default)s)")
    parser.add_argument('--save_dir', default=None, help='directory for saving testing result')
    parser.add_argument('--gop', default=32, type=int)
    
    parser.add_argument('--experiment_name', type=str, default='HyTIP')

    parser.add_argument('--compute_macs', action='store_true')
    parser.add_argument('--compute_decode_macs', action='store_true')
    parser.add_argument('--compute_model_size', action='store_true')
    parser.add_argument('-s', '--remove_scene_cut', action='store_true')

    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--decompress', action='store_true')

    parser.add_argument('-data', '--dataset_root', type=str, default=None, help='Root for dataset')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-c', '--test_crop', action='store_true')
    parser.add_argument('--crop_factor', type=int, default=64)
    parser.add_argument('--align_factor', type=int, default=16)
    parser.add_argument('--test_dataset', type=str, nargs='+', default=['HEVC-B', 'UVG'])
    parser.add_argument('--test_seqs', type=str, nargs='+', choices=list(SEQUENCES.keys()), default=[])
    parser.add_argument('--test_seq_len', type=int, default=96)
    parser.add_argument('--full_seq', action='store_true')
    parser.add_argument('--color_transform', type=str, default='BT601', choices=['BT601', 'BT709', 'BT601_Bilinear'])
    parser.add_argument('-vis', '--visualize', action='store_true')
    parser.add_argument("--YUV_FILE", action='store_true')
    parser.add_argument("--PNG_FILE", action='store_true')
    
    parser.add_argument("--index_map", type=int, nargs='+', default=[0, 1, 0, 2, 0, 2, 0, 2])
    parser.add_argument("--rate_gop_size", type=int, default=8, choices=[4, 8])
    parser.add_argument('--reset_interval', type=int, default=32, required=False)

    args = parser.parse_args(argv)

    return args

def main(argv):
    args = parse_args(argv)
    assert args.gpus <= torch.cuda.device_count(), "Can't find enough gpus in the machine."

    if args.save_dir is None:
        os.makedirs('results', exist_ok=True)
        args.save_dir = os.path.join('results', args.experiment_name + '-' + str(args.QP))

    seed_everything(888888)
    
    gpu_ids = [0]
    for i in range(1, args.gpus):
        gpu_ids.append(i)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config coders
    assert not (args.cond_motion_coder_conf is None)
    cond_mo_coder_cfg = yaml.safe_load(open(args.cond_motion_coder_conf, 'r'))
    assert cond_mo_coder_cfg['model_architecture'] in __CODER_TYPES__.keys()
    cond_mo_coder_arch = __CODER_TYPES__[cond_mo_coder_cfg['model_architecture']]
    cond_mo_coder = cond_mo_coder_arch(**cond_mo_coder_cfg['model_params'])

    assert not (args.residual_coder_conf is None)
    res_coder_cfg = yaml.safe_load(open(args.residual_coder_conf, 'r'))
    assert res_coder_cfg['model_architecture'] in __CODER_TYPES__.keys()
    res_coder_arch = __CODER_TYPES__[res_coder_cfg['model_architecture']]
    res_coder = res_coder_arch(**res_coder_cfg['model_params'])
    
    model = Pframe(args, cond_mo_coder, res_coder)

    if args.compute_model_size:
        ## Method 1 ##
        modules = {'ME' : model.MENet, 'CondMotion' : model.CondMotion, 'Residual' : model.Residual, 'Whole': model}
        for key in modules.keys():
            summary(modules[key])
            param_size = 0
            for param in modules[key].parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in modules[key].buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            size_all_mb = (param_size + buffer_size) / 1024**2
            print(f'{key} size: {size_all_mb:.3f}MB')
        
        ## Method 2 ##
        def show_model_size(net, model_name=None):
            if model_name is not None:
                print(f"{model_name}:")
            print("=============== Model Size ===============")
            total = 0
            for name, module in net.named_children():
                sum = 0
                for param in module.parameters():
                    sum += param.numel()
                total += sum
                print(f"{name}: {sum/1e6:.3f} M params")
            print("==========================================")
            print(f"Total: {total/1e6:.3f} M params\n")
        
        print("\n")
        show_model_size(model, "Whole")
        show_model_size(model.CondMotion.net, "CondMotion")
        show_model_size(model.Residual.net, "Residual")

        exit()

    if args.compute_macs:
        current_epoch = 1

    else:
        if args.ssim:
            checkpoint = torch.load(os.path.join('models', 'MS-SSIM-RGB.pth.tar'), map_location=device)
        else:
            checkpoint = torch.load(os.path.join('models', 'PSNR-RGB.pth.tar'), map_location=device)
        current_epoch = checkpoint['epoch'] + 1

        ckpt = {}
        for k, v in checkpoint["state_dict"].items():
            k = k.split('.')
            if k[0] != 'feature_frame_generate' and k[0] != 'criterion':
                k.pop(1)
            if k[0] == 'Residual' and k[2].find('mc_generate') != -1:
                continue
            
            ckpt['.'.join(k)] = v

        model.load_state_dict(ckpt, strict=True)

        model.if_model.interpolate()

        model.to(device)

    if args.gpus >= 1 and torch.cuda.device_count() >= 1:
        model.parallel(device_ids=gpu_ids)
    
    if args.compress or args.decompress:
        model.update()
    
    trainer = Trainer(args, model, current_epoch, device)
    
    if args.test:
        assert not (args.compress and args.decompress), "Can't use compress and decompress mode at the same time."
        if args.compress:
            logger.add(f"{args.save_dir}/{args.test_seqs[0]}_compress.log", level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)
            trainer.compress()
        elif args.decompress:
            logger.add(f"{args.save_dir}/{args.test_seqs[0]}_decompress.log", level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)
            trainer.decompress()
        else:
            trainer.test()


if __name__ == "__main__":
    # log_level = "DEBUG"
    log_level = "INFO"
    log_format = "<green>{time:YYMMDD HH:mm:ss}</green> | <level>{level: <5}</level> | <yellow>{file}:{line:<3d}</yellow> | <b>{message}</b>"
    logger.add(sys.stdout, level=log_level, format=log_format, colorize=True, backtrace=True, diagnose=True)
    logger.remove(0)
    
    main(sys.argv[1:])