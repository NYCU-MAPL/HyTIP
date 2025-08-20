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
import json
import math
import os
import random
import sys
from comet_ml import Experiment, ExistingExperiment
import flowiz as fz
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from torchvision.utils import make_grid

from advance_model import *
from compressai.models import __CODER_TYPES__
from compressai.models.utils import get_padding_size
from dataloader import VideoData, VideoTestData, BVI_Dataset, SEQUENCES
from trainer import Trainer
from util.estimate_bpp import estimate_bpp
from util.math import lower_bound
from util.psnr import mse2psnr
from util.sampler import Resampler
from util.seed import seed_everything
from pytorch_msssim import MS_SSIM as MS_SSIM_PyTorch
from util.vision import PlotFlow, PlotHeatMap, save_image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from loguru import logger

plot_flow = PlotFlow().cuda()
plot_bitalloc = PlotHeatMap("RB").cuda()

lmda = {2: 0.0035, 3: 0.0067, 4: 0.0130, 5: 0.03125}

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

class Pframe(CompressesModel):
    def __init__(self, args, cond_mo_coder, res_coder, train_cfg, comet_logger):
        super(Pframe, self).__init__()
        self.args = args
        self.mo_cond_source = cond_mo_coder.cond_source
        self.mo_temp_source = cond_mo_coder.temp_source
        self.res_no_implicit = res_coder.no_implicit
        self.res_add_mask = res_coder.Add_Mask

        if self.args.ssim:
            self.criterion = MS_SSIM_PyTorch(data_range=1.).cuda()
        else:
            self.criterion = nn.MSELoss(reduction='none')

        if not self.args.compute_model_size:
            self.if_model = Iframe_Coder(args.if_coder, args.iframe_quality, args.ssim, q_in_ckpt=True)
            self.if_model.eval()

        self.MENet = MENet(args.MENet)
        self.CondMotion = MotionCoder(cond_mo_coder)
        self.Resampler = Resampler()
        self.Residual = InterCoder(res_coder)

        if (not self.res_no_implicit) and self.args.Pretrain_frame_feature:
            self.feature_frame_generate = nn.Conv2d(self.args.feature_channel, 3, 3, stride=1, padding=1)

        self.train_cfg = train_cfg

        self.logger = comet_logger

        self.ref_flow = None

        self.implicit_buffer = {}
        if self.mo_cond_source.find('ref_mv_feature') != -1:
            self.implicit_buffer.update({"ref_mv_feature": None})
        if self.mo_temp_source.find('ref_mv_y') != -1:
            self.implicit_buffer.update({"ref_mv_y": None})
        if not self.res_no_implicit:
            self.implicit_buffer.update({"ref_feature": None, "ref_y": None})

        self.CondMotion.set_noise_level(0.45)
        self.Residual.set_noise_level(0.45)

        if self.args.frame_weight is not None:
            self.frame_weight = self.args.frame_weight

    def update(self):
        self.if_model.update()
        self.CondMotion.update()
        self.Residual.update()

    def freeze(self, modules):
        '''
            modules (list): contain modules that need to freeze 
        '''
        self.requires_grad_(True)
        for module in modules:
            module = module.split('.')
            _modules = self._modules
            for m in module[:-1]:
                _modules = _modules[m]._modules
            
            for param in _modules[module[-1]].parameters(): 
                self.optimizer.state_dict()[param] = {}

            _modules[module[-1]].requires_grad_(False)

    def activate(self, modules):
        '''
            modules (list): contain modules that need to activate 
        '''
        for activate_list in modules:
            for name, parm in self._modules[activate_list[0]].named_parameters():
                if name.find(activate_list[1]) != -1:
                    parm.requires_grad_(True)
                
    def motion_forward(self, ref_frame, coding_frame, predict=False, RNN=True):
        if not self.args.compute_decode_macs:
            flow = self.MENet(ref_frame, coding_frame)
        else:
            flow = torch.zeros_like(ref_frame[:, 0:2, :, :])

        aux_buf = {
            'cond_input': self.ref_flow if predict else None,
            'temporal_input': torch.cat([ref_frame, self.ref_flow], dim=1) if predict else None,
            'compute_decode_macs': self.args.compute_decode_macs,
        }
        if self.mo_cond_source.find('ref_mv_feature') != -1:
            aux_buf.update({'ref_mv_feature': self.implicit_buffer["ref_mv_feature"]})
        if self.mo_temp_source.find('ref_mv_y') != -1:
            aux_buf.update({'ref_mv_y': self.implicit_buffer["ref_mv_y"]})
        
        flow_hat, likelihood_m, data = self.CondMotion(flow, aux_buf=aux_buf)
        
        if self.mo_cond_source.find('ref_mv_feature') != -1:
            self.implicit_buffer["ref_mv_feature"] = data["ref_mv_feature"]
        if self.mo_temp_source.find('ref_mv_y') != -1:
            self.implicit_buffer["ref_mv_y"] = data["ref_mv_y"]
        del data
        
        self.ref_flow = flow_hat if RNN else flow_hat.detach()

        warped_frame = self.Resampler(ref_frame, flow_hat)
        
        m_info = {'likelihood_m': likelihood_m, 
                  'flow': flow, 'flow_hat': flow_hat,
                  'warped_frame': warped_frame}
        
        return likelihood_m, m_info

    def forward(self, ref_frame, coding_frame, predict, RNN=True, frame_idx=None):
        
        if self.res_no_implicit:
            fa_idx = 0
        elif self.training:
            fa_idx = self.args.index_map_training[frame_idx % self.args.rate_gop_size]
        else:
            fa_idx = self.args.index_map[frame_idx % self.args.rate_gop_size]
        
        likelihood_m, m_info = self.motion_forward(ref_frame, coding_frame, predict, RNN)

        aux_buf = {
            "fa_idx": fa_idx,
            "mv_hat": m_info['flow_hat'],
            "ref_frame": ref_frame,
            "compute_decode_macs": self.args.compute_decode_macs,
        }
        if not self.res_no_implicit:
            aux_buf.update({
                "ref_feature": self.implicit_buffer["ref_feature"],
                "ref_y": self.implicit_buffer["ref_y"],
            })

        rec_frame, likelihood_r, info = self.Residual(coding_frame, aux_buf=aux_buf)

        if not self.res_no_implicit:
            self.implicit_buffer["ref_feature"] = info["ref_feature"]
            self.implicit_buffer["ref_y"] = info["ref_y"]
        
        likelihoods = likelihood_m + likelihood_r

        r_info = {
            'res'     : info['res'],
            'res_hat' : info['res_hat'],
            'mc_frame': info['mc_frame']
        }
        if self.res_add_mask:
            r_info.update({'mask': info['mask']})

        if (not self.res_no_implicit) and self.args.Pretrain_frame_feature:
            feature_frame = self.feature_frame_generate(info['ref_feature'])
            r_info.update({'feature_frame': feature_frame})
        
        return rec_frame, likelihoods, m_info, r_info

    def train_Motion_1Pframe(self, batch, loss_on, random_train=False):
        idx = [0, 1]

        if random_train:
            idx[0] = random.randint(0, 6)
            if idx[0] == 0:
                direction = 1
            elif idx[0] == 6:
                direction = -1
            else:
                direction = random.choice([-1, 1])

            idx[1] = idx[0]+direction

        loss = torch.tensor(0., dtype=torch.float, device=batch.device)

        with torch.no_grad():
            rec_frame, _ = self.if_model(batch[:, idx[0]])
            ref_frame = rec_frame
            self.implicit_buffer = {}
            if self.mo_cond_source.find('ref_mv_feature') != -1:
                self.implicit_buffer.update({"ref_mv_feature": None})
            if self.mo_temp_source.find('ref_mv_y') != -1:
                self.implicit_buffer.update({"ref_mv_y": None})
            if not self.res_no_implicit:
                self.implicit_buffer.update({"ref_feature": None, "ref_y": None})
          
        coding_frame = batch[:, 1]

        likelihood_m, m_info = self.motion_forward(ref_frame, coding_frame, predict=False, RNN=False)

        m_y = estimate_bpp(likelihood_m[0], input=m_info['flow']).mean()
        m_z = estimate_bpp(likelihood_m[1], input=m_info['flow']).mean()

        result = {
            'train/m_rate'              : m_y + m_z,
            'train/warped_distortion'   : self.criterion(coding_frame, m_info['warped_frame']).mean(),
            # 'train/mc_distortion'       : self.criterion(coding_frame, mc_frame).mean(),
        }

        if not self.args.ssim:
            result.update({
                'train/motion_mse'      : self.criterion(m_info['flow'], m_info['flow_hat']).mean(),
            })

        rate = torch.tensor(0., dtype=torch.float, device=rec_frame.device)
        for term in loss_on['R'].split('/'):
            coefficient = 1
            if '*' in term:
                coefficient = float(term.split('*')[0])
                term = term.split('*')[1]

            if term == 'None':
                continue
            
            rate += coefficient * result['train/'+term]

        distortion = torch.tensor(0., dtype=torch.float, device=rec_frame.device)
        for term in loss_on['D'].split('/'):
            coefficient = 1
            if '*' in term:
                coefficient = float(term.split('*')[0])
                term = term.split('*')[1]

            if term == 'None':
                continue
            
            if self.args.ssim:
                distortion += coefficient * (1 - result['train/'+term]) / (32 * math.sqrt(2))
            else: 
                distortion += coefficient * result['train/'+term]

        loss += rate + 255**2 * lmda[self.args.quality_level] * distortion
        result.update({'train/loss': loss})

        return loss, result

    def train_1Pframe(self, batch, loss_on, random_train=False):
        idx = [0, 1]

        if random_train:
            idx[0] = random.randint(0, 6)
            if idx[0] == 0:
                direction = 1
            elif idx[0] == 6:
                direction = -1
            else:
                direction = random.choice([-1, 1])

            idx[1] = idx[0]+direction

        loss = torch.tensor(0., dtype=torch.float, device=batch.device)

        with torch.no_grad():
            rec_frame, _ = self.if_model(batch[:, idx[0]])
            ref_frame = rec_frame
            self.implicit_buffer = {}
            if self.mo_cond_source.find('ref_mv_feature') != -1:
                self.implicit_buffer.update({"ref_mv_feature": None})
            if self.mo_temp_source.find('ref_mv_y') != -1:
                self.implicit_buffer.update({"ref_mv_y": None})
            if not self.res_no_implicit:
                self.implicit_buffer.update({"ref_feature": None, "ref_y": None})
          
        coding_frame = batch[:, 1]

        rec_frame, likelihoods, m_info, r_info = self(ref_frame, coding_frame, predict=False, RNN=False, frame_idx=1)

        m_rate = estimate_bpp(likelihoods[:2], input=m_info['flow']).mean()
        r_rate = estimate_bpp(likelihoods[2:], input=coding_frame).mean()

        result = {
            'train/m_rate'              : m_rate,
            'train/r_rate'              : r_rate,
            'train/rate'                : m_rate + r_rate,
            'train/warped_distortion'   : self.criterion(coding_frame, m_info['warped_frame']).mean(),
            'train/mc_distortion'       : self.criterion(coding_frame, r_info['mc_frame']).mean(),
            'train/distortion'          : self.criterion(coding_frame, rec_frame).mean(),
        }

        if not self.args.ssim:
            result.update({
                'train/motion_mse'      : self.criterion(m_info['flow'], m_info['flow_hat']).mean(),
            })

        if (not self.res_no_implicit) and self.args.Pretrain_frame_feature:
            result.update({'train/feature_frame_distortion': self.criterion(coding_frame, r_info['feature_frame']).mean()})

        rate = torch.tensor(0., dtype=torch.float, device=rec_frame.device)
        for term in loss_on['R'].split('/'):
            coefficient = 1
            if '*' in term:
                coefficient = float(term.split('*')[0])
                term = term.split('*')[1]

            if term == 'None':
                continue
            
            rate += coefficient * result['train/'+term]

        distortion = torch.tensor(0., dtype=torch.float, device=rec_frame.device)
        for term in loss_on['D'].split('/'):
            coefficient = 1
            if '*' in term:
                coefficient = float(term.split('*')[0])
                term = term.split('*')[1]

            if term == 'None':
                continue
            
            if self.args.ssim:
                distortion += coefficient * (1 - result['train/'+term]) / (32 * math.sqrt(2))
            else: 
                distortion += coefficient * result['train/'+term]

        loss = rate + 255**2 * lmda[self.args.quality_level] * distortion
        result.update({'train/loss': loss})
        
        del rec_frame, likelihoods, m_info, ref_frame, coding_frame

        return loss, result
    
    def train_2Pframes(self, batch, loss_on, mode, random_train=False):    
        assert mode in ['motion', 'residual']

        idx = [0, 1, 2]

        if random_train:
            idx[0] = random.randint(0, 6)
            if idx[0] in [0, 1]:
                direction = 1
            elif idx[0] in [5, 6]:
                direction = -1
            else:
                direction = random.choice([-1, 1])

            idx[1:] = [idx[0]+direction, idx[0]+2*direction]

        total_loss = torch.tensor(0., dtype=torch.float, device=batch.device)
        log = {}
        

        with torch.no_grad():
            rec_frame, _ = self.if_model(batch[:, idx[0]])
            ref_frame = rec_frame
            self.implicit_buffer = {}
            if self.mo_cond_source.find('ref_mv_feature') != -1:
                self.implicit_buffer.update({"ref_mv_feature": None})
            if self.mo_temp_source.find('ref_mv_y') != -1:
                self.implicit_buffer.update({"ref_mv_y": None})
            if not self.res_no_implicit:
                self.implicit_buffer.update({"ref_feature": None, "ref_y": None})
        
        for frame_idx, i in enumerate(idx[1:], start=1):
            coding_frame = batch[:, i]

            if mode == 'motion':
                ref_frame = batch[:, i-direction] if frame_idx != 1 else rec_frame.detach()
                likelihood_m, m_info = self.motion_forward(ref_frame, coding_frame, frame_idx!=1, RNN=False)

                result = {
                    'train/m_rate'              : estimate_bpp(likelihood_m, input=m_info['flow']).mean(),
                    'train/warped_distortion'   : self.criterion(coding_frame, m_info['warped_frame']).mean(),
                    # 'train/mc_distortion'       : self.criterion(coding_frame, mc_frame).mean(),
                }
            
            else:
                ref_frame = rec_frame.detach()
                rec_frame, likelihoods, m_info, r_info = self(ref_frame, coding_frame, frame_idx!=1, RNN=False, frame_idx=frame_idx)
                
                m_rate = estimate_bpp(likelihoods[:2], input=m_info['flow']).mean()
                r_rate = estimate_bpp(likelihoods[2:], input=coding_frame).mean()

                result = {
                    'train/m_rate'              : m_rate,
                    'train/r_rate'              : r_rate,
                    'train/rate'                : m_rate + r_rate,
                    'train/warped_distortion'   : self.criterion(coding_frame, m_info['warped_frame']).mean(),
                    'train/mc_distortion'       : self.criterion(coding_frame, r_info['mc_frame']).mean(),
                    'train/distortion'          : self.criterion(coding_frame, rec_frame).mean(),
                }

                if (not self.res_no_implicit) and self.args.Pretrain_frame_feature:
                    result.update({'train/feature_frame_distortion': self.criterion(coding_frame, r_info['feature_frame']).mean()})

                del likelihoods, r_info


            rate = torch.tensor(0., dtype=torch.float, device=rec_frame.device)
            for term in loss_on['R'].split('/'):
                coefficient = 1
                if '*' in term:
                    coefficient = float(term.split('*')[0])
                    term = term.split('*')[1]

                if term == 'None':
                    continue
                
                rate += coefficient * result['train/'+term]

            distortion = torch.tensor(0., dtype=torch.float, device=rec_frame.device)
            for term in loss_on['D'].split('/'):
                coefficient = 1
                if '*' in term:
                    coefficient = float(term.split('*')[0])
                    term = term.split('*')[1]

                if term == 'None':
                    continue
                
                if self.args.ssim:
                    distortion += coefficient * (1 - result['train/'+term]) / (32 * math.sqrt(2))
                else: 
                    distortion += coefficient * result['train/'+term]

            loss = rate + 255**2 * lmda[self.args.quality_level] * distortion
            total_loss += loss

            for k, v in result.items():
                if k not in log:
                    log[k] = []
                
                log[k].append(v)

        for k in log.keys():
            log[k] = sum(log[k]) / len(log[k])

        total_loss /= 2
        log.update({'train/loss': total_loss})

        del rec_frame, m_info, ref_frame, coding_frame

        return total_loss, log
    
    def trainMCNet(self, batch, loss_on, RNN=False, max_num_Pframe=5):   
        total_loss = torch.tensor(0., dtype=torch.float, device=batch.device)
        log = {}

        self.implicit_buffer = {}
        if self.mo_cond_source.find('ref_mv_feature') != -1:
            self.implicit_buffer.update({"ref_mv_feature": None})
        if self.mo_temp_source.find('ref_mv_y') != -1:
            self.implicit_buffer.update({"ref_mv_y": None})
        if not self.res_no_implicit:
            self.implicit_buffer.update({"ref_feature": None, "ref_y": None})

        with torch.no_grad():
            rec_frame, _ = self.if_model(batch[:, 0])
            ref_frame = rec_frame
            
        Weight, Denominator = loss_on['D'].split('/')
        Denominator = int(Denominator)

        for frame_idx in range(1, max_num_Pframe+1):
            
            ref_frame = batch[:, frame_idx-1] # Take ground frame
            coding_frame = batch[:, frame_idx]
            
            likelihood_m, m_info = self.motion_forward(ref_frame, coding_frame, frame_idx!=1, False)
            del likelihood_m

            if self.res_no_implicit:
                fa_idx = 0
            else:
                fa_idx = self.args.index_map_training[frame_idx % self.args.rate_gop_size]

            aux_buf = {
                "mv_hat": m_info['flow_hat'],
                "ref_frame": ref_frame,
            }
            if not self.res_no_implicit:
                aux_buf.update({"ref_feature": None})

            del m_info

            context1, context2, context3, _, mc_frame = self.Residual.motion_compensation(aux_buf, aux_buf["mv_hat"], fa_idx, Pretrain=True)

            result = {
                'train/mc_distortion0'       : self.criterion(coding_frame, mc_frame).mean(),
                'train/mc_distortion1'       : self.criterion(coding_frame, context1).mean(),
                'train/mc_distortion2'       : self.criterion(coding_frame, context2).mean(),
                'train/mc_distortion3'       : self.criterion(coding_frame, context3).mean(),
            }
            
            del context1, context2, context3, mc_frame

            distortion = torch.tensor(0., dtype=torch.float, device=batch.device)
            for idx, term in enumerate(Weight.split('_')):
                if self.args.ssim:
                    if not np.isnan(result[f'train/mc_distortion{idx}'].item()):
                        distortion += float(term) * (1 - result[f'train/mc_distortion{idx}']) / (32 * math.sqrt(2))
                    else:
                        if idx == 0:
                            print(f"train/mc_distortion{idx}={result[f'train/mc_distortion{idx}'].item()}")
                        Denominator = Denominator - 1 if Denominator != 1 else 1
                else:
                    distortion += float(term) * result[f'train/mc_distortion{idx}']
                
            distortion /= Denominator
            total_loss += distortion

            for k, v in result.items():
                if k not in log:
                    log[k] = []
                
                log[k].append(v)

        del rec_frame, ref_frame, coding_frame

        for k in log.keys():
            log[k] = sum(log[k]) / len(log[k])

        total_loss /= max_num_Pframe
        log.update({'train/loss': total_loss})
        
        return total_loss, log

    def trainRNN(self, batch, loss_on, RNN=False, max_num_Pframe=5):   
        total_loss = torch.tensor(0., dtype=torch.float, device=batch.device)
        log = {}

        self.implicit_buffer = {}
        if self.mo_cond_source.find('ref_mv_feature') != -1:
            self.implicit_buffer.update({"ref_mv_feature": None})
        if self.mo_temp_source.find('ref_mv_y') != -1:
            self.implicit_buffer.update({"ref_mv_y": None})
        if not self.res_no_implicit:
            self.implicit_buffer.update({"ref_feature": None, "ref_y": None})

        with torch.no_grad():
            rec_frame, _ = self.if_model(batch[:, 0])
            ref_frame = rec_frame

        for frame_idx in range(1, max_num_Pframe+1):
            
            if not RNN and not self.args.FGOP_Inter_XtRNN:
                ref_frame = ref_frame.detach()

            coding_frame = batch[:, frame_idx]
            rec_frame, likelihoods, m_info, r_info = self(ref_frame, coding_frame, frame_idx!=1, RNN, frame_idx)
            
            m_rate = estimate_bpp(likelihoods[:2], input=m_info['flow']).mean()
            r_rate = estimate_bpp(likelihoods[2:], input=coding_frame).mean()

            result = {
                'train/m_rate'              : m_rate,
                'train/r_rate'              : r_rate,
                'train/rate'                : m_rate + r_rate,
                'train/warped_distortion'   : self.criterion(coding_frame, m_info['warped_frame']).mean(),
                'train/mc_distortion'       : self.criterion(coding_frame, r_info['mc_frame']).mean(),
                'train/distortion'          : self.criterion(coding_frame, rec_frame).mean(),
            }

            if (not self.res_no_implicit) and self.args.Pretrain_frame_feature:
                result.update({'train/feature_frame_distortion': self.criterion(coding_frame, r_info['feature_frame']).mean()})
            
            rate = torch.tensor(0., dtype=torch.float, device=rec_frame.device)
            for term in loss_on['R'].split('/'):
                coefficient = 1
                if '*' in term:
                    coefficient = float(term.split('*')[0])
                    term = term.split('*')[1]

                if term == 'None':
                    continue
                
                rate += coefficient * result['train/'+term]

            distortion = torch.tensor(0., dtype=torch.float, device=rec_frame.device)
            for term in loss_on['D'].split('/'):
                coefficient = 1
                if '*' in term:
                    if term.split('*')[0] == 'frame_weight':
                        coefficient = self.frame_weight[frame_idx]
                    else: 
                        coefficient = float(term.split('*')[0])
                    term = term.split('*')[1]

                if term == 'None':
                    continue
                
                if self.args.ssim:
                    distortion += coefficient * (1 - result['train/'+term]) / (32 * math.sqrt(2))
                else: 
                    distortion += coefficient * result['train/'+term]

            loss = rate + 255**2 * lmda[self.args.quality_level] * distortion
            total_loss += loss

            ref_frame = rec_frame

            for k, v in result.items():
                if k not in log:
                    log[k] = []
                
                log[k].append(v)

        for k in log.keys():
            log[k] = sum(log[k]) / len(log[k])

        total_loss /= max_num_Pframe
        log.update({'train/loss': total_loss})
        
        return total_loss, log

    def training_step(self, batch, phase):
        device = next(self.parameters()).device
        batch = batch.to(device)

        self.freeze(self.train_cfg[phase]['frozen_modules'])
        self.activate(self.train_cfg[phase]['re-activate_modules'])

        if self.train_cfg[phase]['strategy']['stage'] == 'Motion_1Pframe':
            loss, result = self.train_Motion_1Pframe(batch,
                                                     self.train_cfg[phase]['loss_on'],
                                                     bool(self.train_cfg[phase]['strategy']['random']))
        elif self.train_cfg[phase]['strategy']['stage'] == '1Pframe':
            loss, result = self.train_1Pframe(batch,
                                              self.train_cfg[phase]['loss_on'],
                                              bool(self.train_cfg[phase]['strategy']['random']))
        elif self.train_cfg[phase]['strategy']['stage'] == '2Pframes':            
            loss, result = self.train_2Pframes(batch, 
                                              self.train_cfg[phase]['loss_on'], 
                                              self.train_cfg[phase]['mode'],
                                              bool(self.train_cfg[phase]['strategy']['random']))

        elif self.train_cfg[phase]['strategy']['stage'] == 'fullgop':
            loss, result = self.trainRNN(batch,
                                         self.train_cfg[phase]['loss_on'],
                                         bool(self.train_cfg[phase]['RNN']), 
                                         self.train_cfg[phase]['max_num_Pframe'])
        elif self.train_cfg[phase]['strategy']['stage'] == 'trainmc':
            loss, result = self.trainMCNet(batch,
                                         self.train_cfg[phase]['loss_on'],
                                         bool(self.train_cfg[phase]['RNN']), 
                                         self.train_cfg[phase]['max_num_Pframe'])

        else:
            raise NotImplementedError

        return loss, result

    def training_step_end(self, result, epoch=None, step=None):
        logs = {}
        for k, v in result.items():
            if k[-10:] == 'distortion':
                if self.args.ssim:
                    logs[k.replace('distortion', 'MS-SSIM')] = v.item()
                else:
                    logs[k.replace('distortion', 'PSNR')] = mse2psnr(v.item())
                # to observe the ratio of mc_distortion and distortion
                if k.split('/')[-1] == 'mc_distortion':
                    logs[k] = v.item()
                elif k.split('/')[-1] == 'distortion':
                    logs[k] = v.item()
            elif k[-11:-1] == 'distortion':
                if self.args.ssim:
                    logs[k.replace('distortion', 'MS-SSIM')] = v.item()
                else:
                    logs[k.replace('distortion', 'PSNR')] = mse2psnr(v.item())
            else:
                if isinstance(v, torch.Tensor):
                    logs[k] = v.item()
                else:
                    logs[k] = v
        
        self.logger.log_metrics(logs, epoch=epoch, step=step)

    @torch.no_grad()
    def validation_step(self, batch, epoch):
        def create_grid(img):
            return make_grid(torch.unsqueeze(img, 1)).cpu().detach().numpy()[0]

        def upload_img(tnsr, tnsr_name, current_epoch, ch="first", grid=True):
            if grid:
                tnsr = create_grid(tnsr)

            self.logger.log_image(tnsr, name=tnsr_name, step=current_epoch,
                                  image_channels=ch, overwrite=True)

        dataset_name, seq_name, batch, _ = batch
        device = next(self.parameters()).device
        
        batch = batch.to(device)
        H, W = batch.shape[-2:]
        batch = transforms.CenterCrop((self.args.crop_factor * (H // self.args.crop_factor), self.args.crop_factor * (W // self.args.crop_factor)))(batch)

        seq_name = seq_name
        dataset_name = dataset_name

        gop_size = batch.size(1)

        m_rate_list = [[] for _ in range(len(dataset_name))]
        rate_list = [[] for _ in range(len(dataset_name))]
        distortion_list = [[] for _ in range(len(dataset_name))]
        warped_distortion_list = [[] for _ in range(len(dataset_name))]
        mc_distortion_list = [[] for _ in range(len(dataset_name))]
        loss_list = [[] for _ in range(len(dataset_name))]
        if (not self.res_no_implicit) and self.args.Pretrain_frame_feature:
            feature_frame_psnr_list = [[] for _ in range(len(dataset_name))]

        for frame_idx in range(gop_size):
            coding_frame = batch[:, frame_idx]

            # I frame
            if frame_idx == 0:

                self.implicit_buffer = {}
                if self.mo_cond_source.find('ref_mv_feature') != -1:
                    self.implicit_buffer.update({"ref_mv_feature": None})
                if self.mo_temp_source.find('ref_mv_y') != -1:
                    self.implicit_buffer.update({"ref_mv_y": None})
                if not self.res_no_implicit:
                    self.implicit_buffer.update({"ref_feature": None, "ref_y": None})
                
                rec_frame, likelihoods = self.if_model(coding_frame)
                rec_frame = rec_frame.clamp(0, 1)

                for i in range(len(dataset_name)):
                    r_y = estimate_bpp(likelihoods[0][i:i+1], input=rec_frame).mean().cpu().item()
                    r_z = estimate_bpp(likelihoods[1][i:i+1], input=rec_frame).mean().cpu().item()
                    rate = r_y + r_z

                    if self.args.ssim:
                        distortion = self.criterion(rec_frame[i:i+1], coding_frame[i:i+1]).mean().cpu().item()

                        loss = rate + 255**2 * lmda[self.args.quality_level] * (1 - distortion) / (32 * math.sqrt(2))
                    
                    else:
                        mse = self.criterion(rec_frame[i:i+1], coding_frame[i:i+1]).mean().cpu().item()
                        distortion = mse2psnr(mse)

                        loss = rate + 255**2 * lmda[self.args.quality_level] * mse

                    rate_list[i].append(rate)
                    distortion_list[i].append(distortion)
                    loss_list[i].append(loss)
            
            # P frame
            else:
                rec_frame, likelihoods, m_info, r_info = self(ref_frame, coding_frame, frame_idx!=1, RNN=False, frame_idx=frame_idx)
                rec_frame = rec_frame.clamp(0, 1)
                warped_frame = m_info['warped_frame'].clamp(0, 1)
                mc_frame = r_info['mc_frame'].clamp(0, 1)

                for i, seq in enumerate(seq_name):
                    m_y = estimate_bpp(likelihoods[0][i:i+1], input=m_info['flow']).mean().cpu().item()
                    m_z = estimate_bpp(likelihoods[1][i:i+1], input=m_info['flow']).mean().cpu().item()
                    r_y = estimate_bpp(likelihoods[2][i:i+1], input=coding_frame).mean().cpu().item()
                    r_z = estimate_bpp(likelihoods[3][i:i+1], input=coding_frame).mean().cpu().item()

                    m_rate = m_y + m_z
                    r_rate = r_y + r_z
                    rate = m_rate + r_rate

                    if self.args.ssim: 
                        distortion = self.criterion(rec_frame[i:i+1], coding_frame[i:i+1]).mean().item()
                        warped_distortion = self.criterion(warped_frame[i:i+1], coding_frame[i:i+1]).mean().item()
                        mc_distortion = self.criterion(mc_frame[i:i+1], coding_frame[i:i+1]).mean().item()

                        loss = rate + 255**2 * lmda[self.args.quality_level] * (1 - distortion) / (32 * math.sqrt(2))
                        
                    else:
                        mse = self.criterion(rec_frame[i:i+1], coding_frame[i:i+1]).mean().item()
                        distortion = mse2psnr(mse)
                        
                        warped_mse = self.criterion(warped_frame[i:i+1], coding_frame[i:i+1]).mean().item()
                        warped_distortion = mse2psnr(warped_mse)
                        
                        mc_mse = self.criterion(mc_frame[i:i+1], coding_frame[i:i+1]).mean().item()
                        mc_distortion = mse2psnr(mc_mse)

                        loss = rate + 255**2 * lmda[self.args.quality_level] * mse

                    rate_list[i].append(rate)
                    distortion_list[i].append(distortion)
                    loss_list[i].append(loss)
                    m_rate_list[i].append(m_rate)
                    warped_distortion_list[i].append(warped_distortion)
                    mc_distortion_list[i].append(mc_distortion)

                    if (not self.res_no_implicit) and self.args.Pretrain_frame_feature:
                        feature_frame_mse = self.criterion(r_info['feature_frame'][i:i+1], coding_frame[i:i+1]).mean().item()
                        feature_frame_psnr = mse2psnr(feature_frame_mse)
                        feature_frame_psnr_list[i].append(feature_frame_psnr)

                    if frame_idx == 5:
                        if seq not in ['Beauty', 'Jockey', 'Kimono1', 'HoneyBee']:
                            continue

                        flow = m_info['flow'][i]
                        flow_rgb = torch.from_numpy(
                            fz.convert_from_flow(flow.permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                        upload_img(flow_rgb.cpu().numpy(), f'{seq}_{epoch}_est_flow.png', current_epoch=epoch, grid=False)

                        flow_hat = m_info['flow_hat'][i]
                        flow_rgb = torch.from_numpy(
                            fz.convert_from_flow(flow_hat.permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                        upload_img(flow_rgb.cpu().numpy(), f'{seq}_{epoch}_flow_hat.png', current_epoch=epoch, grid=False)

                        upload_img(warped_frame[i].cpu().numpy(), f'{seq}_{epoch}_{warped_distortion:.3f}_warped_frame.png', current_epoch=epoch, grid=False)
                        upload_img(ref_frame[i].cpu().numpy(), f'{seq}_{epoch}_ref_frame.png', current_epoch=epoch, grid=False)
                        upload_img(coding_frame[i].cpu().numpy(), f'{seq}_{epoch}_gt_frame.png', current_epoch=epoch, grid=False)
                        upload_img(rec_frame[i].cpu().numpy(), f'{seq}_{epoch}_{distortion:.3f}_rec_frame.png', current_epoch=epoch, grid=False)
                        upload_img(mc_frame[i].cpu().numpy(), f'{seq}_{epoch}_{mc_distortion:.3f}_mc_frame.png', current_epoch=epoch, grid=False)

                        if (not self.res_no_implicit) and self.args.Pretrain_frame_feature:
                            upload_img(r_info['feature_frame'][i].cpu().numpy(), f'{seq}_{epoch}_feature_frame.png', current_epoch=epoch, grid=False)

                        # upload rate map
                        cm = plt.get_cmap('hot')
                        lll = lower_bound(likelihoods[0][i:i+1], 1e-9).log() / -np.log(2.)
                        rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                        upload_img(np.transpose(rate_map[:, :, :3], (2, 0, 1)), f'{seq}_{epoch}_my.png', current_epoch=epoch, grid=False)

                        lll = lower_bound(likelihoods[1][i:i+1], 1e-9).log() / -np.log(2.)
                        rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                        upload_img(np.transpose(rate_map[:, :, :3], (2, 0, 1)), f'{seq}_{epoch}_mz.png', current_epoch=epoch, grid=False)

                        lll = lower_bound(likelihoods[2][i:i+1], 1e-9).log() / -np.log(2.)
                        rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                        upload_img(np.transpose(rate_map[:, :, :3], (2, 0, 1)), f'{seq}_{epoch}_ry.png', current_epoch=epoch, grid=False)

                        lll = lower_bound(likelihoods[3][i:i+1], 1e-9).log() / -np.log(2.)
                        rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                        upload_img(np.transpose(rate_map[:, :, :3], (2, 0, 1)), f'{seq}_{epoch}_rz.png', current_epoch=epoch, grid=False)

                        upload_img(r_info['res'][i].cpu().numpy(), f'{seq}_{epoch}_res.png', current_epoch=epoch, grid=False)
                        upload_img(r_info['res_hat'][i].cpu().numpy(), f'{seq}_{epoch}_res_hat.png', current_epoch=epoch, grid=False)

                        if self.res_add_mask:
                            # upload_img(r_info['mask'][i].cpu().numpy(), f'{seq}_{epoch}_mask.png', current_epoch=epoch, grid=False)

                            m_lower = np.quantile(r_info['mask'][i].cpu().numpy()[0], 0.00005)
                            m_upper = np.quantile(r_info['mask'][i].cpu().numpy()[0], 0.99995)
                            fig, ax = plt.subplots()
                            plt_mask = ax.imshow(r_info['mask'][i].cpu().numpy()[0], cmap='Oranges_r', vmin=m_lower, vmax=m_upper)
                            ax.axis('off')
                            fig.colorbar(mappable=plt_mask, shrink=0.55, pad=0.01)
                            fig.tight_layout()
                            # ax.margins(0)
                            canvas = FigureCanvas(fig)
                            canvas.draw()
                            image_mask = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
                            image_mask = image_mask.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                            upload_img(image_mask.transpose(2, 0, 1), f'{seq}_{epoch}_mask_plt.png', current_epoch=epoch, grid=False)
                            plt.close()

            ref_frame = rec_frame

        m_rate              = [np.mean(l) for l in m_rate_list]
        rate                = [np.mean(l) for l in rate_list]
        distortion          = [np.mean(l) for l in distortion_list]
        warped_distortion   = [np.mean(l) for l in warped_distortion_list]
        mc_distortion       = [np.mean(l) for l in mc_distortion_list]
        loss                = [np.mean(l) for l in loss_list]

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 
                'val/loss': loss, 'val/rate': rate, 'val/m_rate': m_rate}
        
        if self.args.ssim:
            logs.update({'val/ms-ssim': distortion, 'val/mc_ms-ssim': mc_distortion, 'val/warped_ms-ssim': warped_distortion})
        else:
            logs.update({'val/psnr': distortion, 'val/mc_psnr': mc_distortion, 'val/warped_psnr': warped_distortion})

        if (not self.res_no_implicit) and self.args.Pretrain_frame_feature:
            feature_frame_psnr = [np.mean(l) for l in feature_frame_psnr_list]
            logs.update({'val/feature_frame_psnr': feature_frame_psnr})

        return logs

    @torch.no_grad()
    def validation_epoch_end(self, outputs, epoch):
        if self.args.ssim: 
            dist_metric = 'ms-ssim'
        else: 
            dist_metric = 'psnr'
        
        metrics_name = [dist_metric, f'warped_{dist_metric}', f'mc_{dist_metric}', 'rate', 'm_rate', 'loss']
        
        if (not self.res_no_implicit) and self.args.Pretrain_frame_feature:
            metrics_name.append('feature_frame_psnr')
        
        rd_dict = {}
        loss = []

        for logs in outputs:
            for i in range(len(logs['dataset_name'])):
                dataset_name = logs['dataset_name'][i]

                if not (dataset_name in rd_dict.keys()):
                    rd_dict[dataset_name] = {}
                    for metric in metrics_name:
                        rd_dict[dataset_name][metric] = []

                for metric in metrics_name:
                    rd_dict[dataset_name][metric].append(logs[f'val/{metric}'][i])
    
                loss.append(logs['val/loss'][i])

        avg_loss = np.mean(loss)
        
        logs = {'val/loss': avg_loss}

        for dataset_name, rd in rd_dict.items():
            for metric in metrics_name:
                logs[f'val/{dataset_name} {metric}'] = np.mean(rd[metric])

        self.logger.log_metrics(logs, epoch=epoch)

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
            os.makedirs(self.args.save_dir + f'/{seq_name}/flow', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/flow_hat', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/warped_frame', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/likelihood_r', exist_ok=True)
            # os.makedirs(self.args.save_dir + f'/{seq_name}/likelihood_m', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/res', exist_ok=True)
            os.makedirs(self.args.save_dir + f'/{seq_name}/res_hat', exist_ok=True)
            if self.res_add_mask:
                os.makedirs(self.args.save_dir + f'/{seq_name}/mask', exist_ok=True)
            # if not self.res_no_implicit:
            #     os.makedirs(self.args.save_dir + f'/{seq_name}/ref_feature', exist_ok=True)
            #     os.makedirs(self.args.save_dir + f'/{seq_name}/ref_mv_feature', exist_ok=True)

        for frame_idx in range(gop_size):
            TO_VISUALIZE = TO_VISUALIZE_Flag and frame_id_start == 1 and frame_idx < 8 and seq_name in ['BasketballDrive', 'Kimono1', 'HoneyBee', 'Jockey']
            SCENE_CUT = {
                'Kimono1': [141], 'Kimono1_10b_rgb': [141],
                'videoSRC20': [43, 81, 106], 'videoSRC21': [49, 64, 90], 'videoSRC25': [18, 84], 'videoSRC26': [62], 'videoSRC27': [40], 'videoSRC29': [71]
            }

            coding_frame = batch[:, frame_idx]

            # pad if necessary
            height, width = coding_frame.size(2), coding_frame.size(3)
            padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, self.args.align_factor)
            coding_frame_pad = F.pad(coding_frame, (padding_l, padding_r, padding_t, padding_b), mode="replicate")

            if (not self.res_no_implicit) and (not self.training) and self.args.reset_interval > 0 and frame_idx % self.args.reset_interval == 1:
                self.implicit_buffer["ref_feature"] = None

            # I frame
            if frame_idx == 0 or (self.args.remove_scene_cut and seq_name in SCENE_CUT and frame_id_start + frame_idx in SCENE_CUT[seq_name]):

                self.implicit_buffer = {}
                if self.mo_cond_source.find('ref_mv_feature') != -1:
                    self.implicit_buffer.update({"ref_mv_feature": None})
                if self.mo_temp_source.find('ref_mv_y') != -1:
                    self.implicit_buffer.update({"ref_mv_y": None})
                if not self.res_no_implicit:
                    self.implicit_buffer.update({"ref_feature": None, "ref_y": None})

                rec_frame, likelihoods = self.if_model(coding_frame_pad)
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
                
                log_list.append({f'{dist_metric}': distortion, 'Rate': rate, 
                                 f'MC-{dist_metric}': 0, f'Warped-{dist_metric}': 0,
                                 'my': 0, 
                                 'mz': 0,
                                 'ry': 0, 
                                 'rz': 0})
            # P frame
            else:
                if self.args.compute_macs and frame_idx != 1:
                    def dummy_inputs(shape):
                        inputs = torch.ones(shape).cuda()
                        return {
                            'ref_frame'     : inputs, 
                            'coding_frame'  : inputs, 
                            'predict'       : True,
                            'frame_idx'     : 2
                        }
                    
                    macs, _ = get_model_complexity_info(self, tuple(ref_frame.shape), as_strings=False, input_constructor=dummy_inputs)
                    print("========================================")
                    print("MACs:", macs)
                    print(macs / 1e9, "GMac")
                    print(macs / height / width / 1e3, "kMACs/pixel")
                    print(f"coding_frame shape: {(height, width)}")
                    exit()

                predict = False if (self.args.remove_scene_cut and (seq_name in SCENE_CUT) and ((frame_id_start + frame_idx - 1) in SCENE_CUT[seq_name])) else (frame_idx!=1)
                rec_frame, likelihoods, m_info, r_info = self(ref_frame, coding_frame_pad, predict, RNN=False, frame_idx=frame_idx)
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
                    flow_map = plot_flow(m_info['flow'])
                    save_image(flow_map,
                               self.args.save_dir + f'/{seq_name}/flow/'
                                                    f'frame_{int(frame_id_start + frame_idx)}_flow.png', nrow=1)
                    flow_map = plot_flow(m_info['flow_hat'])
                    save_image(flow_map,
                               self.args.save_dir + f'/{seq_name}/flow_hat/'
                                                    f'frame_{int(frame_id_start + frame_idx)}_flow.png', nrow=1)
                    save_image(coding_frame[0], 
                               self.args.save_dir + f'/{seq_name}/gt_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_frame[0], 
                               self.args.save_dir + f'/{seq_name}/mc_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(warped_frame[0], 
                               self.args.save_dir + f'/{seq_name}/warped_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(rec_frame[0], 
                               self.args.save_dir + f'/{seq_name}/rec_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')


                    # if not self.res_no_implicit:
                    #     ref_feature = make_grid(torch.transpose(self.implicit_buffer["ref_feature"], 0, 1), nrow=1)
                    #     save_image(ref_feature, self.args.save_dir + f'/{seq_name}/ref_feature/frame_{int(frame_id_start + frame_idx)}.png')
                        

                    #     ref_mv_feature = make_grid(torch.transpose(self.implicit_buffer["ref_mv_feature"], 0, 1), nrow=1)
                    #     save_image(ref_mv_feature, self.args.save_dir + f'/{seq_name}/ref_mv_feature/frame_{int(frame_id_start + frame_idx)}.png')

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

                    save_image(r_info['res'][0], 
                               self.args.save_dir + f'/{seq_name}/res/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    
                    save_image(r_info['res_hat'][0], 
                               self.args.save_dir + f'/{seq_name}/res_hat/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')

                    if self.res_add_mask:
                        m_lower = np.quantile(r_info['mask'].cpu().numpy()[0].squeeze(0), 0.00005)
                        m_upper = np.quantile(r_info['mask'].cpu().numpy()[0].squeeze(0), 0.99995)
                        plt.imshow(r_info['mask'].cpu().numpy()[0].squeeze(0), cmap='Oranges_r', vmin=m_lower, vmax=m_upper)
                        plt.axis('off')
                        plt.colorbar(shrink=0.55, pad=0.01)
                        plt.savefig(self.args.save_dir + f'/{seq_name}/mask/'
                                    f'frame_{int(frame_id_start + frame_idx)}.png',
                                    bbox_inches='tight', pad_inches=0.01)
                        plt.close()
                    
                    # del flow_map, rate_map, lll, cm
                    del flow_map

                del r_info, m_info, mc_frame, warped_frame, coding_frame
                
                log_list.append({f'{dist_metric}': distortion, 'Rate': rate, 
                                 f'MC-{dist_metric}': mc_distortion, f'Warped-{dist_metric}': warped_distortion,
                                 'my': m_y, 
                                 'mz': m_z,
                                 'ry': r_y, 
                                 'rz': r_z})

                metrics['Mo_Rate'].append(m_rate)
                metrics[f'MC-{dist_metric}'].append(mc_distortion)
                metrics[f'Warped-{dist_metric}'].append(warped_distortion)
                metrics['Mo_Ratio'].append(m_rate/rate)
                metrics['Res_Ratio'].append(r_rate/rate)

            del likelihoods, x_hat
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

        if self.args.full_seq:
            summary_txt_name += '_FullSeq'
        
        with open(self.args.save_dir + f'/{summary_txt_name}.txt', 'w', newline='') as report:
            report.write(print_log)

        self.logger.log_metrics(logs)

    def optimizer_step(self):
        def clip_gradient(opt, grad_clip):
            for group in opt.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        clip_gradient(self.optimizer, 5)

        self.optimizer.step()

    def configure_optimizers(self, lr, include_module_name=None, exclude_module_name=None):
        assert not (include_module_name is not None and exclude_module_name is not None)
        if include_module_name is not None:
            self.optimizer = optim.Adam(params=self.include_main_parameters(include_module_name), lr=lr)
        elif exclude_module_name is not None:
            self.optimizer = optim.Adam(params=self.exclude_main_parameters(exclude_module_name), lr=lr)
        else:
            self.optimizer = optim.Adam(params=self.main_parameters(), lr=lr)
    
    def configure_lr_scheduler(self, milestones=[300, 500], gamma=0.1):
        lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        return lr_scheduler

    def setup(self, stage, max_num_Pframe=6, epoch_ratio=1):

        if stage == 'fit':
            transformer = transforms.Compose([
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            
            if self.args.train_dataset == ['vimeo_septuplet']:
                self.train_dataset = VideoData(self.args.dataset_root + "/vimeo_septuplet/", max_num_Pframe+1, transform=transformer, epoch_ratio=epoch_ratio) 
            elif self.args.train_dataset == ['BVI-DVC']:
                self.train_dataset = BVI_Dataset(f"{self.args.dataset_root}/BVI-DVC/", max_num_Pframe+1, transform=transformer, epoch_ratio=epoch_ratio)
            
            self.val_dataset = VideoTestData(self.args.dataset_root, sequence=self.args.test_dataset, first_gop=True, GOP=self.args.gop)
        
        elif stage == 'test':
            self.test_dataset = VideoTestData(self.args.dataset_root, sequence=self.args.test_dataset, GOP=self.args.gop, test_seq_len=self.args.test_seq_len, use_seqs=self.args.test_seqs, full_seq=self.args.full_seq, color_transform=self.args.color_transform)

        else:
            raise NotImplementedError

    def train_dataloader(self, batch_size):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=batch_size,
                                  num_workers=self.args.num_workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        return train_loader

    def val_dataloader(self, batch_size):
        val_loader = DataLoader(self.val_dataset,
                                batch_size=batch_size,
                                num_workers=self.args.num_workers,
                                shuffle=False,
                                drop_last=True)
        return val_loader

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
    parser.add_argument('--MENet', type=str, default='SPy')
    parser.add_argument('--train_conf', type=str, default=None)
    parser.add_argument('--if_coder', type=str, default='Intra')
    parser.add_argument('--cond_motion_coder_conf', type=str, default=None)
    parser.add_argument('--residual_coder_conf', type=str, default=None)
    parser.add_argument("-n", "--num_workers", type=int, default=4, help="Dataloaders threads (default: %(default)s)")
    parser.add_argument("-q", "--quality_level", type=int, default=6, help="Quality level (default: %(default)s)")
    parser.add_argument("-i", "--iframe_quality", type=int, default=None, help="Quality level of i frame coder (default: %(default)s)")
    parser.add_argument('--ssim', action='store_true', help="Optimize for MS-SSIM")

    parser.add_argument("--gpus", type=int, default=1, help="Number of GPU (default: %(default)s)")
    parser.add_argument('--save_dir', default=None, help='directory for saving testing result')
    parser.add_argument('--gop', default=32, type=int)
    parser.add_argument('--epoch_ratio', default=1, type=float)

    parser.add_argument('--project_name', type=str, default='HyTIP')
    parser.add_argument('--experiment_name', type=str, default='HyTIP')
    parser.add_argument('--restore', type=str, default='none', choices=['none', 'load', 'resume', 'load_pretrained', 'CRC_to_MCR', 'Motion_Hybrid', 'Inter_Hybrid', 'ReduceMotionBufferSize', 'ReduceInterBufferSize', 'FA', 'CTM', 'context'])
    parser.add_argument('--restore_exp_key', type=str, default=None)
    parser.add_argument('--restore_exp_epoch', type=int, default=None)
    parser.add_argument("--end_epoch", type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_sanity', action='store_true')

    parser.add_argument('--compute_macs', action='store_true')
    parser.add_argument('--compute_decode_macs', action='store_true')
    parser.add_argument('--compute_model_size', action='store_true')
    parser.add_argument('-s', '--remove_scene_cut', action='store_true')
    parser.add_argument('--feature_channel', type=int, default=48)
    parser.add_argument('--FGOP_Inter_XtRNN', action='store_true')
    parser.add_argument('--Pretrain_frame_feature', action='store_true')

    parser.add_argument('-data', '--dataset_root', type=str, default=None, help='Root for dataset')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-c', '--test_crop', action='store_true')
    parser.add_argument('--crop_factor', type=int, default=64)
    parser.add_argument('--align_factor', type=int, default=16)
    parser.add_argument('--test_dataset', type=str, nargs='+', default=['HEVC-B', 'UVG'])
    parser.add_argument('--test_seqs', type=str, nargs='+', choices=list(SEQUENCES.keys()), default=[])
    parser.add_argument('--test_seq_len', type=int, default=96)
    parser.add_argument('--full_seq', action='store_true')
    parser.add_argument('--color_transform', type=str, default='BT601', choices=['BT601', 'BT709'])
    parser.add_argument('-vis', '--visualize', action='store_true')

    parser.add_argument('--frame_weight', type=float, nargs='+', default=None)
    parser.add_argument("--index_map_training", type=int, nargs='+', default=[0, 0, 0, 0, 0, 0, 0, 0])
    parser.add_argument("--index_map", type=int, nargs='+', default=[0, 1, 0, 2, 0, 2, 0, 2])
    parser.add_argument("--rate_gop_size", type=int, default=8, choices=[4, 8])
    parser.add_argument('--reset_interval', type=int, default=32, required=False)
    
    # for long sequence training
    parser.add_argument('--train_dataset', type=str, nargs='+', default=['vimeo_septuplet'], choices=['vimeo_septuplet', 'BVI-DVC'])

    args = parser.parse_args(argv)

    return args

def main(argv):
    args = parse_args(argv)
    assert args.gpus <= torch.cuda.device_count(), "Can't find enough gpus in the machine."

    if args.ssim:
        for k, v in lmda.items():
            if k < 5:
                lmda[k] = int(v * (10**4)) * 125 / 10**6

    if args.save_dir is None:
        os.makedirs('results', exist_ok=True)
        args.save_dir = os.path.join('results', args.experiment_name + '-' + str(args.quality_level))

    assert os.path.exists('./user_info/key.yml')
    user_info = yaml.safe_load(open('./user_info/key.yml', 'r'))

    exp = Experiment if args.restore != 'resume' else ExistingExperiment
    experiment = exp(
        api_key=user_info['api_key'],
        project_name=args.project_name,
        workspace=user_info['workspace'],
        experiment_key = None if args.restore != 'resume' else args.restore_exp_key,
        disabled=args.debug or args.test
    )

    experiment.set_name(f'{args.experiment_name}-{args.quality_level}')
    experiment.log_parameters(vars(args))
    ckpt_dir = os.path.join('results', experiment.get_key(), 'checkpoints')
    if not (args.test or args.debug):
        os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(888888)
    
    gpu_ids = [0]
    for i in range(1, args.gpus):
        gpu_ids.append(i)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # log model config
    for cfg in [args.cond_motion_coder_conf, args.residual_coder_conf]:
        if os.path.exists(cfg):
            experiment.log_code(cfg)

    # train config
    if args.train_conf is not None:
        experiment.log_code(args.train_conf)
        with open(args.train_conf, 'r') as jsonfile:
            train_cfg = json.load(jsonfile)
    else:
        train_cfg = None

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
    
    model = Pframe(args, cond_mo_coder, res_coder, train_cfg, experiment).to(device)

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

    elif args.restore == 'load':
        checkpoint = torch.load(os.path.join('results', args.restore_exp_key, 'checkpoints', f'epoch={args.restore_exp_epoch}.pth.tar'), map_location=device)
        current_epoch = checkpoint['epoch'] + 1

        ckpt = {}
        for k, v in checkpoint["state_dict"].items():
            k = k.split('.')
            if k[0] != 'feature_frame_generate' and k[0] != 'criterion':
                k.pop(1)
            if k[0] == 'criterion':
                continue
            
            ckpt['.'.join(k)] = v

        model.load_state_dict(ckpt, strict=True)

    elif args.restore == 'resume':
        checkpoint = torch.load(os.path.join('results', args.restore_exp_key, 'checkpoints', f'epoch={args.restore_exp_epoch}.pth.tar'), map_location=device)
        current_epoch = checkpoint['epoch'] + 1

        ckpt = {}
        for k, v in checkpoint["state_dict"].items():
            k = k.split('.')
            if k[0] != 'feature_frame_generate' and k[0] != 'criterion':
                k.pop(1)
            if k[0] == 'criterion':
                continue
            
            ckpt['.'.join(k)] = v

        model.load_state_dict(ckpt, strict=True)
    
    elif args.restore == 'load_pretrained':
        current_epoch = 1
        
        ckpt = {}
        intra_checkpoint = torch.load(os.path.join('./models/cvpr2023_image_psnr.pth.tar'), map_location=device)
        for k, v in intra_checkpoint.items():
            k = k.split('.')
            ckpt['if_model.net.'+ '.'.join(k)] = v

        ME_checkpoint = torch.load(os.path.join('./models/spy_net-sintel-final.pytorch'), map_location=device)
        for k, v in ME_checkpoint.items():
            ckpt['MENet.net.'+k] = v

        inter_checkpoint = torch.load(os.path.join('./models/cvpr2024_video.pth.tar'), map_location=device)
        for k, v in inter_checkpoint.items():
            if k.find('align') != -1:
                continue
            k = k.split('.')
            if k[0] in ['optic_flow', 'feature_adaptor', 'y_q_enc', 'y_q_dec', 'y_spatial_prior', 'y_spatial_prior_adaptor_1', 'y_spatial_prior_adaptor_2', 'y_spatial_prior_adaptor_3', 'y_prior_fusion_adaptor_1', 'mv_y_q_enc', 'mv_y_q_dec']:
                continue
            elif 'mv' in k[0]:
                if 'spatial_prior' in k[0]:
                    continue
                if k[0] == 'mv_y_prior_fusion_adaptor_0':
                    k[0] = 'mv_y_prior_fusion_adaptor_temp_0'
                elif k[0] == 'mv_y_prior_fusion_adaptor_1':
                    k[0] = 'mv_y_prior_fusion_adaptor_temp_1'
                elif k[0] == 'mv_y_prior_fusion':
                    k[0] = 'PA'
                ckpt['CondMotion.net.'+ '.'.join(k)] = v
                continue
            else:
                ckpt['Residual.net.' + '.'.join(k)] = v
        
        model.load_state_dict(ckpt, strict=False)

    elif args.restore == 'CRC_to_MCR':
        checkpoint = torch.load(os.path.join('results', args.restore_exp_key, 'checkpoints', f'epoch={args.restore_exp_epoch}.pth.tar'), map_location=device)
        current_epoch = 1

        ckpt = {}
        for k, v in checkpoint["state_dict"].items():
            if k.find('mc_generate') != -1:
                continue
            k = k.split('.')
            if k[0] != 'feature_frame_generate' and k[0] != 'criterion':
                k.pop(1)
            
            ckpt['.'.join(k)] = v

        model.load_state_dict(ckpt, strict=False)

    elif args.restore == 'Motion_Hybrid':
        checkpoint = torch.load(os.path.join('results', args.restore_exp_key, 'checkpoints', f'epoch={args.restore_exp_epoch}.pth.tar'), map_location=device)
        current_epoch = 1

        ckpt = {}
        for k, v in checkpoint["state_dict"].items():
            k = k.split('.')
            if k[0] != 'feature_frame_generate' and k[0] != 'criterion':
                k.pop(1)
            if k[0] == 'CondMotion' and k[2] == 'mv_y_prior_fusion_adaptor_temp_0':
                k[2] = 'temp_prior_fusion_adaptor_0'
                ckpt['.'.join(k)] = v
                continue
            if k[0] == 'CondMotion' and k[2] in ['mv_y_prior_fusion_adaptor_temp_1', 'PA']:
                continue
            
            ckpt['.'.join(k)] = v

        model.load_state_dict(ckpt, strict=False)

    elif args.restore == 'Inter_Hybrid':
        checkpoint = torch.load(os.path.join('results', args.restore_exp_key, 'checkpoints', f'epoch={args.restore_exp_epoch}.pth.tar'), map_location=device)
        current_epoch = 1

        ckpt = {}
        for k, v in checkpoint["state_dict"].items():
            k = k.split('.')
            if k[0] != 'feature_frame_generate' and k[0] != 'criterion':
                k.pop(1)
            
            ckpt['.'.join(k)] = v

        model.load_state_dict(ckpt, strict=False)

    elif args.restore == 'ReduceMotionBufferSize':
        checkpoint = torch.load(os.path.join('results', args.restore_exp_key, 'checkpoints', f'epoch={args.restore_exp_epoch}.pth.tar'), map_location=device)
        current_epoch = 1

        ckpt = {}
        for k, v in checkpoint["state_dict"].items():
            k = k.split('.')
            if k[0] != 'feature_frame_generate' and k[0] != 'criterion':
                k.pop(1)
            if len(k) >= 2 and k[0] == 'CondMotion' and k[2] =='feature_generator':
                continue
            if len(k) >= 2 and k[0] == 'CondMotion' and k[2] =='feature_extractor' and k[3] == 'fusion':
                continue
            
            ckpt['.'.join(k)] = v

        model.load_state_dict(ckpt, strict=False)

    elif args.restore == 'ReduceInterBufferSize':
        checkpoint = torch.load(os.path.join('results', args.restore_exp_key, 'checkpoints', f'epoch={args.restore_exp_epoch}.pth.tar'), map_location=device)
        current_epoch = 1

        ckpt = {}
        for k, v in checkpoint["state_dict"].items():
            k = k.split('.')
            if k[0] != 'feature_frame_generate' and k[0] != 'criterion':
                k.pop(1)
            if len(k) >= 2 and k[0] == 'Residual' and k[2] in ['frame_feature_generator', 'feature_adaptor']:
                continue
            if k[0] == 'feature_frame_generate':
                continue
            ckpt['.'.join(k)] = v

        model.load_state_dict(ckpt, strict=True)

    elif args.restore == 'FA':
        checkpoint = torch.load(os.path.join('results', args.restore_exp_key, 'checkpoints', f'epoch={args.restore_exp_epoch}.pth.tar'), map_location=device)
        current_epoch = 1

        ckpt = {}
        for k, v in checkpoint["state_dict"].items():
            k = k.split('.')
            if k[0] != 'feature_frame_generate' and k[0] != 'criterion':
                k.pop(1)
            if len(k) >= 2 and k[0] == 'Residual' and k[2] == 'feature_adaptor':
                if k[3] in ['1', '2']:
                    continue
                elif k[3] == '0':
                    ckpt['.'.join(k)] = v
                    k[3] = '1'
                    ckpt['.'.join(k)] = v
                    k[3] = '2'
                    ckpt['.'.join(k)] = v
                    continue
            if k[0] == 'feature_frame_generate':
                continue
            
            ckpt['.'.join(k)] = v   

        model.load_state_dict(ckpt, strict=True)
        
    elif args.restore == 'CTM':
        checkpoint = torch.load(os.path.join('results', args.restore_exp_key, 'checkpoints', f'epoch={args.restore_exp_epoch}.pth.tar'), map_location=device)
        current_epoch = 1

        ckpt = {}
        for k, v in checkpoint["state_dict"].items():
            k = k.split('.')
            if args.ssim and k[0] == 'criterion':
                ckpt['.'.join(k)] = v
                continue
            if k[0] != 'feature_frame_generate':
                k.pop(1)
            if k[0] == 'Residual' and k[2].find('mc_generate') != -1:
                continue
            
            ckpt['.'.join(k)] = v

        model.load_state_dict(ckpt, strict=False)

    elif args.restore == 'context':
        checkpoint = torch.load(os.path.join('results', args.restore_exp_key, 'checkpoints', f'epoch={args.restore_exp_epoch}.pth.tar'), map_location=device)
        current_epoch = 1

        ckpt = {}
        for k, v in checkpoint["state_dict"].items():
            k = k.split('.')
            if not args.Pretrain_frame_feature and k[0] == 'feature_frame_generate':
                continue
            if k[0] != 'feature_frame_generate':
                k.pop(1)
            if k[0] == 'criterion':
                continue
            if k[2] == 'frame_feature_generator' and k[4].find('frame_generate') != -1:
                continue
            if k[0] == 'Residual' and k[2] == 'gaussian_conditional':
                k[2] = 'context_model.gaussian_conditional'

            ckpt['.'.join(k)] = v

        model.load_state_dict(ckpt, strict=False)

    else: 
        current_epoch = 1

    if args.gpus >= 1 and torch.cuda.device_count() >= 1:
        model.parallel(device_ids=gpu_ids)
        
    trainer = Trainer(args, model, train_cfg, current_epoch, ckpt_dir, device, epoch_ratio=args.epoch_ratio)
    
    if args.test:
        trainer.test()
    else:
        if not args.debug:
            try:
                trainer.fit()
                experiment.send_notification(
                    f"Experiment {args.experiment_name}-{args.quality_level} ({experiment.get_key()})",
                    "completed successfully"
                )
            except KeyboardInterrupt:
                torch.save({
                    "epoch": trainer.current_epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": model.optimizer.state_dict(),
                }, 
                f'{ckpt_dir}/last.pth.tar')
                
                experiment.send_notification(
                    f"Experiment {args.experiment_name}-{args.quality_level} ({experiment.get_key()})",
                    "aborted"
                )
            except Exception as e:
                print(e)
                torch.save({
                    "epoch": trainer.current_epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": model.optimizer.state_dict(),
                }, 
                f'{ckpt_dir}/last.pth.tar')

                experiment.send_notification(
                    f"Experiment {args.experiment_name}-{args.quality_level} ({experiment.get_key()})",
                    "failed"
                )
        else:
            trainer.fit()


if __name__ == "__main__":
    # log_level = "DEBUG"
    log_level = "INFO"
    log_format = "<green>{time:YYMMDD HH:mm:ss}</green> | <level>{level: <5}</level> | <yellow>{file}:{line:<3d}</yellow> | <b>{message}</b>"
    logger.add(sys.stdout, level=log_level, format=log_format, colorize=True, backtrace=True, diagnose=True)
    logger.remove(0)
    
    main(sys.argv[1:])