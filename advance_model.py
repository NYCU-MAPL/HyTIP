import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models.hub import Intra_NoAR
from compressai.zoo.image import _load_model, model_architectures
from flownets import SPyNet
from util.alignment import Alignment


class Iframe_Coder(nn.Module):
    def __init__(self, model_name="Intra", quality_level=6, ms_ssim=False, q_in_ckpt=False):
        super().__init__()
        assert model_name in model_architectures or model_name in ['Intra'], f'{model_name} is an invalid i-frame coder.'

        self.model_name = model_name

        if model_name in model_architectures:
            self.net = _load_model(model_name, "mse" if not ms_ssim else "ms-ssim", quality_level, pretrained=True)
        
        elif model_name == 'Intra':
            if q_in_ckpt:
                assert quality_level in [5, 4, 3, 2], f"Intra can't support quality level {quality_level}."
            else:
                assert quality_level in range(64), f"Intra can't support quality level {quality_level}."
            self.net = Intra_NoAR()
            
            if ms_ssim:
                self.net.load_state_dict(torch.load('./models/cvpr2023_image_ssim.pth.tar', map_location='cuda'), strict=True)
            elif not ms_ssim:
                self.net.load_state_dict(torch.load('./models/cvpr2023_image_psnr.pth.tar', map_location='cuda'), strict=True)

        self.align = Alignment(64)
        if model_name == 'Intra':
            self.q_index = quality_level - 2
        else:
            self.q_index = None
        
        self.q_in_ckpt = q_in_ckpt

    def interpolate(self):
        if self.model_name == 'Intra':
            self.net.interpolate()

    def forward(self, coding_frame, q_index=None):
        coding_frame = self.align.align(coding_frame)

        if self.model_name == 'Intra':
            I_info = self.net(coding_frame, self.q_in_ckpt, q_index=self.q_index if q_index is None else q_index)
        else:
            I_info = self.net(coding_frame)

        rec_frame = I_info['x_hat']
        rec_frame = self.align.resume(rec_frame)
    
        return rec_frame, (I_info['likelihoods']['y'], I_info['likelihoods']['z'])
    
    def compress(self, coding_frame, q_index=None):
        shape = coding_frame.size()
        coding_frame = self.align.align(coding_frame)

        if self.model_name == 'Intra':
            ret = self.net.compress(coding_frame, self.q_in_ckpt, self.q_index if q_index is None else q_index)
            return ret['x_hat'], ret['bit_stream'], [shape]
        else:
            ret = self.net.compress(coding_frame)
            return ret['strings'], [shape, ret['shape']]

    def decompress(self, strings, shape, q_index=None, sps=None):
        if self.model_name == 'Intra':
            ret = self.net.decompress(strings, shape[-2], shape[-1], self.q_in_ckpt, self.q_index if q_index is None else q_index)
            rec_frame = self.align.resume(ret['x_hat'], shape)

            return rec_frame
        else: 
            ret = self.net.decompress(strings, shape[1:])
            rec_frame = self.align.resume(ret['x_hat'], shape[0])

            return rec_frame
        
    def update(self):
        self.net.update(force=True)

class MENet(nn.Module):
    def __init__(self, mode='SPy'):
        super().__init__()

        if mode == 'SPy':
            self.net = SPyNet(trainable=False)
        else:
            raise ValueError("Invalid ME mode: {}".format(mode))

        self.align = Alignment(16)

    def forward(self, ref_frame, current_frame):
        ref_frame = self.align.align(ref_frame)
        current_frame = self.align.align(current_frame)

        flow = self.net(ref_frame, current_frame)

        flow = self.align.resume(flow)

        return flow
    
##################### For P-frame Coder #####################
class InterCoder(nn.Module):
    def __init__(self, res_coder):
        super().__init__()
        self.net = res_coder

    def update(self):
        self.net.update(force=True)

    def forward(self, coding_frame, aux_buf=None):
        rec_frame, likelihood_r, data = self.net(coding_frame, aux_buf=aux_buf)
            
        return rec_frame, likelihood_r, data
    
    def set_noise_level(self, noise):
        self.net.set_noise_level(noise)

    def motion_compensation(self, dpb, mv, fa_idx, Pretrain=False):
        output = self.net.motion_compensation(dpb, mv, fa_idx, Pretrain)
        return output
    
    def get_qp_num(self):
        return self.net.get_qp_num()
    
    def compress(self, coding_frame, aux_buf=None):

        rec_frame, data = self.net.compress(coding_frame, aux_buf=aux_buf)

        return rec_frame, data
    
    def decompress(self, strings, aux_buf=None):

        rec_frame, data = self.net.decompress(strings, aux_buf=aux_buf)
        
        return rec_frame, data

class MotionCoder(nn.Module):
    def __init__(self, mo_coder):
        super().__init__()
        self.net = mo_coder
        
    def update(self):
        self.net.update(force=True)

    def forward(self, coding_flow, aux_buf=None):
        rec_frame, likelihood_m, data = self.net(coding_flow, aux_buf=aux_buf)
        
        return rec_frame, likelihood_m, data
    
    def set_noise_level(self, noise):
        self.net.set_noise_level(noise)

    def compress(self, coding_flow, aux_buf=None):

        rec_frame, data = self.net.compress(coding_flow, aux_buf=aux_buf)

        return rec_frame, data
    
    def decompress(self, strings, aux_buf=None):

        rec_frame, data = self.net.decompress(strings, aux_buf=aux_buf)
        
        return rec_frame, data
    