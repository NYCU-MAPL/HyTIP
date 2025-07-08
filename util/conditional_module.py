from collections import OrderedDict
from matplotlib.pyplot import sca

import numpy as np
from numpy.lib.arraysetops import isin
import torch
import torch.nn.functional as F
from torch import device, nn

def gen_discrete_condition(lmdas, batch_size, shuffle=False, device='cpu'):
    if not isinstance(lmdas, list) and not isinstance(lmdas, tuple):
        lmdas = [lmdas]
    lmda_map = dict(zip(lmdas, torch.eye(len(lmdas)).unbind(0)))
    lmdas = lmdas * int(np.ceil(batch_size/len(lmdas)))
    if shuffle:
        np.random.shuffle(lmdas)
    lmdas = lmdas[:batch_size]
    conds = []
    for lmda in lmdas:
        conds.append(lmda_map[lmda])
    
    return torch.stack(conds).to(device=device), torch.Tensor(lmdas).view(-1, 1).to(device=device)

def gen_random_condition(lmdas, batch_size, shuffle=False, device='cpu'):
    if not isinstance(lmdas, list) and not isinstance(lmdas, tuple):
        lmdas = [lmdas]
    rands = np.exp(np.random.uniform(np.log(np.min(lmdas)), np.log(
        np.max(lmdas)), batch_size-len(lmdas)))
    lmdas = np.concatenate([np.array(lmdas), rands])
    if shuffle:
        np.random.shuffle(lmdas)
    else:
        lmdas = np.sort(lmdas)
    return torch.Tensor(lmdas[:batch_size]).view(-1, 1).to(device=device)

def get_out_channels(module: nn.Module):
    if hasattr(module, 'out_channels'):
        return module.out_channels
    elif hasattr(module, 'out_features'):
        return module.out_features
    elif hasattr(module, 'num_features'):
        return module.num_features
    elif hasattr(module, 'hidden_size'):
        return module.hidden_size
    raise AttributeError(
        str(module)+" has no avaiable output channels attribute")

class ConditionalLayer(nn.Module):

    def __init__(self, module: nn.Module, out_channels=None, discrete=False, conditions: int = 1, ver=2):
        super(ConditionalLayer, self).__init__()
        self.m = module
        self.discrete = discrete
        assert conditions >= 0, conditions
        self.condition_size = conditions
        self.ver = ver
        if conditions:
            if out_channels is None:
                out_channels = get_out_channels(module)
            self.out_channels = out_channels

            if self.ver == 1:
                self.weight = nn.Parameter(
                    torch.Tensor(conditions, out_channels*2))
                nn.init.kaiming_normal_(self.weight)
            elif self.ver == 2:
                self.affine = nn.Sequential(
                    nn.Linear(conditions, 16),
                    nn.Sigmoid(),
                    nn.Linear(16, out_channels * 2, bias=False)
                )
            elif self.ver == 3:
                self.affine = nn.Sequential(
                    nn.Linear(conditions, 16),
                    nn.ReLU(),
                    nn.Linear(16, out_channels*2, bias=False)
                )
            else:
                self.affine = nn.Sequential(
                    nn.Linear(conditions, 64),
                    nn.Sigmoid(),
                    nn.Linear(64, out_channels * 2, bias=False)
                )

    def extra_repr(self):
        if self.ver == 1:
            s = '(condition): '
            if self.condition_size:
                s += 'Condition({condition_size}, {out_channels})'
            else:
                s += 'skip'
            return s.format(**self.__dict__)
        else:
            return ""

    def _set_condition(self, condition):
        self.condition = condition

    def forward(self, *input, condition=None):
        output = self.m(*input)

        if self.condition_size:
            if condition is None:
                condition = self.condition

            if not isinstance(condition, tuple):
                if condition.device != output.device:
                    condition = condition.to(output.device)
                
                if self.ver == 1:
                    condition = condition.mm(self.weight)
                else:
                    condition = self.affine(condition)
                    if self.ver == 3:
                        condition = torch.exp(condition)
                scale, bias = condition.view(condition.size(0), -1, *(1,)*(output.dim()-2)).chunk(2, dim=1)
                self.condition = (scale, bias)
            else:
                scale, bias = condition

            output = output * F.softplus(scale) + bias

        return output.contiguous()

def conditional_warping(m: nn.Module, exclude_name="", types=(nn.modules.conv._ConvNd), **kwargs):
    def dfs(sub_m: nn.Module, prefix=""):
        for n, chd_m in sub_m.named_children():
            if exclude_name != "" and exclude_name in n:
                continue

            if dfs(chd_m, prefix+"."+n if prefix else n):
                setattr(sub_m, n, ConditionalLayer(chd_m, **kwargs))
        else:
            if isinstance(sub_m, types) and sub_m.in_channels > 14 and sub_m.out_channels > 14:
                return True
            else:
                pass
            
        return False

    dfs(m)

def conditional_warping_forI(m: nn.Module, types=(nn.modules.conv._ConvNd), **kwargs):
    def dfs(sub_m: nn.Module, prefix=""):
        for n, chd_m in sub_m.named_children():
            if dfs(chd_m, prefix+"."+n if prefix else n):
                setattr(sub_m, n, ConditionalLayer(chd_m, **kwargs))
        else:
            if isinstance(sub_m, types):
                return True
            else:
                pass
            
        return False

    dfs(m)
   
def set_condition(model, condition):
    for m in model.modules():
        if isinstance(m, ConditionalLayer):
            m._set_condition(condition)
