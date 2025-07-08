import numbers

import numpy as np
import torch

from util.math import lower_bound


def estimate_bpp(likelihood, num_pixels=None, input=None, likelihood_bound: float = 1e-9, mask=None):
    """estimate bits-per-pixel

    Args:
        likelihood_bound: Float. If positive, the returned likelihood values are
            ensured to be greater than or equal to this value. This prevents very
            large gradients with a typical entropy loss (defaults to 1e-9).
    """
    if num_pixels is None:
        assert torch.is_tensor(input) and input.dim() > 2
        num_pixels = np.prod(input.size()[-2:])
    assert isinstance(num_pixels, numbers.Number), type(num_pixels)
    if torch.is_tensor(likelihood):
        likelihood = [likelihood]
    lll = 0
    for ll in likelihood:
        log_ll = lower_bound(ll, likelihood_bound).log()  
        if mask is not None:
            log_ll = log_ll * mask
            
        lll = lll + log_ll.flatten(1).sum(1)

    return lll / (-np.log(2.) * num_pixels)
