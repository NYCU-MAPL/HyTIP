import torch

def uniform_noise(input):
    """U(-0.5, 0.5)"""
    return torch.empty_like(input).uniform_(-0.5, 0.5)

def quant(input, mode="round", mean=None):
    if mode == "noise":
        return input + uniform_noise(input)
    else:   # dequantize, STE
        if mean is not None:
            input = input - mean

        with torch.no_grad():
            diff = input.round() - input

    return input + diff.clone().detach()

def scale_quant(input, scale=2**8):
    return quant(input * scale) / scale

def noise_quant(input):
    return quant(input, mode='noise')

def random_quant(input, m=noise_quant, mean=None, p=0.5):
    """use `m` method random quantize input with  probability `p`, others use round"""
    idxs = torch.rand_like(input).lt(p).bool()
    round_idx = torch.logical_not(idxs)
    output = torch.empty_like(input)

    if mean is not None:
        output.masked_scatter_(idxs, m(input.masked_select(idxs)))
        mean = mean.masked_select(round_idx)
        output.masked_scatter_(round_idx, quant(input.masked_select(
            round_idx), mode='round', mean=mean) + mean)
    else:
        output.masked_scatter_(idxs, m(input.masked_select(idxs)))
        output.masked_scatter_(round_idx, quant(input.masked_select(
            round_idx), mode='round'))

    return output