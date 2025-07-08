import torch
import numpy as np

# use in dataloader.py
def read_yuv_args(bit_depth, y_size, uv_size):
    y_size = y_size
    uv_size = uv_size
    dtype = np.uint8
    max_val = 255
    if bit_depth > 8 and bit_depth <= 16:
        y_size = y_size * 2
        uv_size = uv_size * 2
        dtype = np.uint16
        max_val = (1 << bit_depth) - 1

    return y_size, uv_size, dtype, max_val

# use in training script
def write_yuv_args(bit_depth, device):
    dtype = np.uint8
    max_val = torch.as_tensor(255, device=device)
    if bit_depth > 8 and bit_depth <= 16:
        dtype = np.uint16
        max_val = torch.as_tensor((1 << bit_depth) - 1, device=device)

    return dtype, max_val