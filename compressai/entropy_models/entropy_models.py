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

import math
import warnings

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from .quantization import random_quant, noise_quant, quant
from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from compressai.ops import LowerBound

from loguru import logger

class _EntropyCoder:
    """Proxy class to an actual entropy coder class."""

    def __init__(self, method):
        if not isinstance(method, str):
            raise ValueError(f'Invalid method type "{type(method)}"')

        from compressai import available_entropy_coders

        if method not in available_entropy_coders():
            methods = ", ".join(available_entropy_coders())
            raise ValueError(
                f'Unknown entropy coder "{method}"' f" (available: {methods})"
            )

        if method == "ans":
            from compressai import ans

            encoder = ans.RansEncoder()
            decoder = ans.RansDecoder()
        elif method == "rangecoder":
            import range_coder

            encoder = range_coder.RangeEncoder()
            decoder = range_coder.RangeDecoder()

        self.name = method
        self._encoder = encoder
        self._decoder = decoder

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)

def default_entropy_coder():
    from compressai import get_entropy_coder

    return get_entropy_coder()

def pmf_to_quantized_cdf(pmf: Tensor, precision: int = 16) -> Tensor:
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf

def _forward(self, *args: Any) -> Any:
    raise NotImplementedError()

class EntropyModel(nn.Module):
    r"""Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(
        self,
        quant_mode: str = 'noise',
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
    ):
        super().__init__()
        self.quant_mode = quant_mode

        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        # to be filled on update()
        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["entropy_coder"] = self.entropy_coder.name
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.entropy_coder = _EntropyCoder(self.__dict__.pop("entropy_coder"))

    @property
    def offset(self):
        return self._offset

    @property
    def quantized_cdf(self):
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length

    # See: https://github.com/python/mypy/issues/8795
    forward: Callable[..., Any] = _forward

    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        if mode not in ("noise", "dequantize", "symbols", "RUN"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            assert self.training
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs

        if mode == 'RUN':
            assert self.training   
            return random_quant(inputs, noise_quant, means)
        
        if self.training:
            outputs = quant(inputs, "round", means)
        else:
            outputs = inputs.clone()
            if means is not None:
                outputs -= means

            outputs = torch.round(outputs)

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    def _quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        warnings.warn("_quantize is deprecated. Use quantize instead.")
        return self.quantize(inputs, mode, means)

    @staticmethod
    def dequantize(
        inputs: Tensor, means: Optional[Tensor] = None, dtype: torch.dtype = torch.float64
    ) -> Tensor:
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.type(dtype)
        return outputs

    @classmethod
    def _dequantize(cls, inputs: Tensor, means: Optional[Tensor] = None) -> Tensor:
        warnings.warn("_dequantize. Use dequantize instead.")
        return cls.dequantize(inputs, means)

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros(
            (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
        )
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.size()}")

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")

    def compress(self, inputs, indexes, means=None, mask=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
            mask (torch.Tensor, optional): used for skip mode
        """
        symbols = self.quantize(inputs, "symbols", means)
        
        if len(inputs.size()) < 2:
            raise ValueError(
                "Invalid `inputs` size. Expected a tensor with at least 2 dimensions."
            )

        if inputs.size() != indexes.size():
            raise ValueError("`inputs` and `indexes` should have the same size.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if mask is not None:
            symbols = torch.masked_select(symbols, mask==1)

        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            strings.append(rv)
        return strings

    def decompress(
        self,
        strings: str,
        indexes: torch.IntTensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor = None,
        mask: torch.Tensor = None
    ):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
            mask (torch.Tensor, optional): used for skip mode
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )

            outputs[i] = torch.tensor(
                values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())

        outputs = self.dequantize(outputs, means, dtype)
        
        return outputs

class EntropyBottleneck(EntropyModel):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://tensorflow.github.io/compression/docs/entropy_bottleneck.html>`__
    for an introduction.
    """

    _offset: Tensor

    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        init_scale: float = 10,
        filters: Tuple[int, ...] = (3, 3, 3, 3),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))

    def _get_medians(self) -> Tensor:
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force: bool = False) -> bool:
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:
            return False

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = float(0.5)

        lower = self._logits_cumulative(samples - half, stop_gradient=True)
        upper = self._logits_cumulative(samples + half, stop_gradient=True)
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self) -> Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    @torch.jit.unused
    def _likelihood(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half
        lower = self._logits_cumulative(v0, stop_gradient=False)
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        return likelihood

    def forward(
        self, x: Tensor, training: Optional[bool] = None
    ) -> Tuple[Tensor, Tensor]:

        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            # Compute inverse permutation
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            raise NotImplementedError()
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            # perm = (1, 2, 3, 0)
            # inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        if self.quant_mode == 'estUN_outR' and training:
            output_UN = self.quantize(values, mode='noise', means=self._get_medians())
            outputs = self.quantize(values, mode='dequantize', means=self._get_medians())
        else:
            outputs = self.quantize(values, self.quant_mode if training else "dequantize", self._get_medians())

        if not torch.jit.is_scripting():
            likelihood = self._likelihood(output_UN) if self.quant_mode == 'estUN_outR' and training else self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
            # TorchScript not yet supported
            # likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    def compress(self, x):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)

class GaussianConditional(EntropyModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`__
    for more information.
    """

    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and (
            scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)
        ):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBound(scale_bound)

        self.register_buffer(
            "scale_table",
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return False
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        self.update()
        return True

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]
        )
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower

        return likelihood

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Optional[Tensor] = None,
        training: Optional[bool] = None,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training

        if self.quant_mode == 'estUN_outR' and self.training:
            outputs = {'entropy_model': self.quantize(inputs, mode='noise', means=means),
                       'decoder'      : self.quantize(inputs, mode='dequantize', means=means)}

        else:
            output = self.quantize(inputs, self.quant_mode if training else "dequantize", means)
            outputs = {'entropy_model': output,
                       'decoder'      : output}

        if mask is not None:
            outputs['entropy_model'] = (1 - mask) * means + mask * outputs['entropy_model']
            outputs['decoder'] = (1 - mask) * means + mask * outputs['decoder']
        
        likelihood = self._likelihood(outputs['entropy_model'], scales, means)
            
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs['decoder'], likelihood

    def build_indexes(self, scales: Tensor) -> Tensor:
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes

############################## For Intra ##############################
def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_left = 0
    padding_right = new_w - width - padding_left
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom

class EntropyCoder():
    def __init__(self, ec_thread=False, stream_part=1):
        super().__init__()

        from compressai.MLCodec_rans_Intra import MLCodec_RansEncoder_Intra, MLCodec_RansDecoder_Intra
        self.encoder = MLCodec_RansEncoder_Intra(ec_thread, stream_part)
        self.decoder = MLCodec_RansDecoder_Intra(stream_part)

    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        from compressai.MLCodec_CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
        cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
        cdf = torch.IntTensor(cdf)
        return cdf

    @staticmethod
    def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        entropy_coder_precision = 16
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = EntropyCoder.pmf_to_quantized_cdf(prob, entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def reset(self):
        self.encoder.reset()

    def encode_with_indexes(self, symbols, indexes, cdf, cdf_length, offset):
        self.encoder.encode_with_indexes(symbols.clamp(-30000, 30000).to(torch.int16).cpu().numpy(),
                                         indexes.to(torch.int16).cpu().numpy(),
                                         cdf, cdf_length, offset)

    def flush(self):
        self.encoder.flush()

    def get_encoded_stream(self):
        return self.encoder.get_encoded_stream().tobytes()

    def set_stream(self, stream):
        self.decoder.set_stream((np.frombuffer(stream, dtype=np.uint8)))

    def decode_stream(self, indexes, cdf, cdf_length, offset):
        rv = self.decoder.decode_stream(indexes.to(torch.int16).cpu().numpy(),
                                        cdf, cdf_length, offset)
        rv = torch.Tensor(rv)
        return rv
    
class EntropyCoder_Inter():
    def __init__(self, ec_thread=False, stream_part=1):
        super().__init__()

        from compressai.MLCodec_rans_Inter import MLCodec_RansEncoder_Inter, MLCodec_RansDecoder_Inter  # pylint: disable=E0401
        self.encoder = MLCodec_RansEncoder_Inter(ec_thread, stream_part)
        self.decoder = MLCodec_RansDecoder_Inter(stream_part)

    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        from compressai.MLCodec_CXX import pmf_to_quantized_cdf as _pmf_to_cdf  # pylint: disable=E0401
        cdf = _pmf_to_cdf(pmf.tolist(), precision)
        cdf = torch.IntTensor(cdf)
        return cdf

    @staticmethod
    def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        entropy_coder_precision = 16
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = EntropyCoder_Inter.pmf_to_quantized_cdf(prob, entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def reset(self):
        self.encoder.reset()

    def add_cdf(self, cdf, cdf_length, offset):
        enc_cdf_idx = self.encoder.add_cdf(cdf, cdf_length, offset)
        dec_cdf_idx = self.decoder.add_cdf(cdf, cdf_length, offset)
        assert enc_cdf_idx == dec_cdf_idx
        return enc_cdf_idx

    def encode_with_indexes(self, symbols, indexes, cdf_group_index):
        self.encoder.encode_with_indexes(symbols.clamp(-30000, 30000).to(torch.int16).cpu().numpy(),
                                         indexes.to(torch.int16).cpu().numpy(),
                                         cdf_group_index)

    def encode_with_indexes_np(self, symbols, indexes, cdf_group_index):
        self.encoder.encode_with_indexes(symbols.clip(-30000, 30000).astype(np.int16).reshape(-1),
                                         indexes.astype(np.int16).reshape(-1),
                                         cdf_group_index)

    def flush(self):
        self.encoder.flush()

    def get_encoded_stream(self):
        return self.encoder.get_encoded_stream().tobytes()

    def set_stream(self, stream):
        self.decoder.set_stream((np.frombuffer(stream, dtype=np.uint8)))

    def decode_stream(self, indexes, cdf_group_index):
        rv = self.decoder.decode_stream(indexes.to(torch.int16).cpu().numpy(),
                                        cdf_group_index)
        rv = torch.Tensor(rv)
        return rv

    def decode_stream_np(self, indexes, cdf_group_index):
        rv = self.decoder.decode_stream(indexes.astype(np.int16).reshape(-1),
                                        cdf_group_index)
        return rv
    
class Bitparm(nn.Module):
    def __init__(self, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(
            torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(
            torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(
                torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        x = x * F.softplus(self.h) + self.b
        if self.final:
            return x

        return x + torch.tanh(x) * torch.tanh(self.a)
    
class Bitparm_Inter(nn.Module):
    def __init__(self, qp_num, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(
            torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(
            torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(
                torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        else:
            self.a = None

    def forward(self, x, index):
        h = torch.index_select(self.h, 0, index)
        b = torch.index_select(self.b, 0, index)
        x = x * F.softplus(h) + b
        if self.final:
            return x

        a = torch.index_select(self.a, 0, index)
        return x + torch.tanh(x) * torch.tanh(a)

class AEHelper():
    def __init__(self):
        super().__init__()
        self.entropy_coder = None
        self.cdf_group_index = None
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    def set_entropy_coder(self, coder):
        self.entropy_coder = coder

    def set_cdf_info(self, quantized_cdf, cdf_length, offset):
        self._quantized_cdf = quantized_cdf.cpu().numpy()
        self._cdf_length = cdf_length.reshape(-1).int().cpu().numpy()
        self._offset = offset.reshape(-1).int().cpu().numpy()

    def get_cdf_info(self):
        return self._quantized_cdf, \
            self._cdf_length, \
            self._offset

class BitEstimator(AEHelper, nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)
        self.channel = channel

    def forward(self, x):
        return self.get_cdf(x)

    def get_logits_cdf(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

    def get_cdf(self, x):
        return torch.sigmoid(self.get_logits_cdf(x))
    
    def update(self, force=False, entropy_coder=None):
        if entropy_coder is not None:
            self.entropy_coder = entropy_coder

        if not force and self._offset is not None:
            return

        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros((self.channel), device=device)

            minima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) - i
                samples = samples[None, :, None, None]
                probs = self.forward(samples)
                probs = torch.squeeze(probs)
                minima = torch.where(probs < torch.zeros_like(medians) + 0.0001,
                                     torch.zeros_like(medians) + i, minima)

            maxima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) + i
                samples = samples[None, :, None, None]
                probs = self.forward(samples)
                probs = torch.squeeze(probs)
                maxima = torch.where(probs > torch.zeros_like(medians) + 0.9999,
                                     torch.zeros_like(medians) + i, maxima)

            minima = minima.int()
            maxima = maxima.int()

            offset = -minima

            pmf_start = medians - minima
            pmf_length = maxima + minima + 1

            max_length = pmf_length.max()
            device = pmf_start.device
            samples = torch.arange(max_length, device=device)

            samples = samples[None, :] + pmf_start[:, None, None]

            half = float(0.5)

            lower = self.forward(samples - half).squeeze(0)
            upper = self.forward(samples + half).squeeze(0)
            pmf = upper - lower

            pmf = pmf[:, 0, :]
            tail_mass = lower[:, 0, :1] + (1.0 - upper[:, 0, -1:])

            quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.set_cdf_info(quantized_cdf, cdf_length, offset)

    @staticmethod
    def build_indexes(size):
        N, C, H, W = size
        indexes = torch.arange(C, dtype=torch.int).view(1, -1, 1, 1)
        return indexes.repeat(N, 1, H, W)

    @staticmethod
    def build_indexes_np(size):
        return BitEstimator.build_indexes(size).cpu().numpy()

    def encode(self, x):
        indexes = self.build_indexes(x.size())
        return self.entropy_coder.encode_with_indexes(x.reshape(-1), indexes.reshape(-1),
                                                      *self.get_cdf_info())

    def decode_stream(self, size, dtype, device):
        output_size = (1, self.channel, size[0], size[1])
        indexes = self.build_indexes(output_size)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1), *self.get_cdf_info())
        val = val.reshape(indexes.shape)
        return val.to(dtype).to(device)
    
class BitEstimator_Inter(AEHelper, nn.Module):
    def __init__(self, qp_num, channel):
        super().__init__()
        self.f1 = Bitparm_Inter(qp_num, channel)
        self.f2 = Bitparm_Inter(qp_num, channel)
        self.f3 = Bitparm_Inter(qp_num, channel)
        self.f4 = Bitparm_Inter(qp_num, channel, True)
        self.qp_num = qp_num
        self.channel = channel

    def forward(self, x, index):
        return self.get_cdf(x, index)

    def get_logits_cdf(self, x, index):
        x = self.f1(x, index)
        x = self.f2(x, index)
        x = self.f3(x, index)
        x = self.f4(x, index)
        return x

    def get_cdf(self, x, index):
        return torch.sigmoid(self.get_logits_cdf(x, index))

    def update(self, force=False, entropy_coder=None):
        assert entropy_coder is not None
        self.entropy_coder = entropy_coder

        if not force and self._offset is not None:
            return

        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros((self.qp_num, self.channel, 1, 1), device=device)
            index = torch.arange(self.qp_num, device=device, dtype=torch.int32)

            minima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) - i
                probs = self.forward(samples, index)
                minima = torch.where(probs < torch.zeros_like(medians) + 0.0001,
                                     torch.zeros_like(medians) + i, minima)

            maxima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) + i
                probs = self.forward(samples, index)
                maxima = torch.where(probs > torch.zeros_like(medians) + 0.9999,
                                     torch.zeros_like(medians) + i, maxima)

            minima = minima.int()
            maxima = maxima.int()

            offset = -minima

            pmf_start = medians - minima
            pmf_length = maxima + minima + 1

            max_length = pmf_length.max()
            device = pmf_start.device
            samples = torch.arange(max_length, device=device)

            samples = samples[None, None, None, :] + pmf_start

            half = float(0.5)

            lower = self.forward(samples - half, index)
            upper = self.forward(samples + half, index)
            pmf = upper - lower

            pmf = pmf[:, :, 0, :]
            tail_mass = lower[:, :, 0, :1] + (1.0 - upper[:, :, 0, -1:])

            pmf = pmf.reshape([-1, max_length])
            tail_mass = tail_mass.reshape([-1, 1])
            pmf_length = pmf_length.reshape([-1])
            offset = offset.reshape([-1])
            quantized_cdf = EntropyCoder_Inter.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.set_cdf_info(quantized_cdf, cdf_length, offset)
            self.cdf_group_index = self.entropy_coder.add_cdf(*self.get_cdf_info())

    def build_indexes(self, size, qp):
        B, C, H, W = size
        indexes = torch.arange(C, dtype=torch.int).view(1, -1, 1, 1) + qp * self.channel
        return indexes.repeat(B, 1, H, W)

    def build_indexes_np(self, size, qp):
        return self.build_indexes(size, qp).cpu().numpy()

    def encode(self, x, qp):
        indexes = self.build_indexes(x.size(), qp)
        return self.entropy_coder.encode_with_indexes(x.reshape(-1), indexes.reshape(-1),
                                                      self.cdf_group_index)

    def decode_stream(self, size, dtype, device, qp):
        output_size = (1, self.channel, size[0], size[1])
        indexes = self.build_indexes(output_size, qp)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1), self.cdf_group_index)
        val = val.reshape(indexes.shape)
        return val.to(dtype).to(device)
    
class GaussianEncoder(AEHelper):
    def __init__(self, distribution='laplace'):
        super().__init__()
        assert distribution in ['laplace', 'gaussian']
        self.distribution = distribution
        if distribution == 'laplace':
            self.cdf_distribution = torch.distributions.laplace.Laplace
            self.scale_min = 0.01
            self.scale_max = 64.0
            self.scale_level = 256
        elif distribution == 'gaussian':
            self.cdf_distribution = torch.distributions.normal.Normal
            self.scale_min = 0.11
            self.scale_max = 64.0
            self.scale_level = 256
        self.scale_table = self.get_scale_table(self.scale_min, self.scale_max, self.scale_level)

        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_scale_step = (self.log_scale_max - self.log_scale_min) / (self.scale_level - 1)

    @staticmethod
    def get_scale_table(min_val, max_val, levels):
        return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))
    
    def update(self, force=False, entropy_coder=None):
        if entropy_coder is not None:
            self.entropy_coder = entropy_coder

        if not force and self._offset is not None:
            return

        pmf_center = torch.zeros_like(self.scale_table) + 50
        scales = torch.zeros_like(pmf_center) + self.scale_table
        mu = torch.zeros_like(scales)
        cdf_distribution = self.cdf_distribution(mu, scales)
        for i in range(50, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = cdf_distribution.cdf(samples)
            probs = torch.squeeze(probs)
            pmf_center = torch.where(probs > torch.zeros_like(pmf_center) + 0.9999,
                                     torch.zeros_like(pmf_center) + i, pmf_center)

        pmf_center = pmf_center.int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.arange(max_length, device=device) - pmf_center[:, None]
        samples = samples.float()

        scales = torch.zeros_like(samples) + self.scale_table[:, None]
        mu = torch.zeros_like(scales)
        cdf_distribution = self.cdf_distribution(mu, scales)

        upper = cdf_distribution.cdf(samples + 0.5)
        lower = cdf_distribution.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)

        self.set_cdf_info(quantized_cdf, pmf_length+2, -pmf_center)

    def build_indexes(self, scales):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-5)
        indexes = (torch.log(scales) - self.log_scale_min) / self.log_scale_step
        indexes = indexes.clamp_(0, self.scale_level - 1)
        return indexes.int()

    def encode(self, x, scales):
        indexes = self.build_indexes(scales)
        return self.entropy_coder.encode_with_indexes(x.reshape(-1), indexes.reshape(-1),
                                                      *self.get_cdf_info())

    def decode_stream(self, scales, dtype, device):
        indexes = self.build_indexes(scales)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1),
                                               *self.get_cdf_info())
        val = val.reshape(scales.shape)
        return val.to(device).to(dtype)
    
class GaussianEncoder_Inter(GaussianEncoder):
    def __init__(self, distribution='laplace'):
        super().__init__(distribution=distribution)

    def update(self, force=False, entropy_coder=None):
        assert entropy_coder is not None
        self.entropy_coder = entropy_coder

        if not force and self._offset is not None:
            return

        pmf_center = torch.zeros_like(self.scale_table) + 50
        scales = torch.zeros_like(pmf_center) + self.scale_table
        mu = torch.zeros_like(scales)
        cdf_distribution = self.cdf_distribution(mu, scales)
        for i in range(50, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = cdf_distribution.cdf(samples)
            probs = torch.squeeze(probs)
            pmf_center = torch.where(probs > torch.zeros_like(pmf_center) + 0.9999,
                                     torch.zeros_like(pmf_center) + i, pmf_center)

        pmf_center = pmf_center.int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.arange(max_length, device=device) - pmf_center[:, None]
        samples = samples.float()

        scales = torch.zeros_like(samples) + self.scale_table[:, None]
        mu = torch.zeros_like(scales)
        cdf_distribution = self.cdf_distribution(mu, scales)

        upper = cdf_distribution.cdf(samples + 0.5)
        lower = cdf_distribution.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = EntropyCoder_Inter.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)

        self.set_cdf_info(quantized_cdf, pmf_length+2, -pmf_center)
        self.cdf_group_index = self.entropy_coder.add_cdf(*self.get_cdf_info())

    def encode(self, x, scales):
        indexes = self.build_indexes(scales)
        return self.entropy_coder.encode_with_indexes(x.reshape(-1), indexes.reshape(-1),
                                                      self.cdf_group_index)

    def decode_stream(self, scales, dtype, device):
        indexes = self.build_indexes(scales)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1),
                                               self.cdf_group_index)
        val = val.reshape(scales.shape)
        return val.to(device).to(dtype)
    
class CompressionModel(nn.Module):
    def __init__(self, y_distribution, z_channel, mv_z_channel=None,
                 ec_thread=False, stream_part=1):
        super().__init__()

        self.y_distribution = y_distribution
        self.z_channel = z_channel
        self.mv_z_channel = mv_z_channel
        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(z_channel)
        self.bit_estimator_z_mv = None
        if mv_z_channel is not None:
            self.bit_estimator_z_mv = BitEstimator(mv_z_channel)
        self.gaussian_encoder = GaussianEncoder(distribution=y_distribution)
        self.ec_thread = ec_thread
        self.stream_part = stream_part

        self.masks = {}

    def quant(self, x, mode="dequantize"):
        # from compressai.entropy_models.EntropyModel.quantize
        if self.training and mode == "noise":
            assert self.training
            half = float(0.5)
            noise = torch.empty_like(x).uniform_(-half, half)
            x = x + noise
            logger.debug(f"Adding noise to input, {self.training=}, {mode=}, {x.sum()=}, {noise.sum()=}")
            return x
        else:   # if mode == "dequantize":
            # return torch.round(x)                     # Round, for inference only
            return (torch.round(x) - x).detach() + x    # STE, for training

    def get_curr_q(self, q_scale, q_basic, q_index=None):
        q_scale = q_scale[q_index]
        return q_basic * q_scale

    @staticmethod
    def probs_to_bits(probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = torch.clamp_min(bits, 0)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs), probs

    def get_y_laplace_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        if torch.any(torch.isnan(sigma)):
            logger.error(f"{torch.any(torch.isnan(sigma))=}")
            return None, torch.tensor(float('nan'))
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs), probs

    def get_z_bits(self, z, bit_estimator):
        probs = bit_estimator.get_cdf(z + 0.5) - bit_estimator.get_cdf(z - 0.5)
        return CompressionModel.probs_to_bits(probs), probs

    def update(self, force=False):
        self.entropy_coder = EntropyCoder(self.ec_thread, self.stream_part)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)
        if self.bit_estimator_z_mv is not None:
            self.bit_estimator_z_mv.update(force=force, entropy_coder=self.entropy_coder)
            
    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(
            y,
            (padding_l, padding_r, padding_t, padding_b),
            mode="replicate",
        )
        return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)

    @staticmethod
    def get_to_y_slice_shape(height, width):
        padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 4)
        return (-padding_l, -padding_r, -padding_t, -padding_b)

    def slice_to_y(self, param, slice_shape):
        return torch.nn.functional.pad(param, slice_shape)

    @staticmethod
    def separate_prior(params):
        return params.chunk(3, 1)

    def process_with_mask(self, y, scales, means, mask):
        mask = mask.to(scales.device)
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

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

    @staticmethod
    def combine_four_parts(x_0_0, x_0_1, x_0_2, x_0_3,
                           x_1_0, x_1_1, x_1_2, x_1_3,
                           x_2_0, x_2_1, x_2_2, x_2_3,
                           x_3_0, x_3_1, x_3_2, x_3_3):
        x_0 = x_0_0 + x_0_1 + x_0_2 + x_0_3
        x_1 = x_1_0 + x_1_1 + x_1_2 + x_1_3
        x_2 = x_2_0 + x_2_1 + x_2_2 + x_2_3
        x_3 = x_3_0 + x_3_1 + x_3_2 + x_3_3
        return torch.cat((x_0, x_1, x_2, x_3), dim=1)

    def forward_four_part_prior(self, y, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        quant_step, scales, means = self.separate_prior(common_params)
        dtype = y.dtype
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)

        quant_step = torch.clamp_min(quant_step, 0.5)
        y = y / quant_step
        y_0, y_1, y_2, y_3 = y.chunk(4, 1)

        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)

        y_res_0_0, y_q_0_0, y_hat_0_0, s_hat_0_0 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_0)
        y_res_1_1, y_q_1_1, y_hat_1_1, s_hat_1_1 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_1)
        y_res_2_2, y_q_2_2, y_hat_2_2, s_hat_2_2 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_2)
        y_res_3_3, y_q_3_3, y_hat_3_3, s_hat_3_3 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_3)
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)

        y_hat_so_far = y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(8, 1)

        y_res_0_3, y_q_0_3, y_hat_0_3, s_hat_0_3 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_3)
        y_res_1_2, y_q_1_2, y_hat_1_2, s_hat_1_2 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_2)
        y_res_2_1, y_q_2_1, y_hat_2_1, s_hat_2_1 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_1)
        y_res_3_0, y_q_3_0, y_hat_3_0, s_hat_3_0 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_0)
        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)

        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(8, 1)

        y_res_0_2, y_q_0_2, y_hat_0_2, s_hat_0_2 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_2)
        y_res_1_3, y_q_1_3, y_hat_1_3, s_hat_1_3 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_3)
        y_res_2_0, y_q_2_0, y_hat_2_0, s_hat_2_0 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_0)
        y_res_3_1, y_q_3_1, y_hat_3_1, s_hat_3_1 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_1)
        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)

        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(8, 1)

        y_res_0_1, y_q_0_1, y_hat_0_1, s_hat_0_1 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_1)
        y_res_1_0, y_q_1_0, y_hat_1_0, s_hat_1_0 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_0)
        y_res_2_3, y_q_2_3, y_hat_2_3, s_hat_2_3 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_3)
        y_res_3_2, y_q_3_2, y_hat_3_2, s_hat_3_2 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_2)

        y_res = self.combine_four_parts(y_res_0_0, y_res_0_1, y_res_0_2, y_res_0_3,
                                        y_res_1_0, y_res_1_1, y_res_1_2, y_res_1_3,
                                        y_res_2_0, y_res_2_1, y_res_2_2, y_res_2_3,
                                        y_res_3_0, y_res_3_1, y_res_3_2, y_res_3_3)
        y_q = self.combine_four_parts(y_q_0_0, y_q_0_1, y_q_0_2, y_q_0_3,
                                      y_q_1_0, y_q_1_1, y_q_1_2, y_q_1_3,
                                      y_q_2_0, y_q_2_1, y_q_2_2, y_q_2_3,
                                      y_q_3_0, y_q_3_1, y_q_3_2, y_q_3_3)
        y_hat = self.combine_four_parts(y_hat_0_0, y_hat_0_1, y_hat_0_2, y_hat_0_3,
                                        y_hat_1_0, y_hat_1_1, y_hat_1_2, y_hat_1_3,
                                        y_hat_2_0, y_hat_2_1, y_hat_2_2, y_hat_2_3,
                                        y_hat_3_0, y_hat_3_1, y_hat_3_2, y_hat_3_3)
        scales_hat = self.combine_four_parts(s_hat_0_0, s_hat_0_1, s_hat_0_2, s_hat_0_3,
                                             s_hat_1_0, s_hat_1_1, s_hat_1_2, s_hat_1_3,
                                             s_hat_2_0, s_hat_2_1, s_hat_2_2, s_hat_2_3,
                                             s_hat_3_0, s_hat_3_1, s_hat_3_2, s_hat_3_3)

        y_hat = y_hat * quant_step

        if write:
            y_q_w_0 = y_q_0_0 + y_q_1_1 + y_q_2_2 + y_q_3_3
            y_q_w_1 = y_q_0_3 + y_q_1_2 + y_q_2_1 + y_q_3_0
            y_q_w_2 = y_q_0_2 + y_q_1_3 + y_q_2_0 + y_q_3_1
            y_q_w_3 = y_q_0_1 + y_q_1_0 + y_q_2_3 + y_q_3_2
            scales_w_0 = s_hat_0_0 + s_hat_1_1 + s_hat_2_2 + s_hat_3_3
            scales_w_1 = s_hat_0_3 + s_hat_1_2 + s_hat_2_1 + s_hat_3_0
            scales_w_2 = s_hat_0_2 + s_hat_1_3 + s_hat_2_0 + s_hat_3_1
            scales_w_3 = s_hat_0_1 + s_hat_1_0 + s_hat_2_3 + s_hat_3_2
            return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3,\
                scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat
        return y_res, y_q, y_hat, scales_hat
    
    def compress_four_part_prior(self, y, common_params,
                                 y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                 y_spatial_prior_adaptor_3, y_spatial_prior):
        return self.forward_four_part_prior(y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior, write=True)

    def decompress_four_part_prior(self, common_params,
                                   y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                   y_spatial_prior_adaptor_3, y_spatial_prior):
        quant_step, scales, means = self.separate_prior(common_params)
        dtype = means.dtype
        device = means.device
        _, _, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)
        quant_step = torch.clamp_min(quant_step, 0.5)

        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)

        scales_r = scales_0 * mask_0 + scales_1 * mask_1 + scales_2 * mask_2 + scales_3 * mask_3
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_0_0 = (y_q_r + means_0) * mask_0
        y_hat_1_1 = (y_q_r + means_1) * mask_1
        y_hat_2_2 = (y_q_r + means_2) * mask_2
        y_hat_3_3 = (y_q_r + means_3) * mask_3
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(8, 1)
        scales_r = scales_0 * mask_3 + scales_1 * mask_2 + scales_2 * mask_1 + scales_3 * mask_0
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_0_3 = (y_q_r + means_0) * mask_3
        y_hat_1_2 = (y_q_r + means_1) * mask_2
        y_hat_2_1 = (y_q_r + means_2) * mask_1
        y_hat_3_0 = (y_q_r + means_3) * mask_0
        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(8, 1)
        scales_r = scales_0 * mask_2 + scales_1 * mask_3 + scales_2 * mask_0 + scales_3 * mask_1
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_0_2 = (y_q_r + means_0) * mask_2
        y_hat_1_3 = (y_q_r + means_1) * mask_3
        y_hat_2_0 = (y_q_r + means_2) * mask_0
        y_hat_3_1 = (y_q_r + means_3) * mask_1
        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(8, 1)
        scales_r = scales_0 * mask_1 + scales_1 * mask_0 + scales_2 * mask_3 + scales_3 * mask_2
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_0_1 = (y_q_r + means_0) * mask_1
        y_hat_1_0 = (y_q_r + means_1) * mask_0
        y_hat_2_3 = (y_q_r + means_2) * mask_3
        y_hat_3_2 = (y_q_r + means_3) * mask_2
        y_hat_curr_step = torch.cat((y_hat_0_1, y_hat_1_0, y_hat_2_3, y_hat_3_2), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * quant_step

        return y_hat

#################################### P-frame ####################################
class CompressionModel_Inter(CompressionModel):
    def __init__(self, y_distribution, z_channel, mv_z_channel=None,
                 ec_thread=False, stream_part=1):
        super().__init__(y_distribution=y_distribution, z_channel=z_channel, mv_z_channel=mv_z_channel,
                 ec_thread=ec_thread, stream_part=stream_part)
        
        if mv_z_channel is None:
            self.bit_estimator_z = BitEstimator_Inter(64, z_channel)
            self.bit_estimator_z_mv = None
        else:
            self.bit_estimator_z = BitEstimator_Inter(1, z_channel)
            self.bit_estimator_z_mv = BitEstimator_Inter(1, mv_z_channel)
        self.gaussian_encoder = GaussianEncoder_Inter(distribution=y_distribution)
        self.ec_thread = ec_thread
        self.stream_part = stream_part

        self.masks = {}

    def quant(self, x, force_detach=False):
        # print("training", self.training)
        if self.training or force_detach:
            n = torch.round(x) - x
            n = n.clone().detach()
            return x + n
        else:
            return torch.round(x)
    
    def get_one_q_scale(self, q_scale, q_index):
        min_q = q_scale[0:1, :, :, :]
        max_q = q_scale[1:2, :, :, :]
        step = (torch.log(max_q) - torch.log(min_q)) / (self.get_qp_num() - 1)
        q = torch.exp(torch.log(min_q) + step * q_index)
        return q

    def get_curr_q(self, q_scale, q_index):
        if isinstance(q_index, list):
            q_step = [self.get_one_q_scale(q_scale, i) for i in q_index]
            q_step = torch.cat(q_step, dim=0)
        else:
            q_step = self.get_one_q_scale(q_scale, q_index)

        return q_step

    @staticmethod
    def get_index_tensor(q_index, device):
        if not isinstance(q_index, list):
            q_index = [q_index]
        return torch.tensor(q_index, dtype=torch.int32, device=device)

    @staticmethod
    def get_qp_num():
        return 64

    @staticmethod
    def probs_to_bits(probs):
        factor = -1.0 / math.log(2.0)
        bits = torch.log(probs + 1e-5) * factor
        bits = torch.clamp(bits, 0, None)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        probs = probs.to(torch.float32)
        return CompressionModel_Inter.probs_to_bits(probs), probs

    def get_y_laplace_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        probs = probs.to(torch.float32)
        return CompressionModel_Inter.probs_to_bits(probs), probs

    def get_z_bits(self, z, bit_estimator, index):
        probs = bit_estimator.get_cdf(z + 0.5, index) - bit_estimator.get_cdf(z - 0.5, index)
        probs = probs.to(torch.float32)
        return CompressionModel_Inter.probs_to_bits(probs), probs
    
    def update(self, force=False):
        self.entropy_coder = EntropyCoder_Inter(self.ec_thread, self.stream_part)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)
        if self.bit_estimator_z_mv is not None:
            self.bit_estimator_z_mv.update(force=force, entropy_coder=self.entropy_coder)

    @staticmethod
    def separate_prior(params, is_video=False):
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            quant_step = torch.clamp(quant_step, 0.5, None)
            q_enc = 1. / quant_step
            q_dec = quant_step
        else:
            q = params[:, :2, :, :]
            q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
            scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means
    
    def process_with_mask(self, y, scales, means, mask):
        mask = mask.to(scales.device)
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat, means_hat

    def get_mask(self, height, width, dtype, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.masks:
            micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
            mask_0 = micro_mask.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_1 = torch.ones_like(mask_0) - mask_0
            self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    @staticmethod
    def get_one_channel_four_parts_mask(height, width, dtype, device):
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

        return mask_0, mask_1, mask_2, mask_3

    def get_mask_four_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0, m1, m2, m3 = self.get_one_channel_four_parts_mask(height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    @staticmethod
    def combine_four_parts(x_0_0, x_0_1, x_0_2, x_0_3,
                           x_1_0, x_1_1, x_1_2, x_1_3,
                           x_2_0, x_2_1, x_2_2, x_2_3,
                           x_3_0, x_3_1, x_3_2, x_3_3):
        x_0 = x_0_0 + x_0_1 + x_0_2 + x_0_3
        x_1 = x_1_0 + x_1_1 + x_1_2 + x_1_3
        x_2 = x_2_0 + x_2_1 + x_2_2 + x_2_3
        x_3 = x_3_0 + x_3_1 + x_3_2 + x_3_3
        return torch.cat((x_0, x_1, x_2, x_3), dim=1)

    @staticmethod
    def combine_for_writing(x):
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)

    def forward_four_part_prior(self, y, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior,
                                y_spatial_prior_reduction=None, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params,
                                                          y_spatial_prior_reduction is None)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        y = y * q_enc

        y_res_0, y_q_0, y_hat_0, s_hat_0, m_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, s_hat_1, m_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_res_2, y_q_2, y_hat_2, s_hat_2, m_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_res_3, y_q_3, y_hat_3, s_hat_3, m_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        y_q = (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        y_hat = y_hat_so_far + y_hat_3
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)
        means_hat = (m_hat_0 + m_hat_1) + (m_hat_2 + m_hat_3)

        y_hat = y_hat * q_dec

        if write:
            y_q_w_0 = self.combine_for_writing(y_q_0)
            y_q_w_1 = self.combine_for_writing(y_q_1)
            y_q_w_2 = self.combine_for_writing(y_q_2)
            y_q_w_3 = self.combine_for_writing(y_q_3)
            scales_w_0 = self.combine_for_writing(s_hat_0)
            scales_w_1 = self.combine_for_writing(s_hat_1)
            scales_w_2 = self.combine_for_writing(s_hat_2)
            scales_w_3 = self.combine_for_writing(s_hat_3)
            return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
                scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat
        return y_res, y_q, y_hat, scales_hat, means_hat

    def compress_four_part_prior(self, y, common_params,
                                 y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                 y_spatial_prior_adaptor_3, y_spatial_prior,
                                 y_spatial_prior_reduction=None):
        return self.forward_four_part_prior(y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior,
                                            y_spatial_prior_reduction, write=True)

    def decompress_four_part_prior(self, common_params,
                                   y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                   y_spatial_prior_adaptor_3, y_spatial_prior,
                                   y_spatial_prior_reduction=None):
        _, quant_step, scales, means = self.separate_prior(common_params,
                                                           y_spatial_prior_reduction is None)
        if y_spatial_prior_reduction is not None:
            common_params = y_spatial_prior_reduction(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)

        scales_r = self.combine_for_writing(scales * mask_0)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_0
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_1)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_1
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_2)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_2
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        scales_r = self.combine_for_writing(scales * mask_3)
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_curr_step = (torch.cat((y_q_r, y_q_r, y_q_r, y_q_r), dim=1) + means) * mask_3
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * quant_step

        return y_hat