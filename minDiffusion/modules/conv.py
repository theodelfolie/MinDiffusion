"""
This module provides utilities functions for convolutions and custom convolutional layers to simplify
handling of padding and output shape inference in 2D convolutions and transpose convolutions.
"""

# -*- coding: utf-8 -*-

# Python System imports
import typing as tp
from typing import Union, Tuple, Optional
import math
import warnings

# Third-party imports
import torch
from torch import nn
from torch.nn import functional as F


def pad4d(x: torch.Tensor, paddings: tp.Tuple[int, int, int, int], mode: str = 'constant', value: float = 0.):
    """Tiny wrapper around F.pad. Need a fixed size argument for static typing in TorchScript.
    Arguments:
        x (torch.Tensor): Input tensor
        paddings (Tuple[int, int, int, int]): (W_left, W_right, H_left, H_right)
        mode (str): padding mode for F.pad
        value (float): Value with which to pad the tensor
    """
    assert len(x.shape) == 4, "Only 4d supported"
    assert len(paddings) == 4, f"Expected paddings to have len == 4 (W_left, W_right, H_left, H_right) but got {len(paddings)}"
    padding_left_W, padding_right_W, padding_left_H, padding_right_H = paddings
    assert padding_left_W >= 0 and padding_right_W >= 0 and padding_left_H >= 0 and padding_right_H >= 0, \
        f"padding expected to be >=0, got {(padding_left_W, padding_right_W, padding_left_H, padding_right_H)}"
    return F.pad(x, paddings, mode, value)


def unpad4d(x: torch.Tensor, paddings: tp.Tuple[int, int, int, int]):
    """Remove padding from x, handling properly zero padding."""
    padding_left_W, padding_right_W, padding_left_H, padding_right_H = paddings
    assert padding_left_W >= 0 and padding_right_W >= 0 and padding_left_H >= 0 and padding_right_H >= 0, \
        f"padding expected to be >=0, got {(padding_left_W, padding_right_W, padding_left_H, padding_right_H)}"
    end_W = x.shape[-1] - padding_right_W
    end_H = x.shape[-2] - padding_right_H
    return x[..., padding_left_H: end_H, padding_left_W: end_W]


def get_extra_padding_for_conv(length: int, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def reset_parameters(module, init_type: Optional[str] = None):
    """Initialize weights and bias based on init_type."""
    if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        return  # ignore non-conv layers
    if init_type is None:
        # Fallback: use PyTorch default init
        return
    elif init_type == "zeros":
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    else:
        raise ValueError(f"Unsupported init type: {init_type}")


class MyConv2d(nn.Module):
    """
    Conv2d with custom handling of padding, making it easier to infer output shapes.

    Note on padding:

    We pad on both dimensions with total_padding = (K - S).
    If input shape = (*, in_1, in_2) and output shape = (*, out_1, out_2), based on the output shape formula for convolutions (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html):
    out_i = 1 + ((in_i + total_padding - K) / S) will simplify into out_i = in_i / S, for i = [1, 2].
    If S doesn't divide `in`, it means that the last window is not full. In this case, we use some extra padding generated with `get_extra_padding_for_conv` function.
    Therefore, the output shape for both dimensions will be out = ceil(in / S).
    total_padding is added equally on both sides of the given dimension.
    extra_padding is always added on the right side.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        pad_mode: str = 'constant',
        dilation: Union[int, Tuple[int, int]] = (1, 1),
        groups: int = 1,
        bias: bool = True,
        init: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,  # padding is handled outside of nn.conv for more flexibility
            groups=groups,
            bias=bias,
            dilation=dilation,
            **factory_kwargs,
        )
        # warn user on unusual setup between dilation and stride
        for idx_dim, stride_dim in enumerate(self.conv.stride):
            dilation_dim = self.conv.dilation[idx_dim]
            if stride_dim > 1 and dilation_dim > 1:
                warnings.warn(f'MyConv2d has been initialized with stride > 1 and dilation > 1 on dimension {idx_dim} (stride={stride_dim}, dilation={dilation_dim}).')

        self.pad_mode = pad_mode
        self.apply(lambda m: reset_parameters(m, init_type=init))

    def forward(self, x):
        B, C, H, W = x.shape
        kernel_size_H, kernel_size_W = self.conv.kernel_size
        stride_H, stride_W = self.conv.stride
        dilation_H, dilation_W = self.conv.dilation
        kernel_size_H = (kernel_size_H - 1) * dilation_H + 1  # Effective kernel size with dilation
        kernel_size_W = (kernel_size_W - 1) * dilation_W + 1  # Effective kernel size with dilation
        padding_total_H = (kernel_size_H - stride_H)
        padding_total_W = (kernel_size_W - stride_W)
        extra_padding_H = get_extra_padding_for_conv(H, kernel_size_H, stride_H, padding_total_H)
        extra_padding_W = get_extra_padding_for_conv(W, kernel_size_W, stride_W, padding_total_W)

        # Along each dimension: always pad left/right equally + extra padding right
        padding_right_H = padding_total_H // 2
        padding_left_H = padding_total_H - padding_right_H
        padding_right_W = padding_total_W // 2
        padding_left_W = padding_total_W - padding_right_W
        x = pad4d(x, (padding_left_W, padding_right_W + extra_padding_W, padding_left_H, padding_right_H + extra_padding_H), mode=self.pad_mode)

        y = self.conv(x)
        return y


class MyConvTranspose2d(nn.Module):
    """
    ConvTranspose2d with custom handling of padding, making it easier to infer output shapes.

    Note on padding:

    This class assumes that padding has been applied in encoding convolutions on both dimensions with total_padding = (K - S). See MyConv2d class for more details.
    Therefore, after passing through the convTransposed, we remove the padding with the unpad4d function so that the output shape is consistent.
    If input shape = (*, in_1, in_2) and output shape = (*, out_1, out_2), based on the output shape formula for convolutions (https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html):
    out_i = (in_i + 1) * S - total_padding + K will simplify into out_i = in_i * S with total_padding = (K - S).
    Therefore, the output shape for both dimensions will be out = in * S.
    Warning: we don't remove potential extra_padding that may have been added in the encoder. This has to be taken care of outside of the class!
    This is because removing it here would require also passing the length at the matching layer in the encoder.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.convtr = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,  # padding is handled outside of nn.convtr for more flexibility
            output_padding=0,  # padding is handled outside of nn.convtr for more flexibility
            groups=groups,
            bias=bias,
            **factory_kwargs,
        )

    def forward(self, x):
        kernel_size_H, kernel_size_W = self.convtr.kernel_size
        stride_H, stride_W = self.convtr.stride
        padding_total_H = (kernel_size_H - stride_H)
        padding_total_W = (kernel_size_W - stride_W)
        padding_right_H = padding_total_H // 2
        padding_left_H = padding_total_H - padding_right_H
        padding_right_W = padding_total_W // 2
        padding_left_W = padding_total_W - padding_right_W

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `get_extra_padding_for_conv` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        y = unpad4d(y, (padding_left_W, padding_right_W, padding_left_H, padding_right_H))
        return y
