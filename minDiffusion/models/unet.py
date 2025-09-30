# -*- coding: utf-8 -*-

# Python System imports
from typing import Optional
import math

# Third-party imports
import torch
from torch import nn, Tensor
from torch.nn import functional as F

# Relative imports
from ..modules.conv import MyConv2d, MyConvTranspose2d
from ..modules.norm import get_norm_module
from ..modules.activation import get_activation_fn


class UpsampleBlock2D(nn.Module):
    """
    Upsampling block with optional transposed convolution.

    Args:
        channels (int): Number of input and output channels.
        use_conv_tr (bool): If True, use a learnable transposed convolution
            for upsampling. If False, use nearest-neighbor interpolation.
        stride (int, optional): Upsampling factor. Default: 2.
        kernel_size (int, optional): Kernel size for transposed convolution.
            Ignored if `use_conv_tr=False`. Default: 3.
    """

    def __init__(
        self,
        channels: int,
        use_conv_tr: bool = True,
        stride: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.channels = channels
        self.use_conv_tr = use_conv_tr
        self.stride = stride

        if use_conv_tr:
            self.conv_tr = MyConvTranspose2d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=stride,
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input of shape (B, C, H, W).

        Returns:
            Tensor: Upsampled feature map of shape
                (B, C, H*stride, W*stride).
        """
        if self.use_conv_tr:
            return self.conv_tr(x)
        return F.interpolate(x, scale_factor=self.stride, mode="nearest")


class DownsampleBlock2D(nn.Module):
    """
    Downsampling block with either strided convolution or pooling.

    Args:
        channels (int): Number of input and output channels.
        use_conv (bool, optional): If True, use strided convolution.
            Otherwise, use average pooling. Default: True.
        stride (int, optional): Downsampling factor. Default: 2.
        kernel_size (int, optional): Kernel size for convolution.
            Ignored if `use_conv=False`. Default: 3.
    """
    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        stride: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.stride = stride
        if use_conv:
            self.downsample_op = MyConv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=stride,
            )
        else:
            self.downsample_op = nn.AvgPool2d(stride)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input of shape (B, C, H, W).

        Returns:
            Tensor: Downsampled feature map of shape
                (B, C, H/stride, W/stride).
        """
        return self.downsample_op(x)


class ResBlock2D(nn.Module):
    """
    Residual block with optional channel projection and timestep conditioning.

    Args:
        in_channels (int): Number of input channels.
        emb_channels (int): Dimension of timestep embeddings.
        dropout (float): Dropout probability.
        out_channels (int, optional): If provided, the number of output channels.
            Otherwise, equals `in_channels`. Default: None.
        use_conv (bool, optional): If True and `out_channels` differs
            from `in_channels`, use a spatial convolution for skip connection.
            Otherwise, use a 1x1 conv to change the channels in the skip connection.
            Default: False.
        use_scale_shift_norm (bool, optional): If True, use feature-wise affine modulation
            conditioning from embeddings (FiLM like conditioning).
            Otherwise, embeddings are simply summed to the input. Default: False.
        activation (str, optional): Activation type (e.g., "silu", "relu"). Default: "silu".
        norm_type (str, optional): Normalization type. Default: "group_norm".
        num_groups (int, optional): num_groups params for GroupNorm (if used). Default: 32.
        kernel_size (int, optional): Kernel size for all convolutions. Default: 3.
    """
    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        activation: str = 'silu',
        norm_type: str = 'group_norm',
        num_groups: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or in_channels
        self.use_scale_shift_norm = use_scale_shift_norm

        activation_fn = get_activation_fn(activation)

        self.in_layers = nn.Sequential(
            get_norm_module(
                channels=in_channels,
                norm_type=norm_type,
                num_groups=num_groups,
            ),
            activation_fn,
            MyConv2d(
                in_channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=1,
            )
        )

        self.emb_layers = nn.Sequential(
            activation_fn,
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            get_norm_module(
                channels=in_channels,
                norm_type=norm_type,
                num_groups=num_groups,
            ),
            activation_fn,
            nn.Dropout(p=dropout),
            MyConv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=1,
                init='zeros',
            )
        )

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = MyConv2d(
                in_channels,
                self.out_channels,
                kernel_size=kernel_size if use_conv else 1,
                stride=1,
            )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C_in, H, W).
            emb (Tensor): Timestep embedding of shape (B, emb_channels).

        Returns:
            Tensor: Output tensor of shape (B, C_out, H, W).
        """
        h = self.in_layers(x)  # (B, C_out, H, W)
        emb_out = self.emb_layers(emb).to(h.dtype)[..., None, None]  # (B, C_out, 1, 1) if not use_scale_shift_norm else (B, 2*C_out, 1, 1)

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift  # (1 + scale) for stability
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps (Tensor): A 1D tensor of shape (B,). Values may be fractional.
        dim (int): Dimension of the embedding.
        max_period (int, optional): Minimum frequency of the embeddings. Default: 10000.

    Returns:
        Tensor: Sinusoidal embeddings of shape (B, dim).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
