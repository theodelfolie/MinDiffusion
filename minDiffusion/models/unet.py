# -*- coding: utf-8 -*-

# Python System imports
from typing import Optional, Sequence
import math

# Third-party imports
import torch
from torch import nn, Tensor
from torch.nn import functional as F

# Relative imports
from ..modules.conv import MyConv2d, MyConvTranspose2d
from ..modules.norm import get_norm_module
from ..modules.activation import get_activation_fn
from ..modules.attention import AttentionBlock


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
                channels=self.out_channels,
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


class EmbeddingSequential(nn.Sequential):
    """
    A sequential container that supports layers with optional timestep embeddings inputs.

    This class extends nn.Sequential to handle modules that may
    require a timestep embedding (`t_emb` in ResBlock) in their forward pass.
    This allows mixing standard layers (e.g., convolutions, normalizations,
    activations) with timestep-conditioned residual blocks inside the same
    sequential container.

    Examples:
    --------
    >>> seq = CustomSequential(
    ...     ResBlock2D(in_channels=64, time_emb_dim=128, dropout=0.1),
    ...     nn.ReLU(),
    ...     nn.Conv2d(64, 128, kernel_size=3, padding=1),
    ... )
    >>> x = torch.randn(8, 64, 32, 32)
    >>> t_emb = torch.randn(8, 128)
    >>> out = seq(x, t_emb)  # ResBlock2D receives t_emb, others do not
    """
    def forward(self, x, t_emb=None):
        for layer in self:
            if isinstance(layer, ResBlock2D):  # needs t_emb
                x = layer(x, t_emb)
            else:  # regular conv/norm/activation
                x = layer(x)
        return x


class DiffusionUNet(nn.Module):
    """
    U-Net backbone for diffusion models with timestep embeddings and optional class conditioning.

    This model architecture follows the core design of DDPM/Improved-DDPM (Ho et al. 2020, Nichol & Dhariwal 2021),
    with minor architectural changes.
    It consists of an encoder–decoder with residual blocks conditioned on sinusoidal timestep embeddings,
    attention blocks, and skip connections.

    Args:
        in_channels (int): Number of channels in the input tensor.
        model_channels (int): Base channel count for the model (in_channels when entering the encoder).
        out_channels (int): Number of channels in the output tensor.
        num_res_blocks (int): Number of residual blocks per resolution.
        attention_resolutions (Sequence[int]): Resolutions at which to apply attention.
        May be a set, list, or tuple. For example, if this contains 4, then at 4x downsampling, an attention
        block will be inserted after the resblock.
        dropout (float, optional): Dropout probability. Default = 0.0.
        channel_growth (Sequence[int], optional): Channel growth factor (multiplier) at each resolution.
        Increases the number of hidden channels by this factor at each encoder layer. Default = (1, 2, 1, 1).
        conv_resample (bool, optional): If True, use learned convolutions for resampling
        Uses strided convolutions in Downsampling blocks and Transposed convolutions in Upsampling blocks. Default = True.
        num_classes (int, optional): If specified, enables class-conditional training. Default = None.
        num_heads (int, optional): Number of attention heads in AttentionBlock. Default = 1.
        use_scale_shift_norm (bool, optional): If True, use a FiLM-like conditioning mechanism for timestep embedding.
        If False, the embedding is summed to the input. Default = False.
        activation (str, optional): Activation function name (e.g. 'silu'). Default = 'silu'.
        norm_type (str, optional): Normalization type ('group_norm', etc.). Default = 'group_norm'.
        num_groups (int, optional): Groups for GroupNorm. Default = 32.
        kernel_size (int, optional): Kernel size for all convolutions. Default = 3.
        stride (int, optional): Stride used for down/upsampling at each resolution. Default = 2.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int = 1,
        attention_resolutions: Sequence[int] = (4, 8),
        dropout: float = 0.,
        channel_growth: Sequence[int] = (1, 2, 1, 1),
        conv_resample: bool = True,
        num_classes: Optional[int] = None,
        num_heads: int = 1,
        use_scale_shift_norm: bool = False,
        activation: str = 'silu',
        norm_type: str = 'group_norm',
        num_groups: int = 32,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_growth = channel_growth
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads

        activation_fn = get_activation_fn(activation)

        time_embed_dim = model_channels * 4  # TODO: make it a param ?
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            activation_fn,
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Conditioning on image labels
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.conv_in = MyConv2d(
            in_channels,
            model_channels,
            kernel_size=kernel_size,
            stride=1,
        )

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])

        common_resblock_kwargs = {
            "kernel_size": kernel_size,
            "norm_type": norm_type,
            "num_groups": num_groups,
            "activation": activation,
            "use_scale_shift_norm": use_scale_shift_norm,
        }

        channels_in = model_channels
        downsampling_factor = 1
        for level, mult in enumerate(channel_growth):
            encode = []
            decode = []
            channels_out = mult * channels_in
            for idx_res_block, _ in enumerate(range(num_res_blocks)):
                encode += [
                    ResBlock2D(
                        channels_in if idx_res_block == 0 else channels_out,
                        time_embed_dim,
                        dropout,
                        out_channels=channels_out,
                        **common_resblock_kwargs,
                    )
                ]
                if downsampling_factor in attention_resolutions:
                    encode.append(
                        AttentionBlock(
                            channels_out,
                            num_heads=num_heads,
                            dropout=dropout,
                        )
                    )
            if level != len(channel_growth) - 1:
                encode.append(
                    DownsampleBlock2D(channels_out, conv_resample, stride=stride, kernel_size=kernel_size)
                )
                downsampling_factor *= stride
            self.encoder.append(EmbeddingSequential(*encode))

            for idx_res_block, _ in enumerate(range(num_res_blocks)):
                decode += [
                    ResBlock2D(
                        2 * channels_out if idx_res_block == 0 else channels_in,
                        time_embed_dim,
                        dropout,
                        out_channels=channels_in,
                        **common_resblock_kwargs,
                    )
                ]
                if downsampling_factor in attention_resolutions:
                    decode.append(
                        AttentionBlock(
                            channels_in,
                            num_heads=num_heads,
                            dropout=dropout,
                        )
                    )
            if level != len(channel_growth) - 1:
                decode.append(UpsampleBlock2D(channels_in, conv_resample, stride=stride, kernel_size=kernel_size))
            self.decoder.insert(0, EmbeddingSequential(*decode))

            channels_in = channels_out

        self.bottleneck_block = EmbeddingSequential(
            ResBlock2D(
                channels_out,
                time_embed_dim,
                dropout,
                **common_resblock_kwargs,
            ),
            AttentionBlock(channels_out, num_heads=num_heads, dropout=dropout),
            ResBlock2D(
                channels_out,
                time_embed_dim,
                dropout,
                **common_resblock_kwargs,
            ),
        )

        self.conv_out = nn.Sequential(
            get_norm_module(
                channels=model_channels,
                norm_type=norm_type,
                num_groups=num_groups,
            ),
            activation_fn,
            MyConv2d(
                model_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                init='zeros',
            )
        )

    @property
    def inner_dtype(self):
        """Return the dtype used by encoder parameters (useful for mixed precision)."""
        return next(self.encoder.parameters()).dtype

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the U-Net.

        Args:
            x (Tensor): Input batch of shape (B, C, H, W).
            timesteps (Tensor): 1-D batch of timestep indices (B,).
            y (Tensor, optional): Class labels (B,), if class-conditional.

        Returns:
            Tensor: Output batch of shape [N, out_channels, H, W].
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        skips = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        x = x.to(self.inner_dtype)
        x = self.conv_in(x)
        for encoder_layer in self.encoder:
            x = encoder_layer(x, emb)
            skips.append(x)
        x = self.bottleneck_block(x, emb)
        for decoder_layer in self.decoder:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = decoder_layer(x, emb)
        return self.conv_out(x)
