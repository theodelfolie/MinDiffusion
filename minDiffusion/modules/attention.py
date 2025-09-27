# -*- coding: utf-8 -*-

# Python System imports
from typing import Optional, Union

# Third-party imports
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

# Relative imports
from .norm import get_norm_module


class MyMultiheadAttention(nn.Module):
    """
    Simple custom MHA (similar to `nn.MultiheadAttention`) that supports Flash Attention.
    This class assumes that keys, querys and values share the same embedding space (have the same dimension).
    Also assumes input shape = (B, T, C) (equivalent to setting batch_first=True in nn.MultiheadAttention).
    Implementation inspired from: https://github.com/mikaylagawarecki/transformer_tutorial_accompaniment/blob/main/mha.py
    and https://github.com/karpathy/nanoGPT/blob/master/model.py.

    Args:
        embed_dim (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout probability in SDPA.
        bias (bool): Use bias in projections.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim is not divisible by nheads"
        assert hasattr(torch.nn.functional, 'scaled_dot_product_attention'), \
            "This class requires Flash Attention implementation: you need pytorch >= 2.0 to use it."
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias

        # key, query, value projections for all heads
        out_dim = 3 * embed_dim
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=bias, **factory_kwargs)
        # We try to follow the default PyTorch MHA convention, to easily compare results.
        self.in_proj_weight = self.in_proj.weight
        self.in_proj_bias = self.in_proj.bias
        if bias:
            self.in_proj_bias.data.zero_()  # Following Pytorch convention
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        if bias:
            self.out_proj.bias.data.zero_()  # Following Pytorch convention

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        causal: bool = False,
    ):
        B, T, C = query.size()  # batch size, sequence length, embed_dim

        # Apply input projection
        if (query is key) and (key is value):
            # self attention: query, key and value are the same
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            result_packed_proj = self.in_proj(query)
            query, key, value = torch.chunk(result_packed_proj, 3, dim=-1)
        else:
            # key, query and value are different (e.g., cross attention). In this case we have to manually split the weights and biases.
            q_weight, k_weight, v_weight = torch.chunk(self.in_proj.weight, 3, dim=0)
            if self.bias:
                q_bias, k_bias, v_bias = torch.chunk(self.in_proj.bias, 3, dim=0)
            else:
                q_bias, k_bias, v_bias = None, None, None
            query, key, value = F.linear(query, q_weight, q_bias), F.linear(key, k_weight, k_bias), F.linear(value, v_weight, v_bias)

        key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)

        # Efficient Scaled Dot Product Attention using Flash Attention CUDA kernels
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0., is_causal=causal)  # (B, num_heads, T, head_dim)

        x = x.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        x = self.out_proj(x)
        return x


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    This block implements the following operation: [Norm -> SelfAttention -> Add] when norm_first = True.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.,
        norm_type: str = 'group_norm',
        num_groups: int = 32,
        norm_eps: float = 1e-5,
        norm_first: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm_first = norm_first

        self.norm = get_norm_module(
            channels=embed_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            eps=norm_eps,
            device=device,
            dtype=dtype,
        )
        self.self_attn = MyMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)  # flatten the spatial dims (H, W) -> (B, C, spatial)

        residual = x
        if self.norm_first:
            x = self.norm(x)
        x = x.permute(0, 2, 1)  # (B, spatial, C)
        x = self.self_attn(x, x, x)
        x = x.permute(0, 2, 1)  # (B, C, spatial)
        if not self.norm_first:
            x = self.norm(x)

        x = x + residual
        return x.reshape(B, C, H, W)
