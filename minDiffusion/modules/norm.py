# -*- coding: utf-8 -*-

# Third-party imports
import torch.nn as nn


def get_norm_module(
    channels: int,
    norm_type: str = 'group_norm',
    num_groups: int = 1,
    **norm_kwargs,
) -> nn.Module:
    """
    Return the proper normalization module for input with shape (B, C, *).
    """
    if norm_type == 'group_norm':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels, **norm_kwargs)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization technique. Expected to be in ['none', 'group_norm'] but got {norm_type}")
