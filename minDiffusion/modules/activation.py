# -*- coding: utf-8 -*-

# Python System imports
from typing import Union, Callable

# Third-party imports
from torch import nn, Tensor


def get_activation_fn(
    activation: Union[str, Callable[[Tensor], Tensor]]
) -> Union[str, Callable[[Tensor], Tensor]]:
    """
    Helper function to map an activation string to the activation class.
    If the supplied activation is not a string, the activation is passed back.

    Args:
        activation (str, or Callable[[Tensor], Tensor]): Activation function
    """
    if isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "silu":
            return nn.SiLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "selu":
            return nn.SELU()
        elif activation == "none":
            return nn.Identity()
        else:
            raise ValueError("activation should be relu/silu/gelu/selu/none, not {}".format(activation))
    return activation
