
def identity(x):
    return x


from torch import nn 
import torch
def get(identifier):
    """Returns function.

    Args:
        identifier: Function or string.

    Returns:
        Function corresponding to the input string or input function.
    """
    if identifier is None:
        return identity
    if isinstance(identifier, str):
        return {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "selu": nn.SELU,
            "sigmoid": nn.Sigmoid,
            "silu": nn.SiLU,  #swish
            "sin": torch.sin,
            "swish": nn.SiLU,
            "tanh": nn.Tanh,
        }[identifier]
    if callable(identifier):
        return identifier
    raise TypeError(
        "Could not interpret activation function identifier: {}".format(identifier)
    )
