from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling"""
    batch, channel, height, width = input.shape
    kh, kw = kernel

    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # Reshape to separate kernel windows
    out = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Permute to get kernel windows as last dimension
    out = out.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Rearrange dimensions to align pooling windows
    out = out.view(batch, channel, new_height, new_width, kh * kw)

    return out, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Average pool the input using the given kernel."""
    batch, channel, _, _ = input.shape
    # Get the tiled tensor
    tiled_tensor, new_height, new_width = tile(input, kernel)
    out = tiled_tensor.mean(dim=-1)
    return out.view(batch, channel, new_height, new_width)


# Task 4.4
max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot encoding"""
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass of max."""
        dim_int = int(dim.item())
        ctx.save_for_backward(input, dim_int)
        return max_reduce(input, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass of max."""
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the maximum of all elements or along a dimension."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute softmax along a dimension."""
    exp_input = input.exp()
    sum_exp = exp_input.sum(dim=dim)
    return exp_input / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute log softmax along a dimension."""
    return input - input.exp().sum(dim=dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Max pool the input using the given kernel."""
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    return max(tiled, dim=-1).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float = 0.5, ignore: bool = False) -> Tensor:
    """Apply dropout to the input."""
    if ignore or rate == 0:
        return input

    mask = rand(input.shape) > rate
    return input * mask
