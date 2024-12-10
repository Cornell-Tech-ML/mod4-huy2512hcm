from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the sum of a and b."""
        return operators.add(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute gradients for addition."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the natural logarithm of a."""
        ctx.save_for_backward(a)
        return float(operators.log(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute gradient for logarithm."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class EQ(ScalarFunction):
    """Equality function $f(x, y) = 1 if x == y, 0 otherwise$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute equality of a and b."""
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for equality (always 0 gradient)."""
        return 0.0, 0.0


class LT(ScalarFunction):
    """Less than function $f(x, y) = 1 if x < y, 0 otherwise$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute less than comparison of a and b."""
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for less than (always 0 gradient)."""
        return 0.0, 0.0


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute product of a and b."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute gradients for multiplication."""
        (a, b) = ctx.saved_values
        return b * d_output, a * d_output


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute e^a."""
        ctx.save_for_backward(a)
        return float(operators.exp(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute gradient of e^x."""
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute 1/a."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute gradient of 1/x."""
        (a,) = ctx.saved_values
        return -(1.0 / (a * a)) * d_output


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute -a."""
        return float(-a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute gradient of -x."""
        return -d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute max(0, a)."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute gradient of ReLU."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute sigmoid of a."""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute gradient of sigmoid."""
        (a,) = ctx.saved_values
        return operators.sigmoid(a) * (1 - operators.sigmoid(a)) * d_output
