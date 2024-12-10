"""Collection of the core mathematical operators used throughout the code base."""

import math


# ## Task 0.1
from typing import Any, Callable, Iterable, Optional, TypeVar

#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Multiply two floats.

    Args:
    ----
        x: First float operand.
        y: Second float operand.

    Returns:
    -------
        The product of `x` and `y`.

    """
    return x * y


def id(x: float) -> float:
    """Return the input float value unchanged.

    Args:
    ----
        x: Input float value.

    Returns:
    -------
        The unchanged input value `x`.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two floats.

    Args:
    ----
        x: First float operand.
        y: Second float operand.

    Returns:
    -------
        The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a float.

    Args:
    ----
        x: Input float value.

    Returns:
    -------
        The negation of `x`.

    """
    return -1.0 * x


def lt(x: float, y: float) -> float:
    """Check if float `x` is less than float `y`.

    Args:
    ----
        x: First float operand.
        y: Second float operand.

    Returns:
    -------
        True if x is less than y, False otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if float `x` is equal to float `y`.

    Args:
    ----
        x: First float operand.
        y: Second float operand.

    Returns:
    -------
        True if `x` equals `y`, False otherwise.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two floats. If the two numbers are equal, return the first number.

    Args:
    ----
        x: First float operand.
        y: Second float operand.

    Returns:
    -------
        The larger value between `x` and `y`, or `x` if they are equal.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two floats are close in value.

    Args:
    ----
        x: First float operand.
        y: Second float operand.

    Returns:
    -------
        True if the absolute difference between `x` and `y` is less than 1e-2 (0.01), False otherwise.

    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Compute the sigmoid function using a numerically stable formula. The sigmoid function maps the input to a float value between 0 and 1.

    Args:
    ----
        x: Input float value.

    Returns:
    -------
        The float value between 0 and 1 that is the sigmoid of `x`.

    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) activation function. The ReLU function returns the input value if it is greater than 0, otherwise it returns 0.

    Args:
    ----
        x: Input float value.

    Returns:
    -------
        The float value of the ReLU of `x`, which is the maximum of 0 and `x`.

    """
    # Used a ternary operator to make it compatible with numba's JIT compilation
    return 0.0 if x < 0 else x


def log(x: float) -> float:
    """Compute the natural logarithm of a float. The natural logarithm is the logarithm of the input value to the base e, or ln(`x`).

    Args:
    ----
        x: Input float value. Assumed to be greater than zero.

    Returns:
    -------
        The natural logarithm of `x`.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function of a float. The exponential function is e raised to the power of the input value.

    Args:
    ----
        x: Input float value.

    Returns:
    -------
        e raised to the power of x.

    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the the derivative of the natural logarithm of the input value times a second argument. Note that the derivative of ln(`x`) with respect to `x` is 1/`x`.

    Args:
    ----
        x: Input float value. Assumed to be greater than zero.
        d: Second float argument.

    Returns:
    -------
        The float value of the derivative of the natural logarithm of x multiplied by d.

    """
    return d / x


def inv(x: float) -> float:
    """Compute the reciprocal of a float. The reciprocal, or multiplicative inverse, of `x` is 1/`x`.

    Args:
    ----
        x: Input float value. Assumed to be non-zero.

    Returns:
    -------
        The float value of the reciprocal of x.

    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of the reciprocal of the input value times a second argument. Note that the derivative of 1/x with respect to x is -1/(`x`^2).

    Args:
    ----
        x: Input float value. Assumed to be non-zero.
        d: Second float argument.

    Returns:
    -------
        The float value of the derivative of the reciprocal of `x` multiplied by `d`.

    """
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of the ReLU of the input value times a second argument. Note that the derivative of ReLU(`x`) with respect to `x` is 1 if `x` > 0, otherwise it is 0.

    Args:
    ----
        x: Input float value.
        d: Second float argument.

    Returns:
    -------
        The float value of the derivative of the ReLU of `x` multiplied by `d`.

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Higher-order functions
MapInputType = TypeVar("MapInputType")
MapOutputType = TypeVar("MapOutputType")


def map(
    function: Callable[[MapInputType], MapOutputType],
) -> Callable[[Iterable[MapInputType]], Iterable[MapOutputType]]:
    """Return a function that applies the input function to each element of the input iterable and returns a new iterable with the results. This is a higher-order function that takes an input function and returns an output function.

    Args:
    ----
        function: Function to apply to each element of the input iterable.

    Returns:
    -------
        A function that applies `function` to each element of the input iterable and returns a new iterable with the results.

    """

    def map_function(iterable: Iterable[MapInputType]) -> Iterable[MapOutputType]:
        for item in iterable:
            yield function(item)

    return map_function


ZipWithInputType1 = TypeVar("ZipWithInputType1")
ZipWithInputType2 = TypeVar("ZipWithInputType2")
ZipWithOutputType = TypeVar("ZipWithOutputType")


def zipWith(
    function: Callable[[ZipWithInputType1, ZipWithInputType2], ZipWithOutputType],
) -> Callable[
    [Iterable[ZipWithInputType1], Iterable[ZipWithInputType2]],
    Iterable[ZipWithOutputType],
]:
    """Return a function that applies the input function to combine corresponding elements of two input iterables and returns a new iterable with the results. This is a higher-order function that takes an input function and returns an output function.

    Args:
    ----
        function: Function to apply to combine corresponding elements of two input iterables.

    Returns:
    -------
        A function that applies `function` to combine corresponding elements of two input iterables and returns a new iterable with the results.

    """

    def zipWith_function(
        iterable1: Iterable[ZipWithInputType1], iterable2: Iterable[ZipWithInputType2]
    ) -> Iterable[ZipWithOutputType]:
        sentinel: Any = object()
        iterator1, iterator2 = iter(iterable1), iter(iterable2)
        while True:
            item1, item2 = next(iterator1, sentinel), next(iterator2, sentinel)
            if item1 is sentinel or item2 is sentinel:
                return
            yield function(item1, item2)

    return zipWith_function


ReduceType = TypeVar("ReduceType")


def reduce(
    function: Callable[[ReduceType, ReduceType], ReduceType],
    initial: Optional[ReduceType] = None,
) -> Callable[[Iterable[ReduceType]], ReduceType]:
    """Return a function that applies the input function to reduce the input iterable to a single value. This is a higher-order function that takes an input function and returns an output function.

    Args:
    ----
        function: Function to apply to all elements of the input iterable to reduce it to a single value.
        initial: Optional initial value for the reduction. If not provided, the first element of the input iterable is used as the initial value.

    Returns:
    -------
        A function that applies `function` to all elements of the input iterable to reduce it to a single value.

    """

    def reduce_function(
        iterable: Iterable[ReduceType],
    ) -> ReduceType:
        iterator = iter(iterable)
        if initial is None:
            reduction = next(iterator)
        else:
            reduction = initial
        for item in iterator:
            reduction = function(reduction, item)
        return reduction

    return reduce_function


# List functions built using higher-order functions and simpler operators
def negList(numbers: Iterable[float]) -> Iterable[float]:
    """Return a new list of floats with each float of the input list negated.

    Args:
    ----
        numbers: List of float values.

    Returns:
    -------
        A new list with each float negated.

    """
    return list(map(neg)(numbers))


def addLists(list1: Iterable[float], list2: Iterable[float]) -> Iterable[float]:
    """Return a new list of floats consisting of each float of the input list `list1` added to the corresponding float of the input list `list2`.

    Args:
    ----
        list1: List of float values.
        list2: List of float values.

    Returns:
    -------
        A new list with each float in `list1` added to the corresponding float in `list2`.

    """
    return list(zipWith(add)(list1, list2))


def sum(numbers: Iterable[float]) -> float:
    """Return the sum of all the float elements in a list.

    Args:
    ----
        numbers: List of float values.

    Returns:
    -------
        The sum of all the float elements in `numbers`.

    """
    return reduce(add, 0.0)(numbers)


def prod(numbers: Iterable[float]) -> float:
    """Return the product of all the float elements in a list.

    Args:
    ----
        numbers: List of float values.

    Returns:
    -------
        The product of all the float elements in `numbers`.

    """
    return reduce(mul, 1.0)(numbers)
