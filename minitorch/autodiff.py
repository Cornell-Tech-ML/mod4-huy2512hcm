from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_plus = list(vals)
    vals_plus[arg] += epsilon
    vals_minus = list(vals)
    vals_minus[arg] -= epsilon
    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative of x."""
        ...

    @property
    def unique_id(self) -> int:
        """Return the unique identifier."""
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node."""
        ...

    def is_constant(self) -> bool:
        """Check if the variable is constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parents of self."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute derivatives."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    lst = deque()

    def dfs(v: Variable) -> None:
        if not (v.unique_id in visited or v.is_constant()):
            visited.add(v.unique_id)
            for p in v.parents:
                dfs(p)
            lst.appendleft(v)

    dfs(variable)
    return lst


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: The derivative of the final output with respect to this variable

    Returns:
    -------
    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    order = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv

    for var in order:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors."""
        return self.saved_values
