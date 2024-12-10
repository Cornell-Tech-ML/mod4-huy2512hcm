import pytest
from hypothesis import given, settings

import minitorch
from minitorch import Tensor

from .tensor_strategies import tensors
import numba
import numba.cuda
import numpy
import random


@pytest.mark.task4_1
def test_conv1d_simple() -> None:
    t = minitorch.tensor([0, 1, 2, 3]).view(1, 1, 4)
    t.requires_grad_(True)
    t2 = minitorch.tensor([[1, 2, 3]]).view(1, 1, 3)
    out = minitorch.Conv1dFun.apply(t, t2)

    assert out[0, 0, 0] == 0 * 1 + 1 * 2 + 2 * 3
    assert out[0, 0, 1] == 1 * 1 + 2 * 2 + 3 * 3
    assert out[0, 0, 2] == 2 * 1 + 3 * 2
    assert out[0, 0, 3] == 3 * 1


@pytest.mark.task4_1
@given(tensors(shape=(1, 1, 6)), tensors(shape=(1, 1, 4)))
def test_conv1d(input: Tensor, weight: Tensor) -> None:
    print(input, weight)
    minitorch.grad_check(minitorch.Conv1dFun.apply, input, weight)


@pytest.mark.task4_1
@given(tensors(shape=(2, 2, 6)), tensors(shape=(3, 2, 2)))
@settings(max_examples=50)
def test_conv1d_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv1dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(1, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
def test_conv(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
@settings(max_examples=10)
def test_conv_batch(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 2, 6, 6)), tensors(shape=(3, 2, 2, 4)))
@settings(max_examples=10)
def test_conv_channel(input: Tensor, weight: Tensor) -> None:
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
def test_conv2() -> None:
    t = minitorch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]).view(
        1, 1, 4, 4
    )
    t.requires_grad_(True)

    t2 = minitorch.tensor([[1, 1], [1, 1]]).view(1, 1, 2, 2)
    t2.requires_grad_(True)
    out = minitorch.Conv2dFun.apply(t, t2)
    out.sum().backward()

    minitorch.grad_check(minitorch.Conv2dFun.apply, t, t2)


if numba.cuda.is_available():

    @pytest.mark.task4_4b
    def test_conv1d_cuda() -> None:
        TEST_SAMPLES = 25
        for t_shape, w_shape in zip(
            [
                (2, 2, 12),
                (4, 3, 10),
                (18, 12, 8),
                (45, 5, 6),
                (8, 25, 9),
                (35, 7, 7),
                (7, 45, 8),
                (25, 6, 4),
            ],
            [
                (2, 2, 4),
                (3, 3, 2),
                (5, 12, 5),
                (7, 5, 3),
                (4, 25, 2),
                (7, 7, 3),
                (4, 45, 4),
                (4, 6, 2),
            ],
        ):
            for _ in range(TEST_SAMPLES):
                t_storage = numpy.array(
                    [random.random() * 1500 - 750 for __ in range(numpy.prod(t_shape))]
                )
                w_storage = numpy.array(
                    [random.random() * 1500 - 750 for __ in range(numpy.prod(w_shape))]
                )
                t_tensor = Tensor.make(
                    t_storage, t_shape, backend=minitorch.SimpleBackend
                )
                w_tensor = Tensor.make(
                    w_storage, w_shape, backend=minitorch.SimpleBackend
                )
                conv_out_a = minitorch.Conv1dFun.apply(t_tensor, w_tensor)
                conv_out_b = minitorch.cuda_conv.Conv1dFun.apply(t_tensor, w_tensor)
                numpy.testing.assert_allclose(
                    conv_out_a._tensor._storage, conv_out_b._tensor._storage, 1e-3, 1e-3
                )
                minitorch.grad_check(
                    minitorch.cuda_conv.Conv1dFun.apply, t_tensor, w_tensor
                )

    @pytest.mark.task4_4b
    def test_conv2d_cuda() -> None:
        TEST_SAMPLES = 25
        for t2d_shape, w2d_shape in zip(
            [
                (2, 2, 10, 10),
                (3, 3, 12, 12),
                (4, 5, 16, 16),
                (6, 25, 20, 10),
                (8, 35, 40, 9),
            ],
            [
                (2, 2, 4, 4),
                (3, 3, 5, 5),
                (4, 4, 6, 6),
                (6, 25, 10, 8),
                (8, 35, 7, 7),
            ],
        ):
            for _ in range(TEST_SAMPLES):
                t2d_storage = numpy.array(
                    [
                        random.random() * 1500 - 750
                        for __ in range(numpy.prod(t2d_shape))
                    ]
                )
                w2d_storage = numpy.array(
                    [
                        random.random() * 1500 - 750
                        for __ in range(numpy.prod(w2d_shape))
                    ]
                )
                t2d_tensor = Tensor.make(
                    t2d_storage, t2d_shape, backend=minitorch.SimpleBackend
                )
                w2d_tensor = Tensor.make(
                    w2d_storage, w2d_shape, backend=minitorch.SimpleBackend
                )
                conv2d_out_a = minitorch.Conv2dFun.apply(t2d_tensor, w2d_tensor)
                conv2d_out_b = minitorch.cuda_conv.Conv2dFun.apply(
                    t2d_tensor, w2d_tensor
                )
                numpy.testing.assert_allclose(
                    conv2d_out_a._tensor._storage,
                    conv2d_out_b._tensor._storage,
                    1e-3,
                    1e-3,
                )
                minitorch.grad_check(
                    minitorch.cuda_conv.Conv2dFun.apply, t2d_tensor, w2d_tensor
                )
