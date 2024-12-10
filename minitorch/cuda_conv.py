# type: ignore
# Currently pyright doesn't support numba.cuda
from typing import Tuple

import numba
from numba import cuda
from typing import TypeVar, Any
import numba.cuda
from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Storage,
    Strides,
    UserShape,
    to_index,
    index_to_position,
    broadcast_index,
)
from .tensor_functions import Function
from numba.cuda import jit as _jit

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Device jit function

    Args:
    ----
        fn : function
        **kwargs : argument

    Returns:
    -------
        jit

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Any, **kwargs: Any) -> FakeCUDAKernel:
    """Jit function

    Args:
    ----
        fn : function
        **kwargs : argument

    Returns:
    -------
        FakeCUDAKernel

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """Implementation of 1D Convolution.

    Performs 1D convolution on an input tensor using a weight kernel, with optional
    kernel reversal for different anchoring modes.

    Args:
    ----
    out (Storage): Storage for the output tensor.
    out_shape (Shape): Shape of the output tensor (batch, out_channels, width).
    out_strides (Strides): Strides of the output tensor.
    out_size (int): Total size of the output tensor.
    input (Storage): Storage for the input tensor.
    input_shape (Shape): Shape of the input tensor (batch, in_channels, width).
    input_strides (Strides): Strides of the input tensor.
    weight (Storage): Storage for the kernel weights.
    weight_shape (Shape): Shape of the weight tensor (out_channels, in_channels, kernel_width).
    weight_strides (Strides): Strides of the weight tensor.
    reverse (bool): Determines weight orientation (left or right).

    Returns:
    -------
    None: Updates the `out` storage in place.

    Notes:
    -----
    - This method assumes the `out` tensor is pre-allocated with appropriate size and shape.
    - Handles parallel computation using CUDA shared memory.

    """
    # Define block dimensions
    BLOCK_DIM = 16
    BLOCK_DIM2 = 32

    batch_out, out_channels, out_width = out_shape
    batch_in, in_channels, input_width = input_shape
    w_out_channels, w_in_channels, kernel_width = weight_shape

    # Ensure batch size, input-output channels, and width are consistent
    assert batch_out == batch_in and out_channels == w_out_channels
    assert in_channels == w_in_channels and out_width <= input_width

    # Calculate indices for thread positions
    thread_width_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    block_width_start = cuda.blockIdx.x * cuda.blockDim.x
    thread_channel_idx = cuda.blockIdx.z
    thread_px, thread_py = cuda.threadIdx.x, cuda.threadIdx.y

    # Allocate shared memory for kernel weights and input cache
    kernel_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    input_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM2), numba.float64)

    # Define strides for weight, input, and output tensors
    w_stride0, w_stride1, w_stride2 = weight_strides
    i_stride0, i_stride1, i_stride2 = input_strides
    o_stride0, o_stride1, o_stride2 = out_strides

    # Set the direction of kernel traversal
    kernel_direction = -1 if reverse else 1

    for batch_idx in range(batch_out):
        # Initialize output accumulator
        result_accumulator = 0.0

        # Iterate over channels in blocks
        for channel_block_start in range(0, in_channels, BLOCK_DIM):
            channel_cache_idx = channel_block_start + thread_px

            # Iterate over kernel width in blocks
            for kernel_block_start in range(0, kernel_width, BLOCK_DIM):
                kernel_cache_idx = thread_py + kernel_block_start

                # Cache kernel weights
                if channel_cache_idx < in_channels and kernel_cache_idx < kernel_width:
                    weight_idx = (
                        thread_channel_idx * w_stride0
                        + channel_cache_idx * w_stride1
                        + kernel_cache_idx * w_stride2
                    )
                    kernel_cache[thread_px, thread_py] = weight[weight_idx]
                else:
                    kernel_cache[thread_px, thread_py] = 0.0

                numba.cuda.syncthreads()

                # Cache input data
                for width_offset in range(0, BLOCK_DIM2, BLOCK_DIM):
                    if reverse:
                        position = (
                            block_width_start
                            - kernel_block_start
                            - BLOCK_DIM
                            + 1
                            + width_offset
                            + thread_py
                        )
                    else:
                        position = (
                            block_width_start
                            + kernel_block_start
                            + width_offset
                            + thread_py
                        )

                    if channel_cache_idx < in_channels and 0 <= position < input_width:
                        input_cache_idx = (
                            batch_idx * i_stride0
                            + channel_cache_idx * i_stride1
                            + position * i_stride2
                        )
                        input_cache[thread_px, width_offset + thread_py] = input[
                            input_cache_idx
                        ]
                    else:
                        input_cache[thread_px, width_offset + thread_py] = 0.0

                numba.cuda.syncthreads()

                # Perform convolution
                if thread_py == 0 and thread_width_idx < out_width:
                    for inner_channel_idx in range(
                        channel_block_start,
                        min(in_channels, channel_block_start + BLOCK_DIM),
                    ):
                        for inner_kernel_idx in range(
                            kernel_block_start,
                            min(kernel_width, kernel_block_start + BLOCK_DIM),
                        ):
                            position = (
                                thread_width_idx + inner_kernel_idx * kernel_direction
                            )

                            if reverse:
                                min_bound = (
                                    block_width_start
                                    - kernel_block_start
                                    - BLOCK_DIM
                                    + 1
                                )
                            else:
                                min_bound = block_width_start + kernel_block_start

                            max_bound = min_bound + BLOCK_DIM2

                            if (
                                min_bound <= position < max_bound
                                and 0 <= position < input_width
                            ):
                                result_accumulator += (
                                    kernel_cache[
                                        inner_channel_idx - channel_block_start,
                                        inner_kernel_idx - kernel_block_start,
                                    ]
                                    * input_cache[
                                        inner_channel_idx - channel_block_start,
                                        abs(position - min_bound),
                                    ]
                                )
                numba.cuda.syncthreads()

        # Write output to output tensor
        if thread_py == 0 and thread_width_idx < out_width:
            output_idx = (
                batch_idx * o_stride0
                + thread_channel_idx * o_stride1
                + thread_width_idx * o_stride2
            )
            out[output_idx] = result_accumulator


# JIT compile the function for CUDA
tensor_conv1d = cuda.jit()(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward_inner(
        out_shape: UserShape,
        in_tensor: Tensor,
        w_tensor: Tensor,
        is_reversed: bool = False,
    ) -> Tensor:
        """Execute a 1D convolution operation (internal helper for forward pass).

        Args:
        ----
        out_shape (UserShape): Specifies the shape of the output tensor to define convolution dimensions.
        in_tensor (Tensor): Input tensor with shape (batch_size, in_channels, width).
        w_tensor (Tensor): Weight tensor with shape (out_channels, in_channels, kernel_width).
        is_reversed (bool, optional):
            - If True, the weight's orientation is reversed during convolution.
            - Calculation: out[b, c_out, w] = in[b, :, w:w-kw:-1] * w_tensor[c_out, :, 0:kw].

        Returns:
        -------
        Tensor: Output tensor with shape (batch_size, out_channels, width).

        """
        batch_size, in_channels, input_width = in_tensor.shape
        out_channels, weight_in_channels, kernel_width = w_tensor.shape
        assert (
            in_channels == weight_in_channels
        ), "Mismatch in input and weight channels."

        # Create an output tensor of the specified shape
        out_tensor = in_tensor.zeros(out_shape)

        # Set CUDA grid and thread block dimensions
        THREADS_PER_BLOCK = 16
        blocks_per_grid = (
            (input_width + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            1,
            out_channels,
        )
        threads_per_block = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        # Launch the CUDA kernel to compute the convolution
        tensor_conv1d[blocks_per_grid, threads_per_block](
            *out_tensor.tuple(),
            out_tensor.size,
            *in_tensor.tuple(),
            *w_tensor.tuple(),
            is_reversed,
        )
        return out_tensor

    @staticmethod
    def forward(ctx: Context, in_tensor: Tensor, w_tensor: Tensor) -> Tensor:
        """Execute the forward pass for the 1D convolution.

        Args:
        ----
        ctx (Context): Stores intermediate values for backpropagation.
        in_tensor (Tensor): Input tensor with shape (batch_size, in_channels, width).
        w_tensor (Tensor): Weight tensor with shape (out_channels, in_channels, kernel_width).

        Returns:
        -------
        Tensor: Output tensor with shape (batch_size, out_channels, width).

        """
        # Store input and weight for use during the backward pass
        ctx.save_for_backward(in_tensor, w_tensor)

        # Compute the forward pass of the 1D convolution
        out_tensor = Conv1dFun.forward_inner(
            (in_tensor.shape[0], w_tensor.shape[0], in_tensor.shape[2]),
            in_tensor,
            w_tensor,
            is_reversed=False,
        )
        return out_tensor

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Execute the backward pass for the 1D convolution.

        Args:
        ----
        ctx (Context): Contains saved forward pass values.
        grad_output (Tensor): Gradient of the output tensor.

        Returns:
        -------
        Tuple[Tensor, Tensor]:
            - Gradient with respect to the input tensor.
            - Gradient with respect to the weight tensor.

        """
        # Retrieve input and weight saved from the forward pass
        in_tensor, w_tensor = ctx.saved_values

        # Calculate gradient with respect to weight
        permuted_input = in_tensor.permute(1, 0, 2)
        permuted_grad_output = grad_output.permute(1, 0, 2)
        grad_w_tensor = Conv1dFun.forward_inner(
            (w_tensor.shape[1], w_tensor.shape[0], w_tensor.shape[2]),
            permuted_input,
            permuted_grad_output,
            is_reversed=False,
        )
        grad_w_tensor = grad_w_tensor.permute(1, 0, 2)

        # Calculate gradient with respect to input
        permuted_w_tensor = w_tensor.permute(1, 0, 2)
        grad_in_tensor = Conv1dFun.forward_inner(
            in_tensor.shape, grad_output, permuted_w_tensor, is_reversed=True
        )

        return grad_in_tensor, grad_w_tensor


# Apply the Conv1dFun class as a function
conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """Compute a 2D convolution on the input tensor.

    This function computes a 2D convolution with the following tensor shapes:
    - Input tensor: (batch, in_channels, height, width)
    - Weight tensor: (out_channels, in_channels, kernel_height, kernel_width)
    - Output tensor: (batch, out_channels, height, width)

    Args:
    ----
    out (Storage): Storage for the output tensor.
    out_shape (Shape): Shape of the output tensor.
    out_strides (Strides): Strides of the output tensor.
    out_size (int): Total number of elements in the output tensor.
    input (Storage): Storage for the input tensor.
    input_shape (Shape): Shape of the input tensor.
    input_strides (Strides): Strides of the input tensor.
    weight (Storage): Storage for the weight tensor.
    weight_shape (Shape): Shape of the weight tensor.
    weight_strides (Strides): Strides of the weight tensor.
    reverse (bool): Whether to reverse the weight orientation.

    Returns:
    -------
        None: The output tensor `out` is updated in place.

    """
    batch_out, out_channels, out_height, out_width = out_shape
    batch_in, in_channels, in_height, in_width = input_shape
    weight_out_channels, weight_in_channels, kernel_height, kernel_width = weight_shape

    BLOCK_DIM = 16
    BLOCK_DIM2 = 32

    # Check if input, weight, and output tensor dimensions match as expected
    assert (
        batch_in == batch_out
        and in_channels == weight_in_channels
        and out_channels == weight_out_channels
    ), "Mismatch in shapes of input, weight, and output tensors."
    assert (
        out_width <= in_width and out_height <= in_height
    ), "Output dimensions cannot exceed input dimensions."

    # Calculate thread positions within the grid and blocks
    thread_width_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    thread_height_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    width_cache_start = cuda.blockIdx.x * cuda.blockDim.x
    height_cache_start = cuda.blockIdx.y * cuda.blockDim.y
    channel_idx = cuda.blockIdx.z
    px = cuda.threadIdx.x
    py = cuda.threadIdx.y

    # Shared memory for weights and input caches
    weight_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    input_cache = cuda.shared.array((BLOCK_DIM2, BLOCK_DIM2), numba.float64)

    # Extract the strides for weight, input, and output tensors
    ws0, ws1, ws2, ws3 = weight_strides
    is0, is1, is2, is3 = input_strides
    os0, os1, os2, os3 = out_strides

    # Set kernel step direction based on the reverse flag
    kernel_direction = -1 if reverse else 1

    for batch_idx in range(batch_in):
        # Calculate the position of the output in the storage
        out_position = (
            batch_idx * os0
            + channel_idx * os1
            + thread_height_idx * os2
            + thread_width_idx * os3
        )
        result_accumulator = 0.0

        for in_channel_idx in range(in_channels):
            for kh_start in range(0, kernel_height, BLOCK_DIM):
                for kw_start in range(0, kernel_width, BLOCK_DIM):
                    kernel_width_idx = kw_start + px
                    kernel_height_idx = kh_start + py

                    # Load weights into the shared memory cache
                    if (
                        kernel_height_idx < kernel_height
                        and kernel_width_idx < kernel_width
                    ):
                        weight_position = (
                            channel_idx * ws0
                            + in_channel_idx * ws1
                            + kernel_height_idx * ws2
                            + kernel_width_idx * ws3
                        )
                        weight_cache[(px, py)] = weight[weight_position]
                    else:
                        weight_cache[(px, py)] = 0.0

                    numba.cuda.syncthreads()

                    # Load input into the shared memory cache
                    for w_cache_offset in range(0, BLOCK_DIM2, BLOCK_DIM):
                        for h_cache_offset in range(0, BLOCK_DIM2, BLOCK_DIM):
                            if reverse:
                                w_cache_pos = (
                                    width_cache_start
                                    - kw_start
                                    - BLOCK_DIM
                                    + 1
                                    + w_cache_offset
                                    + px
                                )
                                h_cache_pos = (
                                    height_cache_start
                                    - kh_start
                                    - BLOCK_DIM
                                    + 1
                                    + h_cache_offset
                                    + py
                                )
                            else:
                                w_cache_pos = (
                                    width_cache_start + kw_start + w_cache_offset + px
                                )
                                h_cache_pos = (
                                    height_cache_start + kh_start + h_cache_offset + py
                                )

                            # Cache the input if the position is within the valid input bounds
                            if (
                                0 <= w_cache_pos < in_width
                                and 0 <= h_cache_pos < in_height
                            ):
                                input_position = (
                                    batch_idx * is0
                                    + in_channel_idx * is1
                                    + h_cache_pos * is2
                                    + w_cache_pos * is3
                                )
                                input_cache[
                                    (w_cache_offset + px, h_cache_offset + py)
                                ] = input[input_position]
                            else:
                                input_cache[
                                    (w_cache_offset + px, h_cache_offset + py)
                                ] = 0.0

                        numba.cuda.syncthreads()

                    # Compute convolution for each valid position in the output
                    if thread_height_idx < out_height and thread_width_idx < out_width:
                        for kh_inner in range(
                            kh_start, min(kernel_height, kh_start + BLOCK_DIM)
                        ):
                            h_current = thread_height_idx + kh_inner * kernel_direction
                            height_min = (
                                height_cache_start - kh_start - BLOCK_DIM + 1
                                if reverse
                                else height_cache_start + kh_start
                            )
                            height_max = height_min + BLOCK_DIM2

                            if not (
                                0 <= h_current < in_height
                                and height_min <= h_current < height_max
                            ):
                                continue

                            for kw_inner in range(
                                kw_start, min(kernel_width, kw_start + BLOCK_DIM)
                            ):
                                w_current = (
                                    thread_width_idx + kw_inner * kernel_direction
                                )
                                width_min = (
                                    width_cache_start - kw_start - BLOCK_DIM + 1
                                    if reverse
                                    else width_cache_start + kw_start
                                )
                                width_max = width_min + BLOCK_DIM2

                                if not (
                                    0 <= w_current < in_width
                                    and width_min <= w_current < width_max
                                ):
                                    continue

                                result_accumulator += (
                                    weight_cache[
                                        (kw_inner - kw_start, kh_inner - kh_start)
                                    ]
                                    * input_cache[
                                        abs(w_current - width_min),
                                        abs(h_current - height_min),
                                    ]
                                )

                    numba.cuda.syncthreads()

        # Store the result in the output storage at the correct position
        if thread_height_idx < out_height and thread_width_idx < out_width:
            out[out_position] = result_accumulator


tensor_conv2d = cuda.jit()(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward_inner(
        out_shape: UserShape,
        in_tensor: Tensor,
        w_tensor: Tensor,
        is_reversed: bool = False,
    ) -> Tensor:
        """Perform a 2D convolution operation, invoked by the forward pass.

        Args:
        ----
        out_shape (UserShape): Shape of the output tensor.
        in_tensor (Tensor): Input tensor with shape (batch_size, in_channels, height, width).
        w_tensor (Tensor): Weight tensor with shape (out_channels, in_channels, kernel_height, kernel_width).
        is_reversed (bool): If True, the convolution is performed in reverse.

        Returns:
        -------
        Tensor: Output tensor with shape (batch_size, out_channels, height, width).

        """
        batch_size, in_channels, height, width = in_tensor.shape
        out_channels, weight_in_channels, kernel_height, kernel_width = w_tensor.shape
        assert (
            in_channels == weight_in_channels
        ), "Mismatch in the number of input and weight channels."

        # Allocate memory for the output tensor
        out_tensor = in_tensor.zeros(out_shape)

        THREADS_PER_BLOCK = 16

        # Configure CUDA grid and block dimensions
        grid_dims = (
            (width + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
            (height + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
            out_channels,
        )
        block_dims = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        # Launch the CUDA kernel to compute the 2D convolution
        tensor_conv2d[grid_dims, block_dims](
            *out_tensor.tuple(),
            out_tensor.size,
            *in_tensor.tuple(),
            *w_tensor.tuple(),
            is_reversed,
        )

        return out_tensor

    @staticmethod
    def forward(ctx: Context, in_tensor: Tensor, w_tensor: Tensor) -> Tensor:
        """Execute the forward pass of the 2D convolution.

        Args:
        ----
        ctx (Context): Stores input and weight for use during the backward pass.
        in_tensor (Tensor): Input tensor with shape (batch_size, in_channels, height, width).
        w_tensor (Tensor): Weight tensor with shape (out_channels, in_channels, kernel_height, kernel_width).

        Returns:
        -------
        Tensor: Output tensor with shape (batch_size, out_channels, height, width).

        """
        ctx.save_for_backward(in_tensor, w_tensor)
        out_shape = (
            in_tensor.shape[0],
            w_tensor.shape[0],
            in_tensor.shape[2],
            in_tensor.shape[3],
        )
        out_tensor = Conv2dFun.forward_inner(
            out_shape, in_tensor, w_tensor, is_reversed=False
        )
        return out_tensor

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass of the 2D convolution.

        Args:
        ----
        ctx (Context): Stores saved input and weight tensors from the forward pass.
        grad_output (Tensor): Gradient of the output tensor.

        Returns:
        -------
        Tuple[Tensor, Tensor]: Gradients of the input tensor and weight tensor.

        """
        # Retrieve input and weight from the context
        in_tensor, w_tensor = ctx.saved_values

        # Calculate gradient with respect to the weight
        permuted_input = in_tensor.permute(1, 0, 2, 3)
        permuted_grad_output = grad_output.permute(1, 0, 2, 3)
        grad_w_shape = (
            w_tensor.shape[1],
            w_tensor.shape[0],
            w_tensor.shape[2],
            w_tensor.shape[3],
        )
        grad_w_tensor = Conv2dFun.forward_inner(
            grad_w_shape, permuted_input, permuted_grad_output, is_reversed=False
        )
        grad_w_tensor = grad_w_tensor.permute(1, 0, 2, 3)

        # Calculate gradient with respect to the input
        permuted_w_tensor = w_tensor.permute(1, 0, 2, 3)
        grad_in_tensor = Conv2dFun.forward_inner(
            in_tensor.shape, grad_output, permuted_w_tensor, is_reversed=True
        )

        return grad_in_tensor, grad_w_tensor


conv2d = Conv2dFun.apply
