# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel."""

from sgl_jax.srt.mem_cache import chunk_cache
import jax
import jax.numpy as jnp
from jax import lax

from sgl_jax.srt.kernels.quantized_matmul.blockwise_utils import (
    get_blockwise_kernel,
    get_safe_blockwise_tuned_value,
    should_use_blockwise_kernel,
)
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor_simple


def xla_quantized_matmul_local(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quantize_activation: bool = True,
    reduce_axis: str | None = None,
    compute_dtype: jnp.dtype | None = None,
    weight_block_size: tuple[int, int] | None = None,
    activation_quant_dtype: jnp.dtype | None = None,
    num_microbatches: int = 1,
) -> jax.Array:
    """
    Local quantized matmul for use inside shard_map.

    All computation (quantize, matmul, dequantize) happens locally on each device.
    If reduce_axis is provided, uses psum to combine partial sums across devices.

    Args:
        x: Activation tensor [batch, n_input_features] (local slice)
        w_q: Quantized weight tensor [n_output_features, n_input_features] (local slice)
        w_scale: Weight quantization scale.  Per-channel: ``[n_output_features]``.
            Block-wise (pre-expanded): ``[in_blocks, 1, n_output_features]``.
        quantize_activation: Whether to quantize activations
        reduce_axis: Axis name for psum reduction (e.g., "tensor"). None skips reduction.
        weight_block_size: ``(block_n, block_k)`` for block-wise quantization.
        activation_quant_dtype: Dtype for activation quantization.

    Returns:
        Output of the quantized matmul.
    Supports both per-channel and block-wise weight quantization.
    """
    out_dtype = x.dtype
    compute_dtype = jnp.float32 if compute_dtype is None else compute_dtype
    act_quant_dtype = w_q.dtype if activation_quant_dtype is None else activation_quant_dtype

    # w_scale.ndim == 3 implies pre-expanded block-wise quantization
    # (scale was expanded from [out_blocks, in_blocks] to [in_blocks, 1, n_out]
    #  at init time via expand_block_scale).
    is_block_quant = w_scale.ndim == 3
    batch_size = x.shape[0]

    if batch_size % num_microbatches != 0:
        num_microbatches = 1

    chunk_size = batch_size // num_microbatches

    if is_block_quant:
        # === Block Quantization Path ===
        out_dim, in_dim = w_q.shape
        in_blocks = w_scale.shape[0]
        block_size_in = in_dim // in_blocks

        block_size_out = (
            int(weight_block_size[0]) if weight_block_size is not None else block_size_in
        )

        blockwise_kernel = get_blockwise_kernel()
        if blockwise_kernel is None:
            raise RuntimeError(
                "Block-wise quantized matmul requires the blockwise kernel, "
                "but it failed to load. Please check your installation."
            )

        # w_scale is already in kernel-ready layout [in_blocks, 1, n_out].
        x_q_dtype = act_quant_dtype if quantize_activation else x.dtype

        tuned_value = get_safe_blockwise_tuned_value(
            n_batch=int(chunk_size),
            n_out=int(out_dim),
            n_in=int(in_dim),
            x_q_dtype=x_q_dtype,
            w_q_dtype=w_q.dtype,
            block_size_in=block_size_in,
        )

    def compute_single(x_chunk):
        # 2. Local Matmul (Compute)
        if is_block_quant:
            out_chunk = blockwise_kernel(
                x=x_chunk,
                w_q=w_q,
                w_scale=w_scale,
                block_size=block_size_in,
                x_q_dtype=x_q_dtype,
                tuned_value=tuned_value,
            )
        else:
            if quantize_activation:
                x_q, x_scale = quantize_tensor_simple(x_chunk, act_quant_dtype, dim=-1)
                out_chunk = lax.dot_general(
                    x_q,
                    w_q,
                    dimension_numbers=(((1,), (1,)), ((), ())),
                    preferred_element_type=compute_dtype,
                )
                out_chunk = (
                    out_chunk.astype(compute_dtype)
                    * x_scale.astype(compute_dtype)
                    * jnp.expand_dims(w_scale, 0).astype(compute_dtype)
                )
            else:
                out_chunk = lax.dot_general(
                    x_chunk,
                    w_q,
                    dimension_numbers=(((1,), (1,)), ((), ())),
                    preferred_element_type=compute_dtype,
                )
                out_chunk = out_chunk.astype(compute_dtype) * jnp.expand_dims(w_scale, 0).astype(
                    compute_dtype
                )
        out_chunk = out_chunk.astype(out_dtype)
        # 3. Overlap Comms
        if reduce_axis is not None:
            out_chunk = lax.psum(out_chunk, axis_name=reduce_axis)

        return out_chunk

    if num_microbatches > 1:
        x_reshaped = x.reshape(num_microbatches, chunk_size, x.shape[-1])
        out_chunks = jax.vmap(compute_single)(x_reshaped)
        # Flatten out_chunks
        out = out_chunks.reshape(batch_size, -1)
    else:
        out = compute_single(x)

    return out
