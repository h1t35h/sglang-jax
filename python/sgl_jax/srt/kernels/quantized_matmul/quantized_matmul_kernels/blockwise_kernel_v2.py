import jax
import jax.numpy as jnp
from typing import Sequence
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import util
from .tuned_block_sizes import TunedValue, get_device_vmem_limit, get_tuned_block_sizes
from .util import get_kernel_name, next_multiple, unfold_args

quantize_tensor = util.quantize_tensor
MXU_SIZE = 256

# TODO(hitesy): check with original author to move to utils
def pad_axis_to_multiple(tensor, axis, block_size):
    """
    Pads a specific axis of an n-dimensional JAX array to the next multiple of a block size.
    """
    orig_size = tensor.shape[axis]
    padded_size = next_multiple(orig_size, block_size)
    pad_amount = padded_size - orig_size
    
    # If no padding is needed, return the tensor as-is to save computation
    if pad_amount == 0:
        return tensor
        
    # Generate the pad_width tuple dynamically based on the tensor's rank
    # E.g., for a 3D tensor, starts as [(0, 0), (0, 0), (0, 0)]
    pad_width = [(0, 0)] * tensor.ndim
    
    # Apply the padding only to the target axis
    pad_width[axis] = (0, pad_amount)
    
    # jnp.pad requires a tuple of tuples
    return jnp.pad(tensor, tuple(pad_width))

@jax.jit(static_argnames=("block_size", "x_q_dtype", "tuned_value"))
def blockwise_kernel_pallas(
  x: jax.Array,
  w_q: jax.Array,
  w_scale: jax.Array,
  mesh: jax.sharding.Mesh | None = None,
  axis_name: str | None = None,
  block_size: int | None = None,
  x_q_dtype: jnp.dtype | None = None,
  *,
  tuned_value: TunedValue | None = None,
) -> jax.Array:
  """
  Custom Kernel for Quantized matmul with asynchronous All-Reduce overlap.
  """
  
  # Initializations
  if block_size is None:
    raise ValueError("Block size was not specified.")
  
  if x_q_dtype is None:
    x_q_dtype = x.dtype # default quantized type to input
  quantize_activation = x_q_dtype != x.dtype

  orig_n_batch, orig_n_in = x.shape
  orig_n_out, *_ = w_q.shape

  if tuned_value is None:
    # TODO(hitesy): The defaults may not be very sane tune based on performance testin
    tuned_value = get_tuned_block_sizes(
      n_batch=orig_n_batch,
      n_out=orig_n_out,
      n_in=orig_n_in,
      x_q_dtype=jnp.dtype(x_q_dtype).name,
      w_q_dtype=jnp.dtype(w_q.dtype).name,
    )
  batch_block_size = tuned_value.batch_block_size
  out_block_size = tuned_value.out_block_size
  in_block_size = tuned_value.in_block_size
  n_lane_multiplier = tuned_value.n_lane_multiplier
  block_size = tuned_value.in_block_size if block_size == orig_n_in else block_size

  # Pad inputs (we need multiple of block_size)
  padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
  padded_n_out = next_multiple(orig_n_out, out_block_size)
  padded_n_in = next_multiple(orig_n_in, in_block_size)

  # batch
  x = pad_axis_to_multiple(x, axis=0, block_size=batch_block_size)
  # out
  w_q = pad_axis_to_multiple(w_q, axis=0, block_size=out_block_size)
  w_scale = pad_axis_to_multiple(w_scale, axis=2, block_size=out_block_size)
  # in
  x = pad_axis_to_multiple(x, axis=1, block_size=in_block_size)
  w_q = pad_axis_to_multiple(w_q, axis=1, block_size=in_block_size)

  if w_scale.dtype != jnp.float32:
    w_scale = w_scale.astype(jnp.float32)
  
  n_batch = padded_n_batch // batch_block_size
  n_out = padded_n_out // out_block_size
  n_in = padded_n_in // in_block_size

  save_acc = n_in > 1
  save_x_q = quantize_activation and n_in == 1 and n_out > 1
  
  acc_dtype = jnp.bfloat16
  if quantize_activation and jnp.issubdtype(w_q.dtype, jnp.integer):
    acc_dtype = jnp.int32
  
  vmem_limit_bytes = util.get_vmem_limit(
        n_batch=n_batch, n_out=n_out, n_in=n_in,
        batch_block_size=batch_block_size, 
        out_block_size=out_block_size,
        in_block_size=in_block_size, 
        x_dtype=x.dtype, x_q_dtype=x_q_dtype, w_q_dtype=w_q.dtype, 
        scale_dtype=jnp.float32, out_dtype=x.dtype,
        acc_dtype=acc_dtype, save_acc=save_acc, save_x_q=save_x_q,
        upper_limit_bytes=get_device_vmem_limit(),
    )

  steps_k = in_block_size // block_size
  compute_tile_n = MXU_SIZE * n_lane_multiplier
  steps_n = out_block_size // compute_tile_n

  # Target neighbor along the axis_name dynamically
  if mesh is not None and axis_name is not None:
    # Get current and size for this axis dynamically in SPMD
    axis_rank = jax.lax.axis_index(axis_name)
    axis_size = jax.lax.axis_size(axis_name)
    
    neighbor_axis_rank = (axis_rank + 1) % axis_size
    
    local_device_idx = axis_rank
    neighbor_device_idx = neighbor_axis_rank
  else:
    local_device_idx = None
    neighbor_device_idx = None


  # === Custom Reduce Kernel ===
  def custom_reduce_kernel(lhs_ref, rhs_ref, w_scales_ref, out_ref, acc_scratch, remote_scratch_ref, send_sem_ref, recv_sem_ref):
    
    pid_k = pl.program_id(2)
    buf_idx = pid_k % 2
    is_first_step = pid_k == 0
    is_last_step = pid_k == (n_in - 1)

    # Double buffering sequence wait: 
    # To prevent overwriting what program pid_k = i was still pushing, 
    # we must ensure its read is done. So pid_k = i + 2 waits for sending read completed.
    if pid_k >= 2:
      pl.semaphore_wait(send_sem_ref[buf_idx], 1)

    def accum(is_first_step, is_last_step):
      accumulators = [None] * steps_n

      for i in range(steps_k):
        k_start, k_end = i * block_size, (i+1) * block_size
        if quantize_activation:
          lhs_sub = lhs_ref[:, k_start:k_end].astype(jnp.float32)
          lhs_q, lhs_scale = util.quantize_block(lhs_sub, 1, x_q_dtype)
          lhs_scale = lhs_scale.astype(acc_dtype)
        else:
          lhs_q = lhs_ref[:, k_start:k_end]
          lhs_scale = None
        
        rhs_q_full = rhs_ref[:, k_start:k_end]
        rhs_scale_full = w_scales_ref[i, :, :].astype(acc_dtype)

        for j in range(steps_n):
          n_start, n_end = j * compute_tile_n, (j + 1) * compute_tile_n
          rhs_q_slice = rhs_q_full[n_start: n_end, :]
          rhs_scale_slice = rhs_scale_full[: , n_start:n_end]

          preferred_element_type = jnp.int32 if jnp.issubdtype(x_q_dtype, jnp.integer) else jnp.float32

          dot_res = jax.lax.dot_general(
            lhs_q, rhs_q_slice,
            (((1,), (1,)), ((), ())),
            preferred_element_type=preferred_element_type,
          )

          res = dot_res.astype(acc_dtype)
          if lhs_scale is not None:
            res = res * lhs_scale
          res = res * rhs_scale_slice
          if i == 0:
            accumulators[j] = res
          else:
            accumulators[j] += res
      
      acc_block = jnp.concatenate(accumulators, axis=1)

      if not is_first_step:
        # Read from OTHER buffer (double buffering!)
        acc_block += acc_scratch[(buf_idx + 1) % 2, ...]

      # Push partial sum to neighbor and Wait for neighbor's partial sum
      if neighbor_device_idx is not None and local_device_idx is not None:
        def get_mesh_device_id(peer_rank):
          return (0, peer_rank)
          
        if not is_last_step:
          # We copy local scratch directly to remote VMEM on neighbor TPU!
          # DMA is configured to signal send_sem when its read is done,
          # and recv_sem (on target device) when its write is done.
          pltpu.make_async_remote_copy(
            src_ref=acc_scratch[buf_idx],
            dst_ref=remote_scratch_ref,
            send_sem=send_sem_ref[buf_idx],
            recv_sem=recv_sem_ref,
            device_id=get_mesh_device_id(neighbor_device_idx),
            device_id_type=pltpu.DeviceIdType.MESH,
          ).start()
        
        if is_last_step:
          # Wait for neighbor's push signal to unblock!
          pltpu.semaphore_wait(recv_sem_ref, 1)
          acc_block += remote_scratch_ref[...]
      
      if is_last_step:
        out_ref[...] = acc_block.astype(out_ref.dtype)
      else:
        acc_scratch[buf_idx] = acc_block
      
    unfold_args((is_first_step, is_last_step), (), accum)

  kernel_pallas = pl.pallas_call(
        custom_reduce_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((batch_block_size, in_block_size), lambda b, o, i: (b, i), memory_space=pltpu.VMEM),  # x
                pl.BlockSpec((out_block_size, in_block_size), lambda b, o, i: (o, i), memory_space=pltpu.VMEM),  # w_q
                pl.BlockSpec((steps_k, 1, out_block_size), lambda _, o, i: (i * steps_k, 0, o), memory_space=pltpu.VMEM), # w_scale
            ],  
            out_specs=pl.BlockSpec((batch_block_size, out_block_size), lambda b, o, i: (b, o)),
            scratch_shapes=[
                pltpu.VMEM((2, batch_block_size, out_block_size), acc_dtype), # local scratch (arithmetic, double buffered) 
                
                # ICI Scratch buffer for incoming data from neighbor
                pl.BlockSpec(
                    (batch_block_size, out_block_size),
                    lambda b, o, i: (b, o),
                    memory_space=pltpu.VMEM,
                    # No device_index! This was the static bug. We make it local copy space.
                ),
            ],
            # Semaphores to lock ICI race conditions and double buffering sequence
            sem_shapes=[
                pltpu.VMEM((2,), jnp.int32), # send_sems [2]
                pltpu.VMEM((), jnp.int32), # recv_sem
            ],
            grid=(n_batch, n_out, n_in),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )
  
  util.validate_inputs(
        x=x, w_q=w_q, w_scale=w_scale, x_abs_max=None, x_q_dtype=x_q_dtype,
        batch_block_size=batch_block_size, out_block_size=out_block_size, in_block_size=in_block_size,
    )
  
  kernel_name = "hitesy_" + get_kernel_name(tuned_value)
  with jax.named_scope(kernel_name):
    out = kernel_pallas(x, w_q, w_scale)
  
  return out[:orig_n_batch, :orig_n_out]
