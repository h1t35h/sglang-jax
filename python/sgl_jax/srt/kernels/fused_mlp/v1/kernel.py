import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
import functools


def fused_mlp_kernel(
    x_ref,
    wg_ref,
    wu_ref,
    wd_ref,
    y_ref,  # Memory references
    y_scratch, # Scratchpad for accumulation
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    b_seq: int,
    b_hidden: int,
    b_inter: int,
    b_hidden_in: int,
):
    seq_idx = pl.program_id(0)
    hidden_out_idx = pl.program_id(1)
    inter_idx = pl.program_id(2)

    # Initialize scratchpad to zero at the beginning of reduction
    @pl.when(inter_idx == 0)
    def _init_y():
        y_scratch[...] = jnp.zeros((b_seq, b_hidden), dtype=jnp.float32)

    # Read whole blocks into values to avoid pl.dslice in loops
    x_val = x_ref[...]
    wg_val = wg_ref[...]
    wu_val = wu_ref[...]
    wd_val = wd_ref[...]

    # Accumulators for Gate (H) and Up (U) projections
    h_acc = jnp.zeros((b_seq, b_inter), dtype=jnp.float32)
    u_acc = jnp.zeros((b_seq, b_inter), dtype=jnp.float32)

    # Use a Python for loop to unroll the loop.
    # This makes hin_idx a static integer, allowing standard Python slicing!
    num_in_blocks = hidden_size // b_hidden_in
    for hin_idx in range(num_in_blocks):
        start = hin_idx * b_hidden_in
        end = (hin_idx + 1) * b_hidden_in
        
        x_tile = x_val[:, start:end]
        wg_tile = wg_val[start:end, :]
        wu_tile = wu_val[start:end, :]
        
        h_acc += pl.dot(x_tile, wg_tile)
        u_acc += pl.dot(x_tile, wu_tile)

    # Apply activation
    a_tile = jax.nn.gelu(h_acc) * u_acc
    a_tile = a_tile.astype(x_ref.dtype)

    # down projection
    # wd_val is already read and has shape (b_inter, b_hidden)
    y_contribution = pl.dot(a_tile, wd_val)

    # Read current accumulator value
    acc = y_scratch[...]
    acc = acc + y_contribution

    is_last = inter_idx == (intermediate_size // b_inter - 1)

    @pl.when(is_last)
    def _write():
        y_ref[...] = acc.astype(y_ref.dtype)

    @pl.when(~is_last)
    def _save():
        y_scratch[...] = acc


@functools.partial(jax.jit, static_argnums=(4,))
def apply_fused_mlp_sharded(
    x: jax.Array, wg: jax.Array, wu: jax.Array, wd: jax.Array, mesh: jax.sharding.Mesh
) -> jax.Array:

    in_specs = (
        P(None, None),  # x
        P(None, "tensor"),  # wg_q
        P(None, "tensor"),  # wu_q
        P("tensor", None),  # wd_q
    )

    out_specs = P(None, None)

    @functools.partial(
        shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False
    )
    def local_fused_mlp(x_loc, wg_loc, wu_loc, wd_loc):
        seq_len, hidden_size = x_loc.shape
        _, local_inter_size = wg_loc.shape

        B_SEQ = 128
        B_HIDDEN = 128
        B_INTER = 128  
        B_HIDDEN_IN = 128

        grid = (seq_len // B_SEQ, hidden_size // B_HIDDEN, local_inter_size // B_INTER)

        in_specs = (
            pl.BlockSpec((B_SEQ, hidden_size), lambda s_i, h_i, i_i: (s_i * B_SEQ, 0)),
            pl.BlockSpec((hidden_size, B_INTER), lambda s_i, h_i, i_i: (0, i_i * B_INTER)),
            pl.BlockSpec((hidden_size, B_INTER), lambda s_i, h_i, i_i: (0, i_i * B_INTER)),
            pl.BlockSpec((B_INTER, B_HIDDEN), lambda s_i, h_i, i_i: (i_i * B_INTER, h_i * B_HIDDEN)),
        )
        out_specs = pl.BlockSpec((B_SEQ, B_HIDDEN), lambda s_i, h_i, i_i: (s_i * B_SEQ, h_i * B_HIDDEN))

        # Execute Pallas on purely local data
        y_loc = pl.pallas_call(
            functools.partial(
                fused_mlp_kernel,
                seq_len=seq_len,
                hidden_size=hidden_size,
                intermediate_size=local_inter_size,
                b_seq=B_SEQ,
                b_hidden=B_HIDDEN,
                b_inter=B_INTER,
                b_hidden_in=B_HIDDEN_IN,
            ),
            out_shape=jax.ShapeDtypeStruct((seq_len, hidden_size), x_loc.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=[pltpu.VMEM((B_SEQ, B_HIDDEN), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary")
            ),
        )(x_loc, wg_loc, wu_loc, wd_loc)

        return y_loc

    return local_fused_mlp(x, wg, wu, wd)


def apply_fused_mlp_with_padding(
    x: jax.Array, wg: jax.Array, wu: jax.Array, wd: jax.Array, mesh: jax.sharding.Mesh
) -> jax.Array:
    """Pads the input tensor to be a multiple of 128 in sequence length."""
    B_SEQ = 128
    seq_len, hidden_size = x.shape
    rem = seq_len % B_SEQ
    if rem == 0:
        return apply_fused_mlp_sharded(x, wg, wu, wd, mesh)
    
    pad_len = B_SEQ - rem
    x_padded = jnp.pad(x, ((0, pad_len), (0, 0)), mode="constant")
    out_padded = apply_fused_mlp_sharded(x_padded, wg, wu, wd, mesh)
    return out_padded[:seq_len, :]
