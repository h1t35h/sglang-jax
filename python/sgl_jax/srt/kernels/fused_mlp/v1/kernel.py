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

    # Accumulators for Gate (H) and Up (U) projections
    h_acc = jnp.zeros((b_seq, b_inter), dtype=jnp.float32)
    u_acc = jnp.zeros((b_seq, b_inter), dtype=jnp.float32)

    # Inner loop over hidden_size to compute H and U
    def hidden_in_loop_body(hin_idx, accs):
        h_acc_val, u_acc_val = accs
        # x_ref is sliced by B_SEQ in in_specs, so shape is (b_seq, hidden_size)
        x_tile = x_ref[
            pl.dslice(0, b_seq), pl.dslice(hin_idx * b_hidden_in, b_hidden_in)
        ]
        # wg_ref is sliced by B_INTER in in_specs, so shape is (hidden_size, b_inter)
        wg_tile = wg_ref[
            pl.dslice(hin_idx * b_hidden_in, b_hidden_in), pl.dslice(0, b_inter)
        ]
        wu_tile = wu_ref[
            pl.dslice(hin_idx * b_hidden_in, b_hidden_in), pl.dslice(0, b_inter)
        ]
        
        h_acc_val += pl.dot(x_tile[...], wg_tile[...])
        u_acc_val += pl.dot(x_tile[...], wu_tile[...])
        return h_acc_val, u_acc_val

    h_acc, u_acc = jax.lax.fori_loop(
        0, hidden_size // b_hidden_in, hidden_in_loop_body, (h_acc, u_acc)
    )

    # Apply activation
    a_tile = jax.nn.gelu(h_acc) * u_acc
    a_tile = a_tile.astype(x_ref.dtype)

    # down projection
    # wd_ref is sliced by B_INTER and B_HIDDEN in in_specs, so shape is (b_inter, b_hidden)
    wd_tile = wd_ref[pl.dslice(0, b_inter), pl.dslice(0, b_hidden)]

    y_contribution = pl.dot(a_tile, wd_tile[...])

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
