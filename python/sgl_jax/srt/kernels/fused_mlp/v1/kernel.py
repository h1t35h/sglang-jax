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
    y_scratch,  # Scratchpad for accumulation
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

    @pl.when(inter_idx == 0)
    def _():
        y_scratch[...] = jnp.zeros((b_seq, b_hidden), dtype=jnp.float32)

    h_sram = jnp.matmul(x_ref[...], wg_ref[...], preferred_element_type=jnp.float32)
    u_sram = jnp.matmul(x_ref[...], wu_ref[...], preferred_element_type=jnp.float32)

    # Apply activation
    a_tile = jax.nn.gelu(h_sram) * u_sram
    a_tile = a_tile.astype(x_ref.dtype)

    # down projection
    # Replace line 42:
    y_current_sram = jnp.matmul(a_tile, wd_ref[...], preferred_element_type=jnp.float32)

    # Read current accumulator value
    acc = y_scratch[...]
    acc = acc + y_current_sram

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
            pl.BlockSpec((B_SEQ, hidden_size), lambda s_i, h_i, i_i: (s_i, 0)),
            pl.BlockSpec((hidden_size, B_INTER), lambda s_i, h_i, i_i: (0, i_i)),
            pl.BlockSpec((hidden_size, B_INTER), lambda s_i, h_i, i_i: (0, i_i)),
            pl.BlockSpec((B_INTER, B_HIDDEN), lambda s_i, h_i, i_i: (i_i, h_i)),
        )
        out_specs = pl.BlockSpec((B_SEQ, B_HIDDEN), lambda s_i, h_i, i_i: (s_i, h_i))

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
                dimension_semantics=("parallel", "arbitrary", "arbitrary")
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
