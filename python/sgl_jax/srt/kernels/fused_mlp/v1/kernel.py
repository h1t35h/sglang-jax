import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
import functools


def fused_mlp_kernel(
    x_ref,
    w_gu_ref,  # Combined wg and wu reference
    wd_ref,
    y_ref,
    y_scratch,
    *,
    b_seq: int,
    b_inter: int,
    hidden_size: int,
    intermediate_size: int,
):
    s_i = pl.program_id(0)
    i_i = pl.program_id(1)

    # Initialize accumulator scratchpad on the first intermediate block
    @pl.when(i_i == 0)
    def _init():
        y_scratch[...] = jnp.zeros((b_seq, hidden_size), dtype=jnp.float32)

    hu_sram = jnp.matmul(x_ref[...], w_gu_ref[...], preferred_element_type=jnp.float32)

    # Split the result in SRAM (zero-cost operation)
    h_sram = hu_sram[:, :b_inter]
    u_sram = hu_sram[:, b_inter:]

    # Apply activation -> Shape: [b_seq, b_inter]
    a_tile = jax.nn.gelu(h_sram) * u_sram
    a_tile = a_tile.astype(x_ref.dtype)

    y_current_sram = jnp.matmul(a_tile, wd_ref[...], preferred_element_type=jnp.float32)

    # 3. Accumulate
    acc = y_scratch[...]
    acc = acc + y_current_sram

    is_last = i_i == (intermediate_size // b_inter - 1)

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

    # Cast weights to bfloat16 to half HBM bandwidth overhead
    wg = wg.astype(jnp.bfloat16)
    wu = wu.astype(jnp.bfloat16)
    wd = wd.astype(jnp.bfloat16)

    in_specs = (
        P(None, None),  # x
        P(None, "tensor"),  # wg
        P(None, "tensor"),  # wu
        P("tensor", None),  # wd
    )

    out_specs = P(None, None)

    @functools.partial(
        shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False
    )
    def local_fused_mlp(x_loc, wg_loc, wu_loc, wd_loc):
        seq_len, hidden_size = x_loc.shape
        _, local_inter_size = wg_loc.shape

        B_SEQ = 64
        B_INTER = 256

        # Interleave wg and wu block-by-block so the kernel fetches them correctly
        num_blocks = local_inter_size // B_INTER
        wg_reshaped = wg_loc.reshape(hidden_size, num_blocks, B_INTER)
        wu_reshaped = wu_loc.reshape(hidden_size, num_blocks, B_INTER)

        # Concat along the block dimension, then flatten back out
        w_gu_loc = jnp.concatenate([wg_reshaped, wu_reshaped], axis=-1)
        w_gu_loc = w_gu_loc.reshape(hidden_size, local_inter_size * 2)

        grid = (seq_len // B_SEQ, local_inter_size // B_INTER)

        pallas_in_specs = (
            pl.BlockSpec(
                (B_SEQ, hidden_size),
                lambda s_i, i_i: (s_i, 0),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
            # Fetch 2 * B_INTER because it contains both wg and wu blocks
            pl.BlockSpec(
                (hidden_size, 2 * B_INTER),
                lambda s_i, i_i: (0, i_i),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
            pl.BlockSpec(
                (B_INTER, hidden_size),
                lambda s_i, i_i: (i_i, 0),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
        )

        pallas_out_specs = pl.BlockSpec((B_SEQ, hidden_size), lambda s_i, i_i: (s_i, 0))

        y_loc = pl.pallas_call(
            functools.partial(
                fused_mlp_kernel,
                b_seq=B_SEQ,
                b_inter=B_INTER,
                hidden_size=hidden_size,
                intermediate_size=local_inter_size,
            ),
            out_shape=jax.ShapeDtypeStruct((seq_len, hidden_size), x_loc.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=pallas_in_specs,
                out_specs=pallas_out_specs,
                scratch_shapes=[pltpu.VMEM((B_SEQ, hidden_size), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        )(x_loc, w_gu_loc, wd_loc)

        return jax.lax.psum(y_loc, axis_name="tensor")

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
