import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
import functools


def fused_mlp_kernel(
    x_ref,
    w_gu_ref,
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

    # 1. Compute Up and Gate projections
    hu_sram = jnp.matmul(x_ref[...], w_gu_ref[...], preferred_element_type=jnp.float32)

    # Split the result in SRAM (zero-cost operation)
    h_sram = hu_sram[:, :b_inter]
    u_sram = hu_sram[:, b_inter:]

    # Apply activation -> Shape: [b_seq, b_inter]
    a_tile = jax.nn.gelu(h_sram) * u_sram
    a_tile = a_tile.astype(x_ref.dtype)

    # 2. Compute Down projection
    # wd_ref is fetched as [hidden_size, b_inter]. We transpose it back in VMEM metadata (.T)
    # This ensures the MXU contracts along the contiguous inner dimension.
    wd_sram = wd_ref[...]
    y_current_sram = jnp.matmul(a_tile, wd_sram.T, preferred_element_type=jnp.float32)

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


@functools.partial(jax.jit, static_argnums=(4, 5))
def apply_fused_mlp_sharded(
    x: jax.Array,
    wg: jax.Array,
    wu: jax.Array,
    wd: jax.Array,
    mesh: jax.sharding.Mesh,
    chunk_size: int = 256,
) -> jax.Array:

    # Cast weights to bfloat16 to halve HBM bandwidth overhead
    wg = wg.astype(jnp.bfloat16)
    wu = wu.astype(jnp.bfloat16)
    wd = wd.astype(jnp.bfloat16)

    # OPTIMIZATION 1: Pre-transpose wd outside the kernel (HBM layout optimization)
    # Original wd: [inter, hidden] -> Transposed wd_t: [hidden, inter]
    wd_t = wd.T

    # Note that transposing wd shifts the TP sharding dimension from 0 to 1
    in_specs = (
        P(None, None),  # x
        P(None, "tensor"),  # wg
        P(None, "tensor"),  # wu
        P(None, "tensor"),  # wd_t (sharded along intermediate dim)
    )

    # OPTIMIZATION 2: Output is now Sequence Parallel!
    out_specs = P("tensor", None)

    @functools.partial(
        shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False
    )
    def local_fused_mlp(x_loc, wg_loc, wu_loc, wd_t_loc):
        seq_len, hidden_size = x_loc.shape
        _, local_inter_size = wg_loc.shape

        B_SEQ = 64
        B_INTER = 256

        # Interleave wg and wu block-by-block so the kernel fetches them correctly
        num_blocks = local_inter_size // B_INTER
        wg_reshaped = wg_loc.reshape(hidden_size, num_blocks, B_INTER)
        wu_reshaped = wu_loc.reshape(hidden_size, num_blocks, B_INTER)

        w_gu_loc = jnp.concatenate([wg_reshaped, wu_reshaped], axis=-1)
        w_gu_loc = w_gu_loc.reshape(hidden_size, local_inter_size * 2)

        # Pallas BlockSpecs
        pallas_in_specs = (
            pl.BlockSpec(
                (B_SEQ, hidden_size),
                lambda s_i, i_i: (s_i, 0),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
            pl.BlockSpec(
                (hidden_size, 2 * B_INTER),
                lambda s_i, i_i: (0, i_i),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
            # OPTIMIZATION 1: Fetch transposed blocks of [hidden_size, B_INTER]
            pl.BlockSpec(
                (hidden_size, B_INTER),
                lambda s_i, i_i: (0, i_i),
                pipeline_mode=pl.Buffered(buffer_count=2),
            ),
        )

        pallas_out_specs = pl.BlockSpec((B_SEQ, hidden_size), lambda s_i, i_i: (s_i, 0))

        # The grid now operates on a single sequence chunk at a time
        grid = (chunk_size // B_SEQ, local_inter_size // B_INTER)

        # OPTIMIZATION 3: Chunking via scan to break the dependency chain
        def compute_chunk(carry, x_chunk):
            y_chunk_loc = pl.pallas_call(
                functools.partial(
                    fused_mlp_kernel,
                    b_seq=B_SEQ,
                    b_inter=B_INTER,
                    hidden_size=hidden_size,
                    intermediate_size=local_inter_size,
                ),
                out_shape=jax.ShapeDtypeStruct((chunk_size, hidden_size), x_chunk.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=grid,
                    in_specs=pallas_in_specs,
                    out_specs=pallas_out_specs,
                    scratch_shapes=[pltpu.VMEM((B_SEQ, hidden_size), jnp.float32)],
                ),
                compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
            )(x_chunk, w_gu_loc, wd_t_loc)

            # OPTIMIZATION 4: Overlap compute and comms. XLA will schedule this network
            # scatter to run concurrently with the next loop's Pallas MXU execution.
            y_chunk_scattered = jax.lax.psum_scatter(
                y_chunk_loc, axis_name="tensor", scatter_dimension=0
            )
            return carry, y_chunk_scattered

        # Reshape x_loc into sequence chunks
        num_chunks = seq_len // chunk_size
        x_loc_chunked = x_loc.reshape(num_chunks, chunk_size, hidden_size)

        # Execute the scan loop
        _, y_loc_scattered_chunks = jax.lax.scan(compute_chunk, None, x_loc_chunked)

        # y_loc_scattered_chunks shape: [num_chunks, chunk_size // TP_size, hidden_size]
        # Flatten back into a continuous Sequence Parallel array
        y_loc_final = y_loc_scattered_chunks.reshape(-1, hidden_size)

        y_loc_gathered = jax.lax.all_gather(y_loc_final, axis_name="tensor", tiled=True)

        return y_loc_gathered

    return local_fused_mlp(x, wg, wu, wd_t)


def apply_fused_mlp_with_padding(
    x: jax.Array,
    wg: jax.Array,
    wu: jax.Array,
    wd: jax.Array,
    mesh: jax.sharding.Mesh,
    chunk_size: int = 256,
) -> jax.Array:
    """Pads the input tensor to be a multiple of chunk_size in sequence length."""
    seq_len, hidden_size = x.shape
    rem = seq_len % chunk_size

    if rem == 0:
        return apply_fused_mlp_sharded(x, wg, wu, wd, mesh, chunk_size)

    pad_len = chunk_size - rem
    x_padded = jnp.pad(x, ((0, pad_len), (0, 0)), mode="constant")

    # Global evaluation yields a Sequence Parallel array
    out_padded = apply_fused_mlp_sharded(x_padded, wg, wu, wd, mesh, chunk_size)

    # JAX correctly routes this global slice across the tensor parallel mesh
    return out_padded[:seq_len, :]
