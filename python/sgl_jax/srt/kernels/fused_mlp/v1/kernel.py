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

    y_acc = jnp.zeros((b_seq, b_hidden), dtype=jnp.float32)

    # Outer loop over the intermediate dimension
    def inter_loop_body(inter_idx, y_acc_val):
        # Accumulators for Gate (H) and Up (U) projections
        h_acc = jnp.zeros((b_seq, b_inter), dtype=jnp.float32)
        u_acc = jnp.zeros((b_seq, b_inter), dtype=jnp.float32)

        # Inner loop over hidden_size to compute H and U
        def hidden_in_loop_body(hin_idx, accs):
            h_acc_val, u_acc_val = accs
            # Load from HBM to VMEM
            x_tile = x_ref[
                pl.dslice(seq_idx * b_seq, b_seq), pl.dslice(hin_idx * b_hidden_in, b_hidden_in)
            ]
            wg_tile = wg_ref[
                pl.dslice(hin_idx * b_hidden_in, b_hidden_in),
                pl.dslice(inter_idx * b_inter, b_inter),
            ]
            wu_tile = wu_ref[
                pl.dslice(hin_idx * b_hidden_in, b_hidden_in),
                pl.dslice(inter_idx * b_inter, b_inter),
            ]
            h_acc_val += pl.dot(x_tile, wg_tile)
            u_acc_val += pl.dot(x_tile, wu_tile)
            return h_acc_val, u_acc_val

        h_acc, u_acc = jax.lax.fori_loop(
            0, hidden_size // b_hidden_in, hidden_in_loop_body, (h_acc, u_acc)
        )

        # Apply activation
        a_tile = jax.nn.gelu(h_acc) * u_acc
        a_tile = a_tile.astype(x_ref.dtype)

        # down projection
        wd_tile = wd_ref[
            pl.dslice(inter_idx * b_inter, b_inter), pl.dslice(hidden_out_idx * b_hidden, b_hidden)
        ]

        y_acc_val += pl.dot(a_tile, wd_tile)
        return y_acc_val

    y_acc = jax.lax.fori_loop(0, intermediate_size // b_inter, inter_loop_body, y_acc)

    y_ref[pl.dslice(seq_idx * b_seq, b_seq), pl.dslice(hidden_out_idx * b_hidden, b_hidden)] = (
        y_acc.astype(y_ref.dtype)
    )


@functools.partial(jax.jit, static_argnums=(4,))
def apply_fused_mlp_sharded(
    x: jax.Array, wg: jax.Array, wu: jax.Array, wd: jax.Array, mesh: jax.sharding.Mesh
) -> jax.Array:

    # 1. Define the sharding layout for the inputs based on QuantizedLinear
    # x is fully replicated (unsharded)
    # Gate/Up weights are column-parallel, so the intermediate dimension (axis 1) is sharded
    # Down weights are row-parallel, so the intermediate dimension (axis 0) is sharded
    in_specs = (
        P(None, None),  # x
        P(None, "tensor"),  # wg_q
        P(None, "tensor"),  # wu_q
        P("tensor", None),  # wd_q
    )

    # The output of the local matmul will be replicated
    out_specs = P(None, None)

    # 2. Wrap the Pallas execution in a shard_map
    @functools.partial(
        shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False
    )
    def local_fused_mlp(x_loc, wg_loc, wu_loc, wd_loc):
        seq_len, hidden_size = x_loc.shape

        # CRITICAL: We now use the LOCAL intermediate size (e.g., 32768 / 8 devices = 4096)
        _, local_inter_size = wg_loc.shape

        B_SEQ = 128
        B_HIDDEN = 128
        B_INTER = 128  # Make sure local_inter_size is cleanly divisible by this!
        B_HIDDEN_IN = 128

        grid = (seq_len // B_SEQ, hidden_size // B_HIDDEN)

        in_specs = (
            pl.BlockSpec(lambda seq_idx, hidden_idx: (seq_idx * B_SEQ, 0), (B_SEQ, hidden_size)),
            pl.BlockSpec(lambda seq_idx, hidden_idx: (0, 0), (hidden_size, local_inter_size)),
            pl.BlockSpec(lambda seq_idx, hidden_idx: (0, 0), (hidden_size, local_inter_size)),
            pl.BlockSpec(lambda seq_idx, hidden_idx: (0, hidden_idx * B_HIDDEN), (local_inter_size, B_HIDDEN)),
        )
        out_specs = pl.BlockSpec(lambda seq_idx, hidden_idx: (seq_idx * B_SEQ, hidden_idx * B_HIDDEN), (B_SEQ, B_HIDDEN))

        # Execute Pallas on purely local data
        y_loc = pl.pallas_call(
            functools.partial(
                fused_mlp_kernel,
                seq_len=seq_len,
                hidden_size=hidden_size,
                intermediate_size=local_inter_size,  # Pass local size to the kernel loop
                b_seq=B_SEQ,
                b_hidden=B_HIDDEN,
                b_inter=B_INTER,
                b_hidden_in=B_HIDDEN_IN,
            ),
            out_shape=jax.ShapeDtypeStruct(x_loc.shape, x_loc.dtype),
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary")),
        )(x_loc, wg_loc, wu_loc, wd_loc)

        # 3. All-Reduce: Sum the partial results across the TP devices
        y_global = jax.lax.psum(y_loc, axis_name="tensor")
        return y_global

    # Execute the mapped function
    return local_fused_mlp(x, wg, wu, wd)


def apply_fused_mlp_with_padding(
    x: jax.Array,
    wg: jax.Array,
    wu: jax.Array,
    wd: jax.Array,
    mesh: jax.sharding.Mesh,
) -> jax.Array:
    """Wraps the Pallas MLP kernel to handle arbitrary sequence lengths via padding."""

    original_seq_len, hidden_size = x.shape

    B_SEQ = 128
    pad_amount = (B_SEQ - (original_seq_len % B_SEQ)) % B_SEQ

    if pad_amount > 0:
        x_padded = jnp.pad(x, ((0, pad_amount), (0, 0)), mode="constant", constant_values=0)
    else:
        x_padded = x

    # Pass the mesh down into the sharded execution
    out_padded = apply_fused_mlp_sharded(x_padded, wg, wu, wd, mesh)

    if pad_amount > 0:
        out_real = out_padded[:original_seq_len, :]
    else:
        out_real = out_padded

    return out_real
