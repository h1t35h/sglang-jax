from flax.nnx import Intermediate
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools


def fused_quantized_mlp_kernel(
    x_ref,
    wg_q_ref,
    wg_scale_ref,
    wu_q_ref,
    wu_scale_ref,
    wd_q_ref,
    wd_scale_ref,
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
            x_tile = pl.load(
                x_ref,
                (pl.dslice(seq_idx * b_seq, b_seq), pl.dslice(hin_idx * b_hidden_in, b_hidden_in)),
            )
            wg_q_tile = pl.load(
                wg_q_ref,
                (
                    pl.dslice(inter_idx * b_inter, b_inter),
                    pl.dslice(hin_idx * b_hidden_in, b_hidden_in),
                ),
            )
            wg_q_tile = wg_q_tile.T

            wu_q_tile = pl.load(
                wu_q_ref,
                (
                    pl.dslice(hin_idx * b_hidden_in, b_hidden_in),
                    pl.dslice(inter_idx * b_inter, b_inter),
                ),
            )

            h_acc_val += pl.dot(x_tile, wg_q_tile)
            u_acc_val += pl.dot(x_tile, wu_q_tile)

            return h_acc_val, u_acc_val

        h_acc, u_acc = jax.lax.fori_loop(
            0, hidden_size // b_hidden_in, hidden_in_loop_body, (h_acc, u_acc)
        )

        wg_scale_tile = pl.load(wg_scale_ref, (pl.dslice(inter_idx * b_inter, b_inter),))
        wu_scale_tile = pl.load(wu_scale_ref, (pl.dslice(inter_idx * b_inter, b_inter),))

        h_acc = h_acc * wg_scale_tile
        u_acc = u_acc * wu_scale_tile

        # Apply activation
        a_tile = jax.nn.gelu(h_acc) * u_acc
        a_tile = a_tile.astype(x_ref.dtype)

        # down projection
        wd_tile = pl.load(
            wd_ref,
            (
                pl.dslice(inter_idx * b_inter, b_inter),
                pl.dslice(hidden_out_idx * b_hidden, b_hidden),
            ),
        )

        y_acc_val += pl.dot(a_tile, wd_tile)
        return y_acc_val

    y_acc = jax.lax.fori_loop(0, intermediate_size // b_inter, inter_loop_body, y_acc)

    pl.store(
        y_ref,
        (pl.dslice(seq_idx * b_seq, b_seq), pl.dslice(hidden_out_idx * b_hidden, b_hidden)),
        y_acc.astype(y_ref.dtype),
    )


@jax.jit
def apply_fused_mlp(
    x: jax.Array,
    wg_q: jax.Array,
    wg_scale: jax.Array,
    wu_q: jax.Array,
    wu_scale: jax.Array,
    wd_q: jax.Array,
    wd_scale: jax.Array,
) -> jax.Array:
    seq_len, hidden_size = x.shape
    _, intermediate_size = wg_q.shape

    # TODO(hitesy): verify clean division
    B_SEQ = 128
    B_HIDDEN = 128
    B_INTER = 256
    B_HIDDEN_IN = 128

    grid = (seq_len // B_SEQ, hidden_size // B_HIDDEN)

    return pl.pallas_call(
        functools.partial(
            fused_quantized_mlp_kernel,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            b_seq=B_SEQ,
            b_hidden=B_HIDDEN,
            b_inter=B_INTER,
            b_hidden_in=B_HIDDEN_IN,
        ),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, wg_q, wg_scale, wu_q, wu_scale, wd_q, wd_scale)


def apply_fused_mlp_with_padding(
    x: jax.Array,
    wg_q: jax.Array,
    wg_scale: jax.Array,
    wu_q: jax.Array,
    wu_scale: jax.Array,
    wd_q: jax.Array,
    wd_scale: jax.Array,
) -> jax.Array:
    """Wraps the Pallas MLP kernel to handle arbitrary sequence lengths via padding."""

    original_seq_len, hidden_size = x.shape

    B_SEQ = 128
    pad_amount = (B_SEQ - (original_seq_len % B_SEQ)) % B_SEQ

    if pad_amount > 0:
        x_padded = jnp.pad(x, ((0, pad_amount), (0, 0)), mode="constant", constant_values=0)
    else:
        x_padded = x

    out_padded = apply_fused_mlp(x_padded, wg_q, wg_scale, wu_q, wu_scale, wd_q, wd_scale)

    if pad_amount > 0:
        out_real = out_padded[:original_seq_len, :]
    else:
        out_real = out_padded

    return out_real
