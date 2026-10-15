import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X,  # pointer to input (rows, width)
    Y,  # pointer to output (rows, width)
    W,  # pointer to weight (width,)
    stride,  # row stride in elements
    N,  # width
    eps,  # epsilon
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK

    # 2D offsets: [ROWS_PER_BLOCK, BLOCK_SIZE]
    row_offsets = row_start + tl.arange(0, ROWS_PER_BLOCK)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Pointer base for X and Y
    X_base = X + row_offsets[:, None] * stride + col_offsets[None, :]
    Y_base = Y + row_offsets[:, None] * stride + col_offsets[None, :]

    # Mask for valid columns and rows
    col_mask = col_offsets[None, :] < N
    row_mask = row_offsets[:, None] < tl.num_programs(0) * ROWS_PER_BLOCK  # not needed if grid covers all rows
    # Actually, we need to know total rows. Let's pass ROWS as a parameter.
    # Wait, I didn't pass ROWS. Let me add it.

    # Let me revise: pass ROWS as a parameter.
    # I'll update the kernel signature.

    # For now, assume row_mask is all True if grid is sized correctly.
    # But we need to mask rows that exceed total rows.
    # Let's add ROWS parameter.

    # Revised approach: pass ROWS.
    # I'll rewrite the kernel properly.

    # Let me just write the correct version directly.
    pass


@triton.jit
def _rmsnorm_kernel(
    X,  # pointer to input (rows, width)
    Y,  # pointer to output (rows, width)
    W,  # pointer to weight (width,)
    stride,  # row stride in elements
    N,  # width
    ROWS,  # total rows
    eps,  # epsilon
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK

    row_offsets = row_start + tl.arange(0, ROWS_PER_BLOCK)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    X_base = X + row_offsets[:, None] * stride + col_offsets[None, :]
    Y_base = Y + row_offsets[:, None] * stride + col_offsets[None, :]

    col_mask = col_offsets[None, :] < N
    row_mask = row_offsets[:, None] < ROWS
    mask = col_mask & row_mask

    # Load X and compute sum of squares per row
    x = tl.load(X_base, mask=mask, other=0.).to(tl.float32)
    sumsq = tl.sum(x * x, axis=1)  # [ROWS_PER_BLOCK]
    mean_sq = sumsq / N
    rstd = tl.rsqrt(mean_sq + eps)  # [ROWS_PER_BLOCK]

    # Normalize, cast to input dtype, multiply by weight, store
    w = tl.load(W + col_offsets, mask=col_offsets < N, other=0.)
    normalized = (x * rstd[:, None]).to(Y.dtype.element_ty)
    y = normalized * w[None, :]
    tl.store(Y_base, y, mask=mask)


def run(x, weight):
    rows, width = x.shape
    y = torch.empty_like(x)

    # Choose block sizes
    BLOCK_SIZE = triton.next_power_of_2(width)
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096

    # Choose ROWS_PER_BLOCK to balance occupancy
    # For small widths, process more rows per program
    if width <= 128:
        ROWS_PER_BLOCK = 8
    elif width <= 512:
        ROWS_PER_BLOCK = 4
    elif width <= 1024:
        ROWS_PER_BLOCK = 2
    else:
        ROWS_PER_BLOCK = 1

    # Ensure ROWS_PER_BLOCK doesn't exceed rows
    if ROWS_PER_BLOCK > rows:
        ROWS_PER_BLOCK = max(1, rows)

    grid = (triton.cdiv(rows, ROWS_PER_BLOCK),)

    _rmsnorm_kernel[grid](
        x, y, weight,
        width, width, rows, 1e-6,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
