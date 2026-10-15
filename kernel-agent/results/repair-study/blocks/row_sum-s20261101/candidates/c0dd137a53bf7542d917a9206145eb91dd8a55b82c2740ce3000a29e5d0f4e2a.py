import torch
import triton
import triton.language as tl

#
@triton.jit
def _row_sum_kernel(
    X,
    Y,
    cols,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    ATOMIC: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    if ATOMIC:
        # Each block handles a chunk of columns
        start = col_block * BLOCK_SIZE
        c = start + tl.arange(0, BLOCK_SIZE)
        mask = c < cols
        v = tl.load(X + row * cols + c, mask=mask, other=0.0).to(tl.float32)
        s = tl.sum(v, axis=0)
        tl.atomic_add(Y + row, s)
    else:
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, cols, BLOCK_SIZE):
            c = off + tl.arange(0, BLOCK_SIZE)
            mask = c < cols
            v = tl.load(X + row * cols + c, mask=mask, other=0.0).to(tl.float32)
            acc += v
        s = tl.sum(acc, axis=0)
        tl.store(Y + row, s.to(OUT_DTYPE))


@triton.jit
def _cast_kernel(
    X,
    Y,
    N,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = c < N
    v = tl.load(X + c, mask=mask).to(OUT_DTYPE)
    tl.store(Y + c, v, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty(rows, dtype=x.dtype, device=x.device)
    out_dtype = tl.float32 if x.dtype == torch.float32 else tl.bfloat16
    BLOCK_SIZE = triton.next_power_of_2(cols)
    if cols <= 1024:
        # Single block per row, direct store
        _row_sum_kernel[(rows, 1)](
            x, out, cols,
            BLOCK_SIZE=BLOCK_SIZE,
            OUT_DTYPE=out_dtype,
            N_BLOCKS=1,
            ATOMIC=False,
        )
    else:
        # Use atomic add with multiple blocks per row
        n_blocks = triton.cdiv(cols, 1024)
        # Allocate fp32 temp for atomic accumulation
        tmp = torch.zeros(rows, dtype=torch.float32, device=x.device)
        _row_sum_kernel[(rows, n_blocks)](
            x, tmp, cols,
            BLOCK_SIZE=1024,
            OUT_DTYPE=tl.float32,
            N_BLOCKS=n_blocks,
            ATOMIC=True,
        )
        # Cast fp32 temp to output dtype
        cast_block = 256
        _cast_kernel[(triton.cdiv(rows, cast_block),)](
            tmp, out, rows,
            BLOCK_SIZE=cast_block,
            OUT_DTYPE=out_dtype,
        )
    return out
