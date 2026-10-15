import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(
    X,
    Y,
    cols,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    SINGLE: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if SINGLE:
        c = tl.arange(0, BLOCK_SIZE)
        mask = c < cols
        v = tl.load(X + row * cols + c, mask=mask, other=0.0).to(tl.float32)
        acc += v
    else:
        for off in range(0, cols, BLOCK_SIZE):
            c = off + tl.arange(0, BLOCK_SIZE)
            mask = c < cols
            v = tl.load(X + row * cols + c, mask=mask, other=0.0).to(tl.float32)
            acc += v
    s = tl.sum(acc, axis=0)
    tl.store(Y + row, s.to(OUT_DTYPE))


def run(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty(rows, dtype=x.dtype, device=x.device)
    if cols <= 256:
        BLOCK_SIZE = 256
    elif cols <= 512:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = triton.next_power_of_2(cols)
    SINGLE = cols <= BLOCK_SIZE
    out_dtype = tl.float32 if x.dtype == torch.float32 else tl.bfloat16
    _row_sum_kernel[(rows,)](x, out, cols, BLOCK_SIZE=BLOCK_SIZE, OUT_DTYPE=out_dtype, SINGLE=SINGLE)
    return out
