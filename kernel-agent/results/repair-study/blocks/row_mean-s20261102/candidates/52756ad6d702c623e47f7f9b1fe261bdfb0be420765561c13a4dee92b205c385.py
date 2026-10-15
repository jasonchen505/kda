import torch
import triton
import triton.language as tl


@triton.jit
def _partial_sum_kernel(
    X,
    P,
    stride_x,
    stride_p,
    N,
    CHUNK,
    SPLIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // SPLIT
    chunk_id = pid % SPLIT
    start = chunk_id * CHUNK
    end = tl.minimum(start + CHUNK, N)
    X += row * stride_x
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(start, end, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < end
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        acc += a
    s = tl.sum(acc, axis=0)
    P += row * stride_p + chunk_id
    tl.store(P, s)


@triton.jit
def _reduce_kernel(
    P,
    Y,
    stride_p,
    N,
    SPLIT: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    P += row * stride_p
    offs = tl.arange(0, SPLIT)
    mask = offs < SPLIT
    partials = tl.load(P + offs, mask=mask, other=0.0)
    total = tl.sum(partials, axis=0)
    mean = total / N
    if IS_BF16:
        out = mean.to(tl.bfloat16)
    else:
        out = mean.to(tl.float32)
    tl.store(Y + row, out)


def run(x):
    rows, width = x.shape
    out = torch.empty(rows, dtype=x.dtype, device=x.device)
    is_bf16 = x.dtype == torch.bfloat16
    SPLIT = max(1, min(8, triton.cdiv(width, 256)))
    CHUNK = triton.cdiv(width, SPLIT)
    BLOCK_SIZE = 256
    partials = torch.empty(rows, SPLIT, dtype=torch.float32, device=x.device)
    grid1 = (rows * SPLIT,)
    _partial_sum_kernel[grid1](x, partials, width, SPLIT, width, CHUNK, SPLIT=SPLIT, BLOCK_SIZE=BLOCK_SIZE, IS_BF16=is_bf16)
    grid2 = (rows,)
    _reduce_kernel[grid2](partials, out, SPLIT, width, SPLIT=SPLIT, IS_BF16=is_bf16)
    return out
