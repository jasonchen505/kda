import torch
import triton
import triton.language as tl


@triton.jit
def _mean_kernel(
    X,
    Y,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_F32: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        acc += a
    mean = tl.sum(acc, axis=0) / N
    if IS_BF16:
        out = mean.to(tl.bfloat16)
    else:
        out = mean.to(tl.float32)
    tl.store(Y + row, out)


def run(x):
    rows, width = x.shape
    out = torch.empty(rows, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 1024
    NUM_BLOCKS = triton.cdiv(width, BLOCK_SIZE)
    is_bf16 = x.dtype == torch.bfloat16
    is_f32 = x.dtype == torch.float32
    grid = (rows,)
    _mean_kernel[grid](x, out, width, width, BLOCK_SIZE=BLOCK_SIZE, NUM_BLOCKS=NUM_BLOCKS, IS_BF16=is_bf16, IS_F32=is_f32)
    return out
