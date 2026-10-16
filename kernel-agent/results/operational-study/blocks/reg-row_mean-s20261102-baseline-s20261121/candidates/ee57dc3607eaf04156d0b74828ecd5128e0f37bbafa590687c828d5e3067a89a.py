import torch
import triton
import triton.language as tl


@triton.jit
def _row_mean_kernel(
    X,
    Y,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        acc += a
    total = tl.sum(acc, axis=0)
    mean = total / N
    if OUT_FP32:
        tl.store(Y + row, mean)
    else:
        tl.store(Y + row, mean.to(tl.bfloat16))


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or width == 0:
        return out
    BLOCK_SIZE = 1024
    grid = (rows,)
    out_fp32 = x.dtype == torch.float32
    _row_mean_kernel[grid](x, out, width, width, BLOCK_SIZE=BLOCK_SIZE, OUT_FP32=out_fp32)
    return out
