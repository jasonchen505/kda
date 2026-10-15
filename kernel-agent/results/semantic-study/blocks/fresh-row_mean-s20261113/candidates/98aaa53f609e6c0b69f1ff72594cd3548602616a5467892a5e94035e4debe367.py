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
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        acc += a
    total = tl.sum(acc, axis=0)
    mean = total / N
    if IS_BF16:
        val = mean.to(tl.bfloat16)
    else:
        val = mean.to(tl.float32)
    tl.store(Y, val)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    rows, width = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or width == 0:
        return out
    BLOCK_SIZE = triton.next_power_of_2(width)
    IS_BF16 = x.dtype == torch.bfloat16
    _row_mean_kernel[(rows,)](
        x, out, width, width, BLOCK_SIZE=BLOCK_SIZE, IS_BF16=IS_BF16
    )
    return out
