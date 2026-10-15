import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(
    X,
    Y,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    acc = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        acc += tl.sum(a, axis=0)
    tl.store(Y + row, acc)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, cols = x.shape
    y = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if cols <= 128:
        BLOCK_SIZE = 128
    elif cols <= 256:
        BLOCK_SIZE = 256
    elif cols <= 512:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    grid = (rows,)
    _row_sum_kernel[grid](x, y, x.stride(0), cols, BLOCK_SIZE=BLOCK_SIZE)
    return y
