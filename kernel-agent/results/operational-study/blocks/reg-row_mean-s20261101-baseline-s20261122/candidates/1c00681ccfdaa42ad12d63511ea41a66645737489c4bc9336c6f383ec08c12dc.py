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
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    acc = tl.zeros([], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        acc += tl.sum(a, axis=0)
    total = acc
    mean = total / N
    tl.store(Y, mean)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    y = torch.empty((rows,), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 1024
    grid = (rows,)
    _mean_kernel[grid](x, y, width, width, BLOCK_SIZE=BLOCK_SIZE)
    return y

