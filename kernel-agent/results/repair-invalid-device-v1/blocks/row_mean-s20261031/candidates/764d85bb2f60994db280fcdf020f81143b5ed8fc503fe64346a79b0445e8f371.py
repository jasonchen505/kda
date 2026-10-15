import torch
import triton
import triton.language as tl


@triton.jit
def _mean_kernel(
    X,
    Y,
    stride_x,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x
    Y += row
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        acc += a
    s = tl.sum(acc, axis=0)
    mean = s / N
    tl.store(Y, mean)
    

def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, cols = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or cols == 0:
        return out
    BLOCK_SIZE = triton.next_power_of_2(cols)
    grid = (rows,)
    _mean_kernel[grid](x, out, cols, cols, BLOCK_SIZE=BLOCK_SIZE)
    return out
