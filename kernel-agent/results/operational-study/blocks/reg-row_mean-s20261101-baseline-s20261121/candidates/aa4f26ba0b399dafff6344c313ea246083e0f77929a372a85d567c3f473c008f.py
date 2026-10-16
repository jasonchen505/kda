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
    IS_BF16: tl.constexpr,
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
    if IS_BF16:
        out = mean.to(tl.bfloat16)
    else:
        out = mean.to(tl.float32)
    tl.store(Y + row, out)


def _pick_block(width: int) -> int:
    if width <= 256:
        return 256
    if width <= 512:
        return 512
    return 1024


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    y = torch.empty((rows,), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = _pick_block(width)
    grid = (rows,)
    is_bf16 = x.dtype == torch.bfloat16
    _mean_kernel[grid](x, y, width, width, BLOCK_SIZE=BLOCK_SIZE, IS_BF16=is_bf16)
    return y
