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
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        acc += a
    s = tl.sum(acc, axis=0)
    tl.store(Y + row, s)


def _configs():
    return [
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
    ]


@triton.autotune(configs=_configs(), key=["N"])
@triton.jit
def _row_sum_kernel_tuned(
    X,
    Y,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        acc += a
    s = tl.sum(acc, axis=0)
    tl.store(Y + row, s)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    rows, cols = x.shape
    y = torch.empty((rows,), dtype=x.dtype, device=x.device)
    grid = (rows,)
    _row_sum_kernel_tuned[grid](x, y, cols, cols)
    return y
