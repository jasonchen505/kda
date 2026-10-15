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
    EVEN: tl.constexpr,
    LOG2_N: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    if EVEN:
        a = tl.load(X + cols).to(tl.float32)
    else:
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    total = tl.sum(a, axis=0)
    mean = total / N
    if IS_BF16:
        val = mean.to(tl.bfloat16)
    else:
        val = mean.to(tl.float32)
    tl.store(Y + row, val)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    rows, width = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or width == 0:
        return out
    BLOCK_SIZE = triton.next_power_of_2(width)
    IS_BF16 = x.dtype == torch.bfloat16
    EVEN = (width == BLOCK_SIZE)
    LOG2_N = max(1, int(triton.cdiv(width, 128)))
    NUM_WARPS = 1
    NUM_STAGES = 1
    _row_mean_kernel[(rows,)](
        x, out, width, width,
        BLOCK_SIZE=BLOCK_SIZE, IS_BF16=IS_BF16, EVEN=EVEN,
        LOG2_N=LOG2_N, NUM_WARPS=NUM_WARPS, NUM_STAGES=NUM_STAGES,
    )
    return out
