import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(x_ptr, y_ptr, M, N,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    mask = offs < total
    i = offs // M
    j = offs % M
    in_off = j + i * N
    x = tl.load(x_ptr + in_off, mask=mask, other=0.0)
    tl.store(y_ptr + offs, x, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    M, N = x.shape
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)
    total = M * N
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    transpose_kernel[grid](x, y, M, N, BLOCK=BLOCK)
    return y
