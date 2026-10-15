import torch
import triton
import triton.language as tl


@triton.jit
def silu_mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    out = (x * sig) * y
    tl.store(out_ptr + offsets, out, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()
    assert x.dtype in (torch.float32, torch.bfloat16)
    out = torch.empty_like(x)
    n = x.numel()
    if n <= 8192:
        BLOCK_SIZE = 256
    elif n <= 131072:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    silu_mul_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
