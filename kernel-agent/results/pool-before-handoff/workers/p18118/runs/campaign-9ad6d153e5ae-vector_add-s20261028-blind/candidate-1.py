import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    result = (x + y).to(OUT_DTYPE)
    tl.store(out_ptr + offsets, result, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    assert x.dtype in (torch.float32, torch.bfloat16)
    assert x.dtype == y.dtype
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    out_dtype = tl.float32 if x.dtype == torch.float32 else tl.bfloat16
    _add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE, OUT_DTYPE=out_dtype)
    return out
