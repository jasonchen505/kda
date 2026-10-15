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
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    s = x + y
    if IS_BF16:
        s = s.to(tl.bfloat16)
    tl.store(out_ptr + offsets, s, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    assert x.dtype in (torch.float32, torch.bfloat16)
    assert x.is_contiguous() and y.is_contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    is_bf16 = x.dtype == torch.bfloat16
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE, IS_BF16=is_bf16)
    return out
