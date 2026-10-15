import torch
import triton
import triton.language as tl
"""
Elementwise SiLU(x) * y kernel for CUDA tensors (fp32 or bf16).
"""

@triton.jit
def silu_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if IS_BF16:
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_x = x * sigmoid_x
    result = silu_x * y
    tl.store(out_ptr + offsets, result, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Elementwise SiLU(x) * y for equal-shape contiguous CUDA tensors."""
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.shape == y.shape, "Inputs must have equal shape"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    assert x.dtype in (torch.float32, torch.bfloat16), "Unsupported dtype"
    assert x.dtype == y.dtype, "Inputs must have same dtype"
    n_elements = x.numel()
    assert n_elements > 0, "Must have positive element count"
    assert n_elements <= 1048576, "Element count exceeds limit"
    output = torch.empty_like(x)
    is_bf16 = x.dtype == torch.bfloat16
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, IS_BF16=is_bf16)
    return output
