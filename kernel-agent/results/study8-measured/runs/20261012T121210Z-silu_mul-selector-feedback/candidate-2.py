import torch
import triton
import triton.language as tl


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
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    result = x * sigmoid_x * y
    if IS_BF16:
        result = result.to(tl.bfloat16)
    else:
        result = result.to(tl.float32)
    tl.store(out_ptr + offsets, result, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.shape == y.shape, "Inputs must have equal shape"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    assert x.dtype in (torch.float32, torch.bfloat16), "Unsupported dtype"
    assert x.dtype == y.dtype, "Inputs must have same dtype"
    out = torch.empty_like(x)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    is_bf16 = x.dtype == torch.bfloat16
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_mul_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, IS_BF16=is_bf16, num_warps=4)
    return out
