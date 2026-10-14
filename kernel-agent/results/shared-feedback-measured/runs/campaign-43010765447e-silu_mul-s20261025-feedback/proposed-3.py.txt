import torch
import triton
import triton.language as tl


@triton.jit
def silu_mul_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    # sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + exp2(-x * log2(e)))
    # log2(e) ≈ 1.4426950408889634
    LOG2E: tl.constexpr = 1.4426950408889634
    sigmoid_x = 1.0 / (1.0 + tl.exp2(-x * LOG2E))
    silu_x = x * sigmoid_x
    result = silu_x * y
    tl.store(output_ptr + offsets, result, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()
    output = torch.empty_like(x)
    n_elements = output.numel()
    if n_elements == 0:
        return output
    # Adaptive block size: smaller for small tensors to reduce launch overhead,
    # larger for big tensors to maximize memory bandwidth utilization.
    if n_elements <= 4096:
        BLOCK_SIZE = 1024
    elif n_elements <= 65536:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
