import torch
import triton
import triton.language as tl

@triton.jit
def _sum_rows_kernel(
    X,  # pointer to input
    Y,  # pointer to output
    rows,  # number of rows
    width,  # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < width
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for r in range(rows):
        ptr = X + r * width + cols
        val = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        acc += val
    # Cast to output dtype (same as input)
    # We need to know the dtype. Since we can't pass dtype, we rely on the fact that
    # the output tensor Y has the same dtype as X.
    # tl.store will cast acc to the dtype of Y automatically if we just store acc?
    # No, tl.store casts the value to the pointer's dtype.
    # So we just store acc.
    tl.store(Y + cols, acc, mask=mask)

def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    y = torch.empty_like(x, shape=(width,), dtype=x.dtype)
    # Choose block size
    # We want to cover the width. Since width <= 4096, we can use a block size that divides 4096 or just use a fixed size and mask.
    # Let's use a block size that is a power of 2.
    # If width is small, we can use a smaller block size to avoid wasting threads, but masking handles it.
    # Let's pick BLOCK_SIZE = 1024 or 2048. 1024 is safe for width up to 4096 (4 blocks).
    # Actually, we can just use a block size that is at least width? No, block size must be power of 2.
    # Let's use BLOCK_SIZE = 1024. If width > 1024, we need multiple programs.
    # Grid size = triton.cdiv(width, BLOCK_SIZE)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(width, BLOCK_SIZE),)
    _sum_rows_kernel[grid](x, y, rows, width, BLOCK_SIZE=BLOCK_SIZE)
    return y
