import torch
import triton
import triton.language as tl


@triton.jit
def _sum_rows_kernel(
    X,
    Y,
    rows,
    width,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    EVEN_W: tl.constexpr,
    EVEN_R: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    row_start = pid_r * BLOCK_R
    if EVEN_W:
        mask_c = None
    else:
        mask_c = cols < width
    if EVEN_R:
        mask_r = None
    else:
        mask_r = (row_start + tl.arange(0, BLOCK_R)) < rows
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for r in range(row_start, row_start + BLOCK_R):
        if EVEN_R:
            ptr = X + r * width + cols
            val = tl.load(ptr).to(tl.float32)
        else:
            ptr = X + r * width + cols
            val = tl.load(ptr, mask=mask_c, other=0.0).to(tl.float32)
        acc += val
    if EVEN_W:
        tl.atomic_add(Y + cols, acc)
    else:
        tl.atomic_add(Y + cols, acc, mask=mask_c)


def run(x: torch.Tensor) -> torch.Tensor:
    rows, width = x.shape
    y = torch.zeros((width,), device=x.device, dtype=x.dtype)
    BLOCK_C = 256
    BLOCK_R = 32
    grid = (triton.cdiv(width, BLOCK_C), triton.cdiv(rows, BLOCK_R))
    _sum_rows_kernel[grid](x, y, rows, width, BLOCK_C=BLOCK_C, BLOCK_R=BLOCK_R, EVEN_W=(width % BLOCK_C == 0), EVEN_R=(rows % BLOCK_R == 0))
    return y
