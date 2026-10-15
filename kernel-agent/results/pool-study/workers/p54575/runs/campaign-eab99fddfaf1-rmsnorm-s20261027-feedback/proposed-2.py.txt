import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X,  # pointer to input (rows, width)
    Y,  # pointer to output (rows, width)
    W,  # pointer to weight (width,)
    stride,  # row stride in elements
    N,  # width (number of columns)
    eps,  # epsilon
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_SIZE)

    # 2D offsets: [ROWS_PER_BLOCK, BLOCK_SIZE]
    row_mask = rows < tl.num_programs(0) * ROWS_PER_BLOCK  # not needed, use actual rows
    # Actually, we need to mask rows that exceed total rows
    # But we don't have total rows as argument. Let's pass it.

    # Wait, I need to pass total rows. Let me revise.

    # For now, assume rows are within bounds. We'll mask based on actual row index.
    # But we need total rows. Let's add it as argument.

    # Revised approach: pass total rows
    # Actually, let's keep it simple and just use 1D per row but with better block size.

    # Let me try a different approach: use 1D kernel but with larger block size for small widths.

    # For small widths, use BLOCK_SIZE = next_power_of_2(width) and process one row per program.
    # For large widths, use BLOCK_SIZE = 1024 and loop.

    # The issue is launch overhead for small rows. Let's try to process multiple rows.

    # I'll pass total rows as an argument.
    pass


def run(x, weight):
    rows, width = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(width)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    grid = (rows,)
    _rmsnorm_kernel[grid](
        x, y, weight,
        width, width, 1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
