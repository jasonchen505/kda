import torch
import triton
import triton.language as tl


@triton.jit
def _partial_sum_kernel(
    X,
    S,
    stride_x,
    N,
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    X += row * stride_x + chunk * CHUNK
    S += row
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < CHUNK
        a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        acc += a
    s = tl.sum(acc, axis=0)
    tl.atomic_add(S, s)


@triton.jit
def _finalize_kernel(
    S,
    Y,
    N,
):
    row = tl.program_id(0)
    s = tl.load(S + row)
    mean = s / N
    tl.store(Y + row, mean)


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, cols = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or cols == 0:
        return out
    s_buf = torch.zeros((rows,), dtype=torch.float32, device=x.device)
    # Choose chunk size to get good parallelism
    # For large cols, split into multiple chunks
    if cols <= 1024:
        chunks = 1
        chunk_size = cols
    elif cols <= 2048:
        chunks = 2
        chunk_size = triton.cdiv(cols, chunks)
    else:
        chunks = 4
        chunk_size = triton.cdiv(cols, chunks)
    BLOCK_SIZE = triton.next_power_of_2(chunk_size)
    grid = (rows, chunks)
    _partial_sum_kernel[grid](x, s_buf, cols, cols, CHUNK=chunk_size, BLOCK_SIZE=BLOCK_SIZE)
    _finalize_kernel[(rows,)](s_buf, out, cols)
    return out
