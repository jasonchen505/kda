import torch
import triton
import triton.language as tl


@triton.jit
def _partial_sum_kernel(
    X,
    S,
    stride,
    N,
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    X += row * stride
    S += row
    base = chunk * CHUNK
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK_SIZE):
        cols = base + off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        acc += a
    total = tl.sum(acc, axis=0)
    tl.atomic_add(S, total)


@triton.jit
def _finalize_kernel(
    S,
    Y,
    N,
    OUT_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.load(S + row)
    mean = s / N
    tl.store(Y + row, mean.to(OUT_DTYPE))


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or width == 0:
        return out
    s_buf = torch.zeros((rows,), dtype=torch.float32, device=x.device)
    BLOCK_SIZE = 1024
    # Choose chunk size to get reasonable parallelism
    # For small rows, split width into multiple chunks
    num_chunks = max(1, min(8, triton.cdiv(width, BLOCK_SIZE)))
    chunk_size = triton.cdiv(width, num_chunks)
    # Round chunk_size up to multiple of BLOCK_SIZE for clean loop
    chunk_size = triton.cdiv(chunk_size, BLOCK_SIZE) * BLOCK_SIZE
    # Recompute num_chunks based on actual chunk_size
    num_chunks = triton.cdiv(width, chunk_size)
    grid1 = (rows, num_chunks)
    _partial_sum_kernel[grid1](x, s_buf, width, width, chunk_size, BLOCK_SIZE=BLOCK_SIZE)
    grid2 = (rows,)
    if x.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    else:
        out_dtype = tl.float32
    _finalize_kernel[grid2](s_buf, out, width, OUT_DTYPE=out_dtype)
    return out
