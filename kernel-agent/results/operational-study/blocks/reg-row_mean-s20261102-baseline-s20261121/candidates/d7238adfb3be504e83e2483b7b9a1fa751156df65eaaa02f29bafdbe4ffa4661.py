import torch
import triton
import triton.language as tl
import math


@triton.jit
def _row_mean_small(
    X,
    Y,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    total = tl.sum(a, axis=0)
    mean = total / N
    if OUT_FP32:
        tl.store(Y + row, mean)
    else:
        tl.store(Y + row, mean.to(tl.bfloat16))


@triton.jit
def _row_mean_chunk_sum(
    X,
    TMP,
    stride,
    N,
    CHUNK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    X += row * stride + chunk * CHUNK
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < (N - chunk * CHUNK)
    a = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    s = tl.sum(a, axis=0)
    tl.atomic_add(TMP + row, s)


@triton.jit
def _row_mean_finalize(
    TMP,
    Y,
    N,
    OUT_FP32: tl.constexpr,
):
    row = tl.program_id(0)
    total = tl.load(TMP + row)
    mean = total / N
    if OUT_FP32:
        tl.store(Y + row, mean)
    else:
        tl.store(Y + row, mean.to(tl.bfloat16))


def _next_pow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def run(x):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    rows, width = x.shape
    out = torch.empty((rows,), dtype=x.dtype, device=x.device)
    if rows == 0 or width == 0:
        return out
    out_fp32 = x.dtype == torch.float32
    if width <= 1024:
        BLOCK_SIZE = _next_pow2(width)
        grid = (rows,)
        _row_mean_small[grid](x, out, width, width, BLOCK_SIZE=BLOCK_SIZE, OUT_FP32=out_fp32)
    else:
        CHUNK = 1024
        BLOCK_SIZE = 1024
        n_chunks = triton.cdiv(width, CHUNK)
        tmp = torch.zeros((rows,), dtype=torch.float32, device=x.device)
        grid = (rows, n_chunks)
        _row_mean_chunk_sum[grid](x, tmp, width, width, CHUNK=CHUNK, BLOCK_SIZE=BLOCK_SIZE)
        grid2 = (rows,)
        _row_mean_finalize[grid2](tmp, out, width, OUT_FP32=out_fp32)
    return out
