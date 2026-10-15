import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    tile = tl.load(x_ptrs, mask=mask, other=0.0)

    tile_t = tl.trans(tile)

    out_ptrs = out_ptr + offs_n[:, None] * stride_om + offs_m[None, :] * stride_on
    tl.store(out_ptrs, tile_t, mask=mask_n[:, None] & mask_m[None, :])


def run(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "input must be on CUDA"
    assert x.dim() == 2, "input must be 2D"
    assert x.is_contiguous(), "input must be contiguous"
    assert x.dtype in (torch.float32, torch.bfloat16), "unsupported dtype"

    M, N = x.shape
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)

    stride_xm = x.stride(0)
    stride_xn = x.stride(1)
    stride_om = out.stride(0)
    stride_on = out.stride(1)

    # Tune block sizes based on matrix dimensions
    if M <= 64 and N <= 64:
        BLOCK_M = 32
        BLOCK_N = 32
    elif M <= 128 and N <= 128:
        BLOCK_M = 64
        BLOCK_N = 64
    elif M <= 256 and N <= 256:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 128
        BLOCK_N = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _transpose_kernel[grid](
        x, out,
        M, N,
        stride_xm, stride_xn,
        stride_om, stride_on,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out
