import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X,
    W,
    Y,
    stride_x,
    stride_y,
    width,
    eps,
    BLOCK: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_F32: tl.constexpr,
    SINGLE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y
    if SINGLE:
        cols = tl.arange(0, BLOCK)
        mask = cols < width
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_x2 = tl.sum(x * x, axis=0)
        mean_x2 = sum_x2 / width
        rstd = tl.rsqrt(mean_x2 + eps)
        normalized = x * rstd
        if IS_BF16:
            normalized = normalized.to(tl.bfloat16)
        elif IS_F32:
            normalized = normalized.to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0)
        y = w * normalized
        tl.store(y_ptr + cols, y, mask=mask)
    else:
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, width, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < width
            x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            acc += x * x
        sum_x2 = tl.sum(acc, axis=0)
        mean_x2 = sum_x2 / width
        rstd = tl.rsqrt(mean_x2 + eps)
        for off in range(0, width, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < width
            x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            normalized = x * rstd
            if IS_BF16:
                normalized = normalized.to(tl.bfloat16)
            elif IS_F32:
                normalized = normalized.to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0)
            y = w * normalized
            tl.store(y_ptr + cols, y, mask=mask)


def run(x, weight):
    rows, width = x.shape
    out = torch.empty_like(x)
    is_bf16 = x.dtype == torch.bfloat16
    is_f32 = x.dtype == torch.float32
    if width > 2048:
        BLOCK = 4096
        SINGLE = True
    else:
        BLOCK = triton.next_power_of_2(width)
        SINGLE = False
    grid = (rows,)
    _rmsnorm_kernel[grid](
        x,
        weight,
        out,
        width,
        width,
        width,
        1e-6,
        BLOCK=BLOCK,
        IS_BF16=is_bf16,
        IS_F32=is_f32,
        SINGLE=SINGLE,
    )
    return out
