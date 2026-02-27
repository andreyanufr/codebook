"""
Optimized one_hot encoding for uint8 tensors using Triton.

Drop-in replacement for torch.nn.functional.one_hot that accepts uint8 input
instead of long, avoiding the costly dtype cast on GPU.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _one_hot_uint8_scatter_kernel(
    indices_ptr,   # [N] uint8 input indices
    output_ptr,    # [N, num_classes] pre-zeroed output
    num_classes,   # int
    N,             # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter 1s into a pre-zeroed output at the correct one-hot positions."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load uint8 indices → int32 for address arithmetic
    idx = tl.load(indices_ptr + offsets, mask=mask, other=0).to(tl.int32)

    # output[i, idx[i]] = 1
    out_offsets = offsets * num_classes + idx
    ones = tl.full([BLOCK_SIZE], value=1, dtype=output_ptr.dtype.element_ty)
    tl.store(output_ptr + out_offsets, ones, mask=mask)


@triton.jit
def _one_hot_uint8_full_kernel(
    indices_ptr,   # [N] uint8 input indices
    output_ptr,    # [N, num_classes] output (written in full, no pre-zero needed)
    num_classes: tl.constexpr,
    N,
    BLOCK_SIZE: tl.constexpr,
    CLASS_BLOCK: tl.constexpr,
):
    """
    Write entire one-hot rows without needing a pre-zeroed buffer.

    Each program handles BLOCK_SIZE rows; it iterates over the class
    dimension in CLASS_BLOCK-sized tiles and writes 0/1 directly.
    """
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)       # [BLOCK_SIZE] - offset + range
    row_mask = row_ids < N

    idx = tl.load(indices_ptr + row_ids, mask=row_mask, other=0).to(tl.int32)  # [BLOCK_SIZE] - block wof class indices for this block of rows

    for class_start in tl.static_range(0, num_classes, CLASS_BLOCK): # one iteration ?
        col_ids = class_start + tl.arange(0, CLASS_BLOCK)       # [CLASS_BLOCK]
        col_mask = col_ids < num_classes

        # 2-D offsets: [BLOCK_SIZE, CLASS_BLOCK]
        out_offsets = row_ids[:, None] * num_classes + col_ids[None, :]
        values = (idx[:, None] == col_ids[None, :]).to(output_ptr.dtype.element_ty)

        full_mask = row_mask[:, None] & col_mask[None, :]
        tl.store(output_ptr + out_offsets, values, mask=full_mask)


@triton.jit
def _one_hot_uint8_full_kernel_flat(
    indices_ptr,   # [N] uint8 input indices
    output_ptr,    # [N, num_classes] output (written in full, no pre-zero needed)
    num_classes: tl.constexpr,
    N,
    BLOCK_SIZE: tl.constexpr,
    CLASS_BLOCK: tl.constexpr,
):
    """
    Write entire one-hot rows without needing a pre-zeroed buffer.

    Each program handles BLOCK_SIZE rows; it iterates over the class
    dimension in CLASS_BLOCK-sized tiles and writes 0/1 directly.
    """
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)       # [BLOCK_SIZE] - offset + range
    row_mask = row_ids < N

    idx = tl.load(indices_ptr + row_ids, mask=row_mask, other=0).to(tl.int32)  # [BLOCK_SIZE] - block wof class indices for this block of rows

    for class_start in tl.static_range(0, num_classes, CLASS_BLOCK): # one iteration ?
        col_ids = class_start + tl.arange(0, CLASS_BLOCK)       # [CLASS_BLOCK]
        col_mask = col_ids < num_classes

        # 2-D offsets: [BLOCK_SIZE, CLASS_BLOCK]
        #out_offsets = row_ids[:, None] * num_classes + col_ids[None, :]
        values = (idx[:, None] == col_ids[None, :]).to(output_ptr.dtype.element_ty)

        full_mask = row_mask[:, None] & col_mask[None, :]
        tl.store(output_ptr + pid * BLOCK_SIZE * num_classes, values, mask=full_mask)


def one_hot(
    tensor: torch.Tensor,
    num_classes: int = -1,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    One-hot encode a **uint8** index tensor on the GPU using a Triton kernel.

    Interface mirrors ``torch.nn.functional.one_hot``:

    * ``tensor``      – index tensor (**must** be ``torch.uint8``).
    * ``num_classes``  – number of classes; ``-1`` to auto-detect (max + 1).
    * ``dtype``        – output dtype; defaults to ``torch.int64`` (same as
                         the standard ``one_hot``).

    Returns a tensor of shape ``(*tensor.shape, num_classes)``.
    """
    # ── validation ──────────────────────────────────────────────────────
    if tensor.dtype != torch.uint8:
        raise TypeError(
            f"one_hot_uint8 expects a uint8 tensor, got {tensor.dtype}"
        )

    if num_classes == -1:
        num_classes = int(tensor.max().item()) + 1

    if not 1 <= num_classes <= 256:
        raise ValueError(
            f"num_classes must be in [1, 256] for uint8 input, got {num_classes}"
        )

    if dtype is None:
        dtype = torch.int64          # match stdlib default

    original_shape = tensor.shape
    flat = tensor.reshape(-1).contiguous()
    N = flat.numel()

    # ── trivial / CPU path ──────────────────────────────────────────────
    if N == 0:
        return torch.zeros(*original_shape, num_classes, dtype=dtype,
                           device=tensor.device)

    if not tensor.is_cuda:
        out = torch.zeros(N, num_classes, dtype=dtype, device=tensor.device)
        out.scatter_(1, flat.to(torch.int64).unsqueeze(1), 1)
        return out.reshape(*original_shape, num_classes)

    # ── Triton GPU path ─────────────────────────────────────────────────
    # Heuristic: for small num_classes the "full" kernel that writes
    # complete rows (avoiding the separate torch.zeros) is faster; for
    # large num_classes the "scatter" kernel that only touches N elements
    # in a pre-zeroed buffer wins.
    USE_FULL_KERNEL = num_classes <= 64

    if USE_FULL_KERNEL:
        out = torch.empty(N, num_classes, dtype=dtype, device=tensor.device)
        BLOCK_SIZE = 256
        CLASS_BLOCK = triton.next_power_of_2(num_classes)
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _one_hot_uint8_full_kernel[grid](
            flat, out, num_classes, N,
            BLOCK_SIZE=BLOCK_SIZE,
            CLASS_BLOCK=CLASS_BLOCK,
        )
    else:
        out = torch.zeros(N, num_classes, dtype=dtype, device=tensor.device)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _one_hot_uint8_scatter_kernel[grid](
            flat, out, num_classes, N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out.reshape(*original_shape, num_classes)


# ── quick self-test & benchmark ─────────────────────────────────────────
if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── correctness check ───────────────────────────────────────────────
    for nc in [2, 16, 64, 128, 256]:
        x = torch.randint(0, nc, (1024,), dtype=torch.uint8, device=device)
        ours  = one_hot(x, num_classes=nc)
        ref   = torch.nn.functional.one_hot(x.long(), num_classes=nc)
        assert ours.shape == ref.shape, f"Shape mismatch for nc={nc}"
        assert (ours == ref).all(),     f"Value mismatch for nc={nc}"
    print("✓ Correctness checks passed\n")

    # ── multi-dim check ─────────────────────────────────────────────────
    x = torch.randint(0, 32, (4, 8, 16), dtype=torch.uint8, device=device)
    ours = one_hot(x, num_classes=32)
    ref  = torch.nn.functional.one_hot(x.long(), num_classes=32)
    assert ours.shape == ref.shape == (4, 8, 16, 32)
    assert (ours == ref).all()
    print("✓ Multi-dimensional check passed\n")

    # ── dtype check ─────────────────────────────────────────────────────
    x = torch.randint(0, 10, (64,), dtype=torch.uint8, device=device)
    for dt in [torch.float32, torch.float16, torch.bfloat16, torch.int32]:
        out = one_hot(x, num_classes=10, dtype=dt)
        assert out.dtype == dt, f"dtype mismatch: expected {dt}, got {out.dtype}"
    print("✓ dtype check passed\n")

    if device == "cuda":
        # ── benchmark ───────────────────────────────────────────────────
        print(f"{'N':>12} {'classes':>8} {'stdlib (ms)':>12} {'ours (ms)':>12} {'speedup':>8}")
        print("-" * 58)
        for N, nc in [(1 << 16, 256), (1 << 18, 256), (1 << 20, 256),
                      (1 << 20, 64), (1 << 20, 16), (1 << 22, 256)]:
            x = torch.randint(0, nc, (N,), dtype=torch.uint8, device=device)
            x_long = x.long()

            # warm up
            for _ in range(10):
                torch.nn.functional.one_hot(x_long, nc)
                one_hot(x, nc)
            torch.cuda.synchronize()

            iters = 100
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                torch.nn.functional.one_hot(x_long, nc)
            torch.cuda.synchronize()
            std_ms = (time.perf_counter() - t0) / iters * 1000

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                one_hot(x, nc)
            torch.cuda.synchronize()
            our_ms = (time.perf_counter() - t0) / iters * 1000

            print(f"{N:>12,} {nc:>8} {std_ms:>12.3f} {our_ms:>12.3f} {std_ms/our_ms:>7.2f}x")
