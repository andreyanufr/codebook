import argparse
import math
import statistics
from dataclasses import dataclass

import torch


@dataclass
class BenchmarkResult:
    name: str
    mode: str
    mean_ms: float
    std_ms: float
    mean_peak_mem_mb: float
    std_peak_mem_mb: float


def _dtype_from_string(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _format_table(results: list[BenchmarkResult]) -> str:
    headers = [
        "Approach",
        "Mode",
        "Time Mean (ms)",
        "Time Std (ms)",
        "Peak Mem Mean (MB)",
        "Peak Mem Std (MB)",
    ]
    rows = [
        [
            r.name,
            r.mode,
            f"{r.mean_ms:.4f}",
            f"{r.std_ms:.4f}",
            f"{r.mean_peak_mem_mb:.2f}",
            f"{r.std_peak_mem_mb:.2f}",
        ]
        for r in results
    ]

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(value))

    def fmt_row(items: list[str]) -> str:
        return " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(items))

    sep = "-+-".join("-" * w for w in col_widths)
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def _run_cuda_benchmark(
    fn,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> tuple[list[float], list[float]]:
    times_ms: list[float] = []
    peak_mem_mb: list[float] = []

    torch.cuda.synchronize(device)
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize(device)

    for _ in range(repeat):
        torch.cuda.synchronize(device)
        start_alloc = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        out = fn()
        end_event.record()

        torch.cuda.synchronize(device)

        elapsed = start_event.elapsed_time(end_event)
        peak_delta_bytes = max(torch.cuda.max_memory_allocated(device) - start_alloc, 0)

        times_ms.append(float(elapsed))
        peak_mem_mb.append(float(peak_delta_bytes) / (1024 ** 2))

        del out

    return times_ms, peak_mem_mb


def _benchmark_one(
    name: str,
    mode: str,
    fn,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> BenchmarkResult:
    times_ms, peak_mem_mb = _run_cuda_benchmark(fn=fn, device=device, warmup=warmup, repeat=repeat)

    return BenchmarkResult(
        name=name,
        mode=mode,
        mean_ms=statistics.mean(times_ms),
        std_ms=statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        mean_peak_mem_mb=statistics.mean(peak_mem_mb),
        std_peak_mem_mb=statistics.pstdev(peak_mem_mb) if len(peak_mem_mb) > 1 else 0.0,
    )


def _build_one_hot(indexes: torch.Tensor, num_codes: int, dtype: torch.dtype) -> torch.Tensor:
    out = torch.zeros(*indexes.shape, num_codes, device=indexes.device, dtype=dtype)
    out.scatter_(-1, indexes.unsqueeze(-1), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark codebook lookup strategies on GPU.")
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--n-bits", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    if args.in_features % args.group_size != 0:
        raise ValueError("in-features must be divisible by group-size")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    dtype = _dtype_from_string(args.dtype)
    num_codes = 2 ** args.n_bits

    grouped_shape = (args.out_features, args.in_features // args.group_size, args.group_size)

    codebook = torch.randn(num_codes, device=device, dtype=dtype, requires_grad=True)
    indexes = torch.randint(0, num_codes, grouped_shape, device=device, dtype=torch.int64)

    one_hot_precomputed = _build_one_hot(indexes, num_codes=num_codes, dtype=dtype).contiguous()

    def direct_indexing():
        return codebook[indexes]

    def index_select_lookup():
        flat_indexes = indexes.reshape(-1)
        return codebook.index_select(0, flat_indexes).view_as(indexes)

    def one_hot_dynamic():
        one_hot = _build_one_hot(indexes, num_codes=num_codes, dtype=dtype)
        return (codebook * one_hot).mean(dim=3)

    def one_hot_precomputed_case():
        return (codebook * one_hot_precomputed).mean(dim=3)

    def direct_indexing_fw_bw():
        codebook.grad = None
        out = direct_indexing()
        out.mean().backward()
        return out

    def index_select_fw_bw():
        codebook.grad = None
        out = index_select_lookup()
        out.mean().backward()
        return out

    def one_hot_dynamic_fw_bw():
        codebook.grad = None
        out = one_hot_dynamic()
        out.mean().backward()
        return out

    def one_hot_precomputed_fw_bw():
        codebook.grad = None
        out = one_hot_precomputed_case()
        out.mean().backward()
        return out

    # Sanity check for output shape compatibility.
    out_direct = direct_indexing()
    out_index_select = index_select_lookup()
    out_one_hot_dynamic = one_hot_dynamic()
    out_one_hot_precomputed = one_hot_precomputed_case()

    expected_shape = grouped_shape
    if out_direct.shape != expected_shape or out_index_select.shape != expected_shape:
        raise RuntimeError("Indexing/index_select outputs do not match expected grouped shape")
    if out_one_hot_dynamic.shape != expected_shape or out_one_hot_precomputed.shape != expected_shape:
        raise RuntimeError("one_hot outputs do not match expected grouped shape")

    max_diff = (out_direct - out_index_select).abs().max().item()
    if not math.isfinite(max_diff):
        raise RuntimeError("Non-finite values detected in lookup outputs")

    del out_direct, out_index_select, out_one_hot_dynamic, out_one_hot_precomputed
    torch.cuda.synchronize(device)

    forward_results = [
        _benchmark_one("direct_indexing", "forward", direct_indexing, device, args.warmup, args.repeat),
        _benchmark_one("index_select", "forward", index_select_lookup, device, args.warmup, args.repeat),
        _benchmark_one("one_hot_dynamic", "forward", one_hot_dynamic, device, args.warmup, args.repeat),
        _benchmark_one("one_hot_precomputed", "forward", one_hot_precomputed_case, device, args.warmup, args.repeat),
    ]

    backward_results = [
        _benchmark_one("direct_indexing", "forward+backward", direct_indexing_fw_bw, device, args.warmup, args.repeat),
        _benchmark_one("index_select", "forward+backward", index_select_fw_bw, device, args.warmup, args.repeat),
        _benchmark_one("one_hot_dynamic", "forward+backward", one_hot_dynamic_fw_bw, device, args.warmup, args.repeat),
        _benchmark_one("one_hot_precomputed", "forward+backward", one_hot_precomputed_fw_bw, device, args.warmup, args.repeat),
    ]

    print("Benchmark configuration:")
    print(
        f"  out_features={args.out_features}, in_features={args.in_features}, ",
        f"group_size={args.group_size}, n_bits={args.n_bits}, dtype={args.dtype}, ",
        f"num_codes={num_codes}, warmup={args.warmup}, repeat={args.repeat}"
    )
    print()
    print("Forward-only:")
    print(_format_table(forward_results))
    print()
    print("Forward+backward:")
    print(_format_table(backward_results))


if __name__ == "__main__":
    main()
