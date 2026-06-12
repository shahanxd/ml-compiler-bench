"""
Shared benchmarking utilities for DVM and BladeDISC paper reproduction.
"""

import torch
import time
import gc
import csv
import os
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional


@dataclass
class BenchmarkResult:
    """Stores a single benchmark measurement."""
    experiment: str
    mode: str  # 'eager', 'compiled', 'bladedisc', etc.
    model_or_op: str
    shape_desc: str
    warmup_iters: int
    measure_iters: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    compile_time_s: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    notes: str = ""


def benchmark_latency(
    fn: Callable,
    warmup: int = 10,
    repeat: int = 100,
    sync: bool = True,
) -> dict:
    """
    Benchmark a callable with proper GPU synchronization.
    
    Args:
        fn: Function to benchmark (no args — use lambda or partial)
        warmup: Number of warmup iterations (discarded)
        repeat: Number of measured iterations
        sync: Whether to call torch.cuda.synchronize() (MUST be True for GPU)
    
    Returns:
        Dict with mean_ms, std_ms, min_ms, max_ms
    """
    # Warmup — lets GPU clock boost, JIT compile, populate caches
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(repeat):
        if sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    t = torch.tensor(times)
    return {
        "mean_ms": t.mean().item(),
        "std_ms": t.std().item(),
        "min_ms": t.min().item(),
        "max_ms": t.max().item(),
    }


def measure_compile_time(compile_fn: Callable) -> tuple:
    """
    Measure the time it takes to compile a model.
    Returns (compiled_model, compile_time_seconds).
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = compile_fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return result, elapsed


def measure_peak_memory(fn: Callable) -> float:
    """
    Measure peak GPU memory usage of a function call.
    Returns peak memory in MB.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def save_results(results: list[BenchmarkResult], filepath: str):
    """Save benchmark results to CSV."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"Results saved to {filepath}")


def get_system_info() -> dict:
    """Collect system info for reproducibility."""
    info = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_vram_mb": torch.cuda.get_device_properties(0).total_memory / (1024**2) if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda or "N/A",
        "pytorch_version": torch.__version__,
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
    }
    return info


def print_system_info():
    """Print system info for report header."""
    info = get_system_info()
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60)
