"""
DVM Paper Baseline Verification
================================
Paper: DVM: A Bytecode Virtual Machine Approach for Dynamic Tensor Computation
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_utils import (
    benchmark_latency, measure_compile_time, measure_peak_memory,
    BenchmarkResult, save_results, print_system_info
)

# ─────────────────────────────────────────────────────────
# Part 1: Individual Operator Benchmarks
# ─────────────────────────────────────────────────────────

def benchmark_operators():
    print("\n" + "=" * 60)
    print("PART 1: INDIVIDUAL OPERATOR BENCHMARKS")
    print("Comparing: PyTorch eager vs TorchScript JIT")
    print("=" * 60)

    results = []

    # --- MatMul ---
    matmul_configs = [
        ((128, 768), (768, 3072), "128x768 @ 768x3072"),
        ((64, 512), (512, 2048), "64x512 @ 512x2048"),
        ((32, 256), (256, 1024), "32x256 @ 256x1024"),
        ((4, 128, 768), (768, 3072), "4x128x768 @ 768x3072 (batched)"),
    ]

    for shape_a, shape_b, desc in matmul_configs:
        print(f"\n--- MatMul: {desc} ---")
        a = torch.randn(*shape_a, device="cuda")
        b = torch.randn(*shape_b, device="cuda")

        # Eager
        stats = benchmark_latency(lambda: torch.matmul(a, b))
        print(f"  Eager:      {stats['mean_ms']:.4f} ± {stats['std_ms']:.4f} ms")
        results.append(BenchmarkResult(
            experiment="DVM_baseline", mode="eager", model_or_op="MatMul",
            shape_desc=desc, warmup_iters=10, measure_iters=100, **stats
        ))

        # JIT traced
        class MatMulModule(nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)
        
        m = MatMulModule().cuda().eval()
        traced, ct = measure_compile_time(lambda: torch.jit.trace(m, (a, b)))
        jit_stats = benchmark_latency(lambda: traced(a, b))
        speedup = stats["mean_ms"] / jit_stats["mean_ms"] if jit_stats["mean_ms"] > 0 else 0
        print(f"  JIT:        {jit_stats['mean_ms']:.4f} ± {jit_stats['std_ms']:.4f} ms  (trace: {ct:.3f}s)")
        print(f"  Speedup:    {speedup:.2f}×")
        results.append(BenchmarkResult(
            experiment="DVM_baseline", mode="jit_trace", model_or_op="MatMul",
            shape_desc=desc, warmup_iters=10, measure_iters=100,
            compile_time_s=ct, **jit_stats
        ))

    # --- Element-wise Add ---
    add_configs = [
        ((128, 3072), "128x3072"),
        ((64, 2048), "64x2048"),
        ((4, 128, 3072), "4x128x3072 (batched)"),
    ]

    for shape, desc in add_configs:
        print(f"\n--- Add: {desc} ---")
        a = torch.randn(*shape, device="cuda")
        b = torch.randn(*shape, device="cuda")

        stats = benchmark_latency(lambda: torch.add(a, b))
        print(f"  Eager:      {stats['mean_ms']:.4f} ± {stats['std_ms']:.4f} ms")
        results.append(BenchmarkResult(
            experiment="DVM_baseline", mode="eager", model_or_op="Add",
            shape_desc=desc, warmup_iters=10, measure_iters=100, **stats
        ))

        class AddModule(nn.Module):
            def forward(self, x, y):
                return torch.add(x, y)

        m = AddModule().cuda().eval()
        traced, ct = measure_compile_time(lambda: torch.jit.trace(m, (a, b)))
        jit_stats = benchmark_latency(lambda: traced(a, b))
        print(f"  JIT:        {jit_stats['mean_ms']:.4f} ± {jit_stats['std_ms']:.4f} ms")
        results.append(BenchmarkResult(
            experiment="DVM_baseline", mode="jit_trace", model_or_op="Add",
            shape_desc=desc, warmup_iters=10, measure_iters=100,
            compile_time_s=ct, **jit_stats
        ))

    # --- LayerNorm ---
    ln_configs = [
        ((4, 128, 768), 768, "4x128x768, norm=768"),
        ((1, 64, 512), 512, "1x64x512, norm=512"),
    ]

    for shape, norm, desc in ln_configs:
        print(f"\n--- LayerNorm: {desc} ---")
        x = torch.randn(*shape, device="cuda")
        ln = nn.LayerNorm(norm).cuda().eval()

        stats = benchmark_latency(lambda: ln(x))
        print(f"  Eager:      {stats['mean_ms']:.4f} ± {stats['std_ms']:.4f} ms")
        results.append(BenchmarkResult(
            experiment="DVM_baseline", mode="eager", model_or_op="LayerNorm",
            shape_desc=desc, warmup_iters=10, measure_iters=100, **stats
        ))

        traced, ct = measure_compile_time(lambda: torch.jit.trace(ln, (x,)))
        jit_stats = benchmark_latency(lambda: traced(x))
        print(f"  JIT:        {jit_stats['mean_ms']:.4f} ± {jit_stats['std_ms']:.4f} ms")
        results.append(BenchmarkResult(
            experiment="DVM_baseline", mode="jit_trace", model_or_op="LayerNorm",
            shape_desc=desc, warmup_iters=10, measure_iters=100,
            compile_time_s=ct, **jit_stats
        ))

    # --- Softmax ---
    sm_configs = [
        ((4, 12, 128, 128), "4x12x128x128 (attention scores)"),
        ((1, 12, 64, 64), "1x12x64x64"),
    ]

    for shape, desc in sm_configs:
        print(f"\n--- Softmax: {desc} ---")
        x = torch.randn(*shape, device="cuda")

        stats = benchmark_latency(lambda: torch.softmax(x, dim=-1))
        print(f"  Eager:      {stats['mean_ms']:.4f} ± {stats['std_ms']:.4f} ms")
        results.append(BenchmarkResult(
            experiment="DVM_baseline", mode="eager", model_or_op="Softmax",
            shape_desc=desc, warmup_iters=10, measure_iters=100, **stats
        ))

        class SoftmaxModule(nn.Module):
            def forward(self, x):
                return torch.softmax(x, dim=-1)

        m = SoftmaxModule().cuda().eval()
        traced, ct = measure_compile_time(lambda: torch.jit.trace(m, (x,)))
        jit_stats = benchmark_latency(lambda: traced(x))
        print(f"  JIT:        {jit_stats['mean_ms']:.4f} ± {jit_stats['std_ms']:.4f} ms")
        results.append(BenchmarkResult(
            experiment="DVM_baseline", mode="jit_trace", model_or_op="Softmax",
            shape_desc=desc, warmup_iters=10, measure_iters=100,
            compile_time_s=ct, **jit_stats
        ))

    # --- Fused MatMul + Add + ReLU (tests fusion potential) ---
    print(f"\n--- Fused MatMul+Add+ReLU (fusion test) ---")

    class FusedBlock(nn.Module):
        def forward(self, x, w, bias):
            return torch.relu(torch.matmul(x, w) + bias)

    a = torch.randn(4, 128, 768, device="cuda")
    w = torch.randn(768, 3072, device="cuda")
    bias = torch.randn(3072, device="cuda")

    fused = FusedBlock().cuda().eval()
    stats = benchmark_latency(lambda: fused(a, w, bias))
    print(f"  Eager:      {stats['mean_ms']:.4f} ± {stats['std_ms']:.4f} ms")
    results.append(BenchmarkResult(
        experiment="DVM_baseline", mode="eager", model_or_op="MatMul+Add+ReLU",
        shape_desc="4x128x768 @ 768x3072 + bias + relu", warmup_iters=10,
        measure_iters=100, **stats
    ))

    traced, ct = measure_compile_time(lambda: torch.jit.trace(fused, (a, w, bias)))
    jit_stats = benchmark_latency(lambda: traced(a, w, bias))
    speedup = stats["mean_ms"] / jit_stats["mean_ms"] if jit_stats["mean_ms"] > 0 else 0
    print(f"  JIT:        {jit_stats['mean_ms']:.4f} ± {jit_stats['std_ms']:.4f} ms  (trace: {ct:.3f}s)")
    print(f"  Speedup:    {speedup:.2f}× (JIT can fuse these ops)")
    results.append(BenchmarkResult(
        experiment="DVM_baseline", mode="jit_trace", model_or_op="MatMul+Add+ReLU",
        shape_desc="4x128x768 @ 768x3072 + bias + relu", warmup_iters=10,
        measure_iters=100, compile_time_s=ct, **jit_stats
    ))

    return results


# ─────────────────────────────────────────────────────────
# Part 2: BERT Full Model Benchmark
# ─────────────────────────────────────────────────────────

def benchmark_bert():
    print("\n" + "=" * 60)
    print("PART 2: BERT-BASE FULL MODEL BENCHMARK")
    print("Comparing: PyTorch eager vs TorchScript JIT")
    print("=" * 60)

    from transformers import BertModel, BertConfig

    config = BertConfig()
    model = BertModel(config).cuda().eval()

    results = []
    seq_lengths = [32, 64, 128, 256]
    batch_sizes = [1, 2, 4]

    for batch in batch_sizes:
        for seq_len in seq_lengths:
            desc = f"batch={batch}, seq={seq_len}"
            print(f"\n--- BERT-base: {desc} ---")

            ids = torch.randint(0, 30522, (batch, seq_len), device="cuda")
            mask = torch.ones(batch, seq_len, dtype=torch.long, device="cuda")
            types = torch.zeros(batch, seq_len, dtype=torch.long, device="cuda")

            try:
                with torch.no_grad():
                    # Eager
                    eager_fn = lambda: model(ids, mask, types)
                    stats = benchmark_latency(eager_fn)
                    mem = measure_peak_memory(eager_fn)
                    print(f"  Eager:  {stats['mean_ms']:>8.2f} ± {stats['std_ms']:.2f} ms | Mem: {mem:.0f} MB")
                    results.append(BenchmarkResult(
                        experiment="DVM_BERT", mode="eager", model_or_op="BERT-base",
                        shape_desc=desc, warmup_iters=10, measure_iters=100,
                        peak_memory_mb=mem, **stats
                    ))

                    # JIT trace
                    traced, ct = measure_compile_time(
                        lambda: torch.jit.trace(model, (ids, mask, types), strict=False)
                    )
                    jit_fn = lambda: traced(ids, mask, types)
                    jit_stats = benchmark_latency(jit_fn)
                    jit_mem = measure_peak_memory(jit_fn)
                    speedup = stats["mean_ms"] / jit_stats["mean_ms"]
                    print(f"  JIT:    {jit_stats['mean_ms']:>8.2f} ± {jit_stats['std_ms']:.2f} ms | Mem: {jit_mem:.0f} MB | Trace: {ct:.1f}s")
                    print(f"  Speedup: {speedup:.2f}×")
                    results.append(BenchmarkResult(
                        experiment="DVM_BERT", mode="jit_trace", model_or_op="BERT-base",
                        shape_desc=desc, warmup_iters=10, measure_iters=100,
                        compile_time_s=ct, peak_memory_mb=jit_mem, **jit_stats
                    ))

            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠️  OOM — skipping")
                results.append(BenchmarkResult(
                    experiment="DVM_BERT", mode="OOM", model_or_op="BERT-base",
                    shape_desc=desc, warmup_iters=0, measure_iters=0,
                    mean_ms=0, std_ms=0, min_ms=0, max_ms=0,
                    notes="OOM on RTX 3050 (4GB VRAM)"
                ))
                torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────
# Part 3: Shape-Change Cost (the problem DVM solves)
# ─────────────────────────────────────────────────────────

def benchmark_shape_change_cost():
    """
    JIT trace is shape-specific. When shapes change, you must re-trace.
    This demonstrates the recompilation problem DVM addresses.
    """
    print("\n" + "=" * 60)
    print("PART 3: SHAPE-CHANGE RECOMPILATION COST")
    print("JIT trace is shape-specific — re-tracing needed per new shape")
    print("This is exactly the problem DVM solves with its bytecode VM")
    print("=" * 60)

    from transformers import BertModel, BertConfig
    import time

    config = BertConfig()
    model = BertModel(config).cuda().eval()
    results = []

    seq_lengths = [32, 48, 64, 96, 128, 160, 192, 224, 256]

    for seq_len in seq_lengths:
        ids = torch.randint(0, 30522, (1, seq_len), device="cuda")
        mask = torch.ones(1, seq_len, dtype=torch.long, device="cuda")
        types = torch.zeros(1, seq_len, dtype=torch.long, device="cuda")

        try:
            with torch.no_grad():
                # Measure trace (compile) time for this specific shape
                torch.cuda.synchronize()
                start = time.perf_counter()
                traced = torch.jit.trace(model, (ids, mask, types), strict=False)
                _ = traced(ids, mask, types)
                torch.cuda.synchronize()
                trace_time_ms = (time.perf_counter() - start) * 1000

                # Measure actual inference after tracing
                inf_stats = benchmark_latency(lambda: traced(ids, mask, types), warmup=5, repeat=50)

                print(f"  seq={seq_len:3d}: trace={trace_time_ms:>8.1f} ms | inference={inf_stats['mean_ms']:.2f} ms | ratio={trace_time_ms/inf_stats['mean_ms']:.0f}×")

                results.append(BenchmarkResult(
                    experiment="shape_change_cost", mode="jit_retrace",
                    model_or_op="BERT-base", shape_desc=f"batch=1, seq={seq_len}",
                    warmup_iters=5, measure_iters=50,
                    compile_time_s=trace_time_ms / 1000,
                    notes=f"Trace time is {trace_time_ms/inf_stats['mean_ms']:.0f}× the inference time",
                    **inf_stats
                ))
        except torch.cuda.OutOfMemoryError:
            print(f"  seq={seq_len}: OOM")
            torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_system_info()
    print("\nNOTE: TorchInductor (torch.compile backend='inductor') requires")
    print("Linux + Triton. On Windows, we use TorchScript JIT as compiled baseline.")
    print("This limitation is itself a reproducibility finding.\n")

    all_results = []
    all_results.extend(benchmark_operators())
    all_results.extend(benchmark_bert())
    all_results.extend(benchmark_shape_change_cost())

    os.makedirs("results", exist_ok=True)
    save_results(all_results, "results/dvm_baselines.csv")

    print("\n" + "=" * 60)
    print("ALL DVM BASELINE BENCHMARKS COMPLETE")
    print("Results saved to results/dvm_baselines.csv")
    print("=" * 60)
