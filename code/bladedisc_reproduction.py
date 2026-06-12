"""
BladeDISC Paper Reproduction
==============================
Paper: BladeDISC: Optimizing Dynamic Shape ML Workloads via Compiler Approach (SIGMOD 2024)

This script runs INSIDE the BladeDISC Docker container.
It benchmarks: PyTorch eager vs TorchScript JIT vs BladeDISC (torch_blade)
on BERT-base with multiple dynamic shapes.
"""

import torch
import torch.nn as nn
import time
import gc
import csv
import os
import sys
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Result:
    experiment: str
    mode: str
    model: str
    shape: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    compile_s: Optional[float] = None
    peak_mem_mb: Optional[float] = None
    max_diff: Optional[float] = None
    notes: str = ""


def bench(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    t = torch.tensor(times)
    return {"mean_ms": t.mean().item(), "std_ms": t.std().item(),
            "min_ms": t.min().item(), "max_ms": t.max().item()}


def peak_mem(fn):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def save(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BLADEDISC REPRODUCTION BENCHMARK")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Check if torch_blade is available
    try:
        import torch_blade
        blade_ver = getattr(torch_blade, '__version__', None) or getattr(torch_blade, 'version', 'available')
        print(f"torch_blade: {blade_ver}")
        HAS_BLADE = True
    except ImportError:
        print("torch_blade: NOT AVAILABLE — running baselines only")
        HAS_BLADE = False

    from transformers import BertModel, BertConfig

    config = BertConfig()  # BERT-base: 12 layers, 768 hidden, 12 heads
    model = BertModel(config).cuda().eval()

    results = []
    seq_lengths = [32, 64, 128, 256]
    batch_sizes = [1, 2]  # Conservative for 4GB VRAM

    # ── Part 1: Baselines ──────────────────────────────
    print("\n--- PART 1: BASELINE MEASUREMENTS ---")

    for batch in batch_sizes:
        for seq in seq_lengths:
            shape = f"batch={batch}, seq={seq}"
            print(f"\n  {shape}")
            ids = torch.randint(0, 30522, (batch, seq), device="cuda")
            mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
            types = torch.zeros(batch, seq, dtype=torch.long, device="cuda")

            try:
                with torch.no_grad():
                    # Eager
                    fn = lambda: model(ids, mask, types)
                    s = bench(fn)
                    m = peak_mem(fn)
                    print(f"    Eager:  {s['mean_ms']:.2f} ± {s['std_ms']:.2f} ms | Mem: {m:.0f} MB")
                    results.append(Result("bladedisc", "eager", "BERT-base", shape, 
                                         peak_mem_mb=m, **s))

                    # JIT
                    traced = torch.jit.trace(model, (ids, mask, types), strict=False)
                    jfn = lambda: traced(ids, mask, types)
                    js = bench(jfn)
                    jm = peak_mem(jfn)
                    print(f"    JIT:    {js['mean_ms']:.2f} ± {js['std_ms']:.2f} ms | Mem: {jm:.0f} MB")
                    results.append(Result("bladedisc", "jit", "BERT-base", shape,
                                         peak_mem_mb=jm, **js))

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM — skipping")
                torch.cuda.empty_cache()

    # ── Part 2: BladeDISC optimization ──────────────────
    if HAS_BLADE:
        print("\n--- PART 2: BLADEDISC OPTIMIZATION ---")

        # Compile with example shape
        compile_ids = torch.randint(0, 30522, (1, 64), device="cuda")
        compile_mask = torch.ones(1, 64, dtype=torch.long, device="cuda")
        compile_types = torch.zeros(1, 64, dtype=torch.long, device="cuda")

        with torch.no_grad():
            t0 = time.perf_counter()
            opt_model = torch_blade.optimize(
                model,
                allow_tracing=True,
                model_inputs=(compile_ids, compile_mask, compile_types)
            )
            compile_time = time.perf_counter() - t0
            print(f"\n  BladeDISC compile time: {compile_time:.2f}s")

        # ── Part 3: Dynamic shape test (CORE CLAIM) ──────
        print("\n--- PART 3: DYNAMIC SHAPE TEST (compile once, run any shape) ---")
        print("  Model was compiled with batch=1, seq=64")
        print("  Now testing with DIFFERENT shapes:\n")

        for batch in batch_sizes:
            for seq in seq_lengths:
                shape = f"batch={batch}, seq={seq}"
                ids = torch.randint(0, 30522, (batch, seq), device="cuda")
                mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
                types = torch.zeros(batch, seq, dtype=torch.long, device="cuda")

                try:
                    with torch.no_grad():
                        # BladeDISC inference
                        bfn = lambda: opt_model(ids, mask, types)
                        bs = bench(bfn)
                        bm = peak_mem(bfn)

                        # Correctness check — BladeDISC returns a dict, eager returns BaseModelOutput
                        out_eager = model(ids, mask, types)
                        out_blade = opt_model(ids, mask, types)

                        # Get the first tensor output from each (last_hidden_state)
                        def get_hidden(out):
                            if hasattr(out, 'last_hidden_state'):
                                return out.last_hidden_state
                            elif isinstance(out, dict) and 'last_hidden_state' in out:
                                return out['last_hidden_state']
                            elif isinstance(out, (tuple, list)):
                                return out[0]
                            else:
                                return list(out.values())[0] if isinstance(out, dict) else out

                        eager_hidden = get_hidden(out_eager)
                        blade_hidden = get_hidden(out_blade)
                        diff = (eager_hidden - blade_hidden).abs().max().item()

                        # Compare to eager baseline
                        eager_r = [r for r in results if r.mode=="eager" and r.shape==shape]
                        speedup = eager_r[0].mean_ms / bs['mean_ms'] if eager_r else 0

                        status = "✓" if diff < 1e-2 else "⚠️ SIGNIFICANT DIFF"
                        print(f"    {shape}: {bs['mean_ms']:.2f} ms | speedup={speedup:.2f}× | max_diff={diff:.6f} {status}")

                        results.append(Result("bladedisc", "bladedisc", "BERT-base", shape,
                                             compile_s=compile_time, peak_mem_mb=bm, max_diff=diff,
                                             notes=f"speedup={speedup:.2f}x vs eager", **bs))

                except Exception as e:
                    print(f"    {shape}: FAILED — {e}")
                    results.append(Result("bladedisc", "bladedisc_error", "BERT-base", shape,
                                         mean_ms=0, std_ms=0, min_ms=0, max_ms=0,
                                         notes=str(e)))
                    torch.cuda.empty_cache()

        # ── Part 4: Compilation overhead comparison ──────
        print("\n--- PART 4: COMPILATION OVERHEAD ---")
        print(f"  BladeDISC compile time: {compile_time:.2f}s (ONE TIME)")

        # JIT re-trace cost per shape
        retrace_times = []
        for seq in [32, 64, 128, 256]:
            ids = torch.randint(0, 30522, (1, seq), device="cuda")
            mask = torch.ones(1, seq, dtype=torch.long, device="cuda")
            types = torch.zeros(1, seq, dtype=torch.long, device="cuda")
            with torch.no_grad():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                tr = torch.jit.trace(model, (ids, mask, types), strict=False)
                _ = tr(ids, mask, types)
                torch.cuda.synchronize()
                rt = (time.perf_counter() - t0) * 1000
                retrace_times.append(rt)
                print(f"  JIT retrace seq={seq}: {rt:.0f} ms")

        print(f"\n  JIT: Must retrace for EACH new shape (~{sum(retrace_times)/len(retrace_times):.0f} ms per shape)")
        print(f"  BladeDISC: Compile ONCE ({compile_time:.1f}s), then run ANY shape with no recompilation")
        print(f"  For {len(seq_lengths)} shapes: JIT total = {sum(retrace_times):.0f} ms, BladeDISC = {compile_time*1000:.0f} ms")

    # Save all results
    save(results, "/workspace/results/bladedisc_results.csv")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
