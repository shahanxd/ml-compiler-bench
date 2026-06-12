# ML Compiler Benchmarks: DVM & BladeDISC Reproduction

Reproduction of two ML compiler papers focused on dynamic shape compilation:

- **DVM** - Bytecode VM approach for dynamic tensor computation ([arXiv 2603.24239v2](https://arxiv.org/abs/2603.24239))
- **BladeDISC** - Compiler-based symbolic shape optimization ([SIGMOD 2024](https://dl.acm.org/doi/10.1145/3626246.3653395))

## Results Summary

| Paper | What was tested | Key result |
|-------|----------------|------------|
| DVM | Baselines (eager, JIT) on BERT-base | Recompilation costs 70-220x more than inference |
| BladeDISC | Full reproduction via Docker | Compile once, run 8 shapes: 1.09-1.56x speedup, numerically correct |

## Structure

```
├── reproduction_report.pdf    # Full report with embedded figures
├── code/                      # All benchmark scripts
├── results/                   # Raw CSV measurements (81 total)
└── plots/                     # 8 publication-quality figures
```

## Hardware

- NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- Windows 11 + WSL2 Ubuntu 22.04 + Docker (for BladeDISC)

## How to reproduce

**DVM baselines** (Windows, needs PyTorch + transformers):
```
python code/dvm_baseline_benchmark.py
python code/plot_results.py
```

**BladeDISC** (needs WSL2 + Docker + NVIDIA Container Toolkit):
```
docker pull bladedisc/bladedisc:latest-runtime-torch-2.0.1-cu118
docker run --rm --gpus all -v $(pwd):/scripts:ro bladedisc/bladedisc:latest-runtime-torch-2.0.1-cu118 \
  bash -c "pip install transformers && python3 /scripts/code/bladedisc_reproduction.py"
```
