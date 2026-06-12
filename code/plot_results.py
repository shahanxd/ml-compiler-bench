"""
Plot generation for DVM baseline reproduction results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

os.makedirs("results/plots", exist_ok=True)

df = pd.read_csv("results/dvm_baselines.csv")

# ─────────────────────────────────────────────────────────
# Plot 1: Operator Latency Comparison (eager vs JIT)
# ─────────────────────────────────────────────────────────

ops = df[df['experiment'] == 'DVM_baseline']
ops_eager = ops[ops['mode'] == 'eager']
ops_jit = ops[ops['mode'] == 'jit_trace']

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(ops_eager))
width = 0.35

bars1 = ax.bar(x - width/2, ops_eager['mean_ms'].values, width, 
               label='PyTorch Eager', color='#4A90D9', alpha=0.85,
               yerr=ops_eager['std_ms'].values, capsize=3)
bars2 = ax.bar(x + width/2, ops_jit['mean_ms'].values, width,
               label='TorchScript JIT', color='#E8704A', alpha=0.85,
               yerr=ops_jit['std_ms'].values, capsize=3)

ax.set_xlabel('Operator + Shape')
ax.set_ylabel('Latency (ms)')
ax.set_title('DVM Baseline: Individual Operator Latency — Eager vs JIT\n(RTX 3050 Laptop GPU, CUDA 12.1, PyTorch 2.5.1)')
ax.set_xticks(x)
labels = [f"{r['model_or_op']}\n{r['shape_desc'][:20]}" for _, r in ops_eager.iterrows()]
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/01_operator_latency.png", dpi=300)
plt.close()
print("Saved: 01_operator_latency.png")


# ─────────────────────────────────────────────────────────
# Plot 2: BERT Latency vs Sequence Length (by batch size)
# ─────────────────────────────────────────────────────────

bert = df[df['experiment'] == 'DVM_BERT']

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

for i, batch in enumerate([1, 2, 4]):
    ax = axes[i]
    b_eager = bert[(bert['mode'] == 'eager') & (bert['shape_desc'].str.contains(f'batch={batch},'))]
    b_jit = bert[(bert['mode'] == 'jit_trace') & (bert['shape_desc'].str.contains(f'batch={batch},'))]
    
    seq_lens = [32, 64, 128, 256]
    
    ax.plot(seq_lens[:len(b_eager)], b_eager['mean_ms'].values, 'o-', color='#4A90D9', 
            label='Eager', linewidth=2, markersize=6)
    ax.fill_between(seq_lens[:len(b_eager)], 
                     b_eager['mean_ms'].values - b_eager['std_ms'].values,
                     b_eager['mean_ms'].values + b_eager['std_ms'].values,
                     alpha=0.15, color='#4A90D9')
    
    ax.plot(seq_lens[:len(b_jit)], b_jit['mean_ms'].values, 's-', color='#E8704A',
            label='JIT Trace', linewidth=2, markersize=6)
    ax.fill_between(seq_lens[:len(b_jit)],
                     b_jit['mean_ms'].values - b_jit['std_ms'].values,
                     b_jit['mean_ms'].values + b_jit['std_ms'].values,
                     alpha=0.15, color='#E8704A')
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'Batch Size = {batch}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle('BERT-base Inference Latency vs Sequence Length\n(RTX 3050 Laptop GPU)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("results/plots/02_bert_latency_vs_seqlen.png", dpi=300)
plt.close()
print("Saved: 02_bert_latency_vs_seqlen.png")


# ─────────────────────────────────────────────────────────
# Plot 3: Recompilation Cost — the KEY finding
# ─────────────────────────────────────────────────────────

recomp = df[df['experiment'] == 'shape_change_cost']

fig, ax1 = plt.subplots(figsize=(10, 6))

seq_lens = [int(s.split('seq=')[1]) for s in recomp['shape_desc']]
trace_times = recomp['compile_time_s'].values * 1000  # to ms
inf_times = recomp['mean_ms'].values

color1 = '#D9534F'
color2 = '#5CB85C'

ax1.bar(range(len(seq_lens)), trace_times, color=color1, alpha=0.75, label='Trace/Compile Time')
ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('Trace Time (ms)', color=color1)
ax1.set_xticks(range(len(seq_lens)))
ax1.set_xticklabels(seq_lens)

ax2 = ax1.twinx()
ax2.plot(range(len(seq_lens)), inf_times, 's-', color=color2, linewidth=2, 
         markersize=8, label='Inference Time', zorder=5)
ax2.set_ylabel('Inference Time (ms)', color=color2)

# Add ratio annotation
for i, (t, inf) in enumerate(zip(trace_times, inf_times)):
    ratio = t / inf
    ax1.annotate(f'{ratio:.0f}×', xy=(i, t), ha='center', va='bottom', 
                fontsize=8, fontweight='bold', color='#333')

ax1.set_title('Shape-Change Recompilation Cost: BERT-base (batch=1)\nTrace time is 70–220× the inference time — This is the problem DVM solves',
              fontsize=11, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig("results/plots/03_recompilation_cost.png", dpi=300)
plt.close()
print("Saved: 03_recompilation_cost.png")


# ─────────────────────────────────────────────────────────
# Plot 4: BERT Speedup heatmap (JIT/Eager)
# ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

batch_sizes = [1, 2, 4]
seq_lengths = [32, 64, 128, 256]
speedup_matrix = np.ones((len(batch_sizes), len(seq_lengths)))

for i, b in enumerate(batch_sizes):
    for j, s in enumerate(seq_lengths):
        desc = f"batch={b}, seq={s}"
        eager_row = bert[(bert['mode'] == 'eager') & (bert['shape_desc'] == desc)]
        jit_row = bert[(bert['mode'] == 'jit_trace') & (bert['shape_desc'] == desc)]
        if len(eager_row) > 0 and len(jit_row) > 0:
            speedup_matrix[i, j] = eager_row['mean_ms'].values[0] / jit_row['mean_ms'].values[0]

im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=0.9, vmax=1.6)
ax.set_xticks(range(len(seq_lengths)))
ax.set_xticklabels(seq_lengths)
ax.set_yticks(range(len(batch_sizes)))
ax.set_yticklabels(batch_sizes)
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Batch Size')
ax.set_title('BERT-base JIT Speedup over Eager (×)\n>1.0 = JIT faster, <1.0 = Eager faster')

for i in range(len(batch_sizes)):
    for j in range(len(seq_lengths)):
        ax.text(j, i, f"{speedup_matrix[i,j]:.2f}×", ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='white' if speedup_matrix[i,j] > 1.3 else 'black')

plt.colorbar(im, ax=ax, label='Speedup (×)')
plt.tight_layout()
plt.savefig("results/plots/04_bert_speedup_heatmap.png", dpi=300)
plt.close()
print("Saved: 04_bert_speedup_heatmap.png")

print("\nAll plots generated successfully!")
