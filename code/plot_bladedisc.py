import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

df = pd.read_csv('results/bladedisc_results.csv')
os.makedirs('results/plots', exist_ok=True)

# Plot 5: BladeDISC vs Eager vs JIT latency
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, batch in enumerate([1, 2]):
    ax = axes[idx]
    sub = df[df['shape'].str.contains(f'batch={batch},')]

    eager = sub[sub['mode'] == 'eager'].sort_values('shape')
    jit = sub[sub['mode'] == 'jit'].sort_values('shape')
    blade = sub[sub['mode'] == 'bladedisc'].sort_values('shape')

    seqs = [32, 64, 128, 256]
    x = np.arange(len(seqs))
    w = 0.25

    ax.bar(x - w, eager['mean_ms'].values, w, label='Eager', color='#e74c3c', alpha=0.85)
    ax.bar(x, jit['mean_ms'].values, w, label='JIT', color='#3498db', alpha=0.85)
    if len(blade) == len(seqs):
        ax.bar(x + w, blade['mean_ms'].values, w, label='BladeDISC', color='#2ecc71', alpha=0.85)

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'BERT-base Latency, Batch={batch}')
    ax.set_xticks(x)
    ax.set_xticklabels(seqs)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('BladeDISC vs Baselines: BERT-base Inference Latency', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/05_bladedisc_latency_comparison.png', dpi=150, bbox_inches='tight')
print('Saved: 05_bladedisc_latency_comparison.png')

# Plot 6: Speedup heatmap
fig, ax = plt.subplots(figsize=(8, 5))

speedups = []
for batch in [1, 2]:
    row = []
    for seq in [32, 64, 128, 256]:
        shape = f'batch={batch}, seq={seq}'
        eager_ms = df[(df['mode']=='eager') & (df['shape']==shape)]['mean_ms'].values
        blade_ms = df[(df['mode']=='bladedisc') & (df['shape']==shape)]['mean_ms'].values
        if len(eager_ms) > 0 and len(blade_ms) > 0 and blade_ms[0] > 0:
            row.append(eager_ms[0] / blade_ms[0])
        else:
            row.append(1.0)
    speedups.append(row)

speedups = np.array(speedups)
im = ax.imshow(speedups, cmap='RdYlGn', aspect='auto', vmin=0.9, vmax=1.7)
for i in range(2):
    for j in range(4):
        ax.text(j, i, f'{speedups[i,j]:.2f}x', ha='center', va='center', fontsize=14, fontweight='bold')

ax.set_xticks(range(4))
ax.set_xticklabels([32, 64, 128, 256])
ax.set_yticks(range(2))
ax.set_yticklabels(['Batch 1', 'Batch 2'])
ax.set_xlabel('Sequence Length')
plt.colorbar(im, ax=ax, label='Speedup vs Eager')
ax.set_title('BladeDISC Speedup over Eager (compile once, run any shape)', fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/06_bladedisc_speedup_heatmap.png', dpi=150, bbox_inches='tight')
print('Saved: 06_bladedisc_speedup_heatmap.png')

# Plot 7: Numerical accuracy (FIXED)
fig, ax = plt.subplots(figsize=(10, 5))

blade_rows = df[df['mode'] == 'bladedisc'].copy()
blade_rows = blade_rows.dropna(subset=['max_diff'])
print(f"  Plot 7 data rows: {len(blade_rows)}")

shapes = blade_rows['shape'].values
diffs = blade_rows['max_diff'].values
colors = ['#2ecc71' if d < 1e-2 else '#e74c3c' for d in diffs]
bars = ax.bar(range(len(shapes)), diffs * 1e6, color=colors, alpha=0.85, edgecolor='#333', linewidth=0.5)

# Add value labels on each bar
for i, (bar, d) in enumerate(zip(bars, diffs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{d*1e6:.1f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(range(len(shapes)))
ax.set_xticklabels([s.replace('batch=', 'B').replace(', seq=', ' S') for s in shapes], rotation=45, ha='right')
ax.set_ylabel('Max Absolute Difference (x1e-6)')
ax.set_title('BladeDISC Numerical Accuracy vs Eager\n(all green = within tolerance)', fontweight='bold')
ax.set_ylim(0, max(diffs * 1e6) * 1.4)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/07_bladedisc_accuracy.png', dpi=150, bbox_inches='tight')
print('Saved: 07_bladedisc_accuracy.png')

# Plot 8: Compilation cost crossover (FIXED - proper x range)
fig, ax = plt.subplots(figsize=(10, 5))

blade_compile = 437.84
jit_retrace_ms = [2690, 1922, 2414, 1823]
avg_retrace_s = np.mean(jit_retrace_ms) / 1000

n_shapes = np.arange(1, 301)
jit_costs = n_shapes * avg_retrace_s
blade_costs = np.full_like(n_shapes, blade_compile, dtype=float)

ax.plot(n_shapes, jit_costs, '-', color='#3498db', label='JIT (retrace per shape)', linewidth=2)
ax.plot(n_shapes, blade_costs, '--', color='#2ecc71', label='BladeDISC (compile once)', linewidth=2)

ax.fill_between(n_shapes, jit_costs, blade_costs,
                where=jit_costs > blade_costs, alpha=0.12, color='#e74c3c')
ax.fill_between(n_shapes, jit_costs, blade_costs,
                where=jit_costs <= blade_costs, alpha=0.12, color='#2ecc71')

crossover = blade_compile / avg_retrace_s
ax.axvline(x=crossover, color='gray', linestyle=':', alpha=0.7)
ax.annotate(f'Crossover: ~{crossover:.0f} shapes', xy=(crossover, blade_compile),
            fontsize=11, fontweight='bold',
            xytext=(crossover + 15, blade_compile * 1.15),
            arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xlabel('Number of Distinct Shapes')
ax.set_ylabel('Total Compilation Cost (seconds)')
ax.set_title('Compilation Cost: JIT Retrace vs BladeDISC Compile-Once', fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/08_compilation_crossover.png', dpi=150, bbox_inches='tight')
print('Saved: 08_compilation_crossover.png')

print('\nAll plots generated!')
