"""
Comprehensive visualization of ScienceFair model results
Analyzes RealTest1.json and creates publication-quality charts
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
results_file = Path('results/RealTest1.json')
with open(results_file, 'r') as f:
    results = json.load(f)

print(f"Loaded {len(results)} experiment results")

# Convert to DataFrame
df = pd.DataFrame([
    {
        'stock': r['stock'],
        'horizon': r['horizon'],
        'test': r['test'],
        'model': r['model'],
        'rmse': r['metrics']['rmse'],
        'mae': r['metrics']['mae'],
        'r2': r['metrics']['r2'],
        'accuracy': r['metrics']['directional_accuracy'],
        'training_time': r.get('training_time', 0)
    }
    for r in results if r['status'] == 'success'
])

print(f"\nDataFrame shape: {df.shape}")
print(f"Stocks: {df['stock'].unique()}")
print(f"Models: {df['model'].unique()}")
print(f"Horizons: {df['horizon'].unique()}")

# Create output directory
output_dir = Path('results/visualizations')
output_dir.mkdir(exist_ok=True)

# ============================================================================
# 1. Model Comparison - Overall Performance
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ANN vs QINN: Overall Performance Comparison', fontsize=16, fontweight='bold')

# RMSE comparison
ax = axes[0, 0]
model_rmse = df.groupby('model')['rmse'].mean()
model_rmse.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
ax.set_title('Average RMSE by Model (Lower is Better)', fontweight='bold')
ax.set_ylabel('RMSE')
ax.set_xlabel('Model')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)

# Directional Accuracy
ax = axes[0, 1]
model_acc = df.groupby('model')['accuracy'].mean()
model_acc.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
ax.set_title('Average Directional Accuracy (Higher is Better)', fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Model')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# R² comparison
ax = axes[1, 0]
model_r2 = df.groupby('model')['r2'].mean()
model_r2.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
ax.set_title('Average R² Score', fontweight='bold')
ax.set_ylabel('R²')
ax.set_xlabel('Model')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.grid(axis='y', alpha=0.3)

# Training Time
ax = axes[1, 1]
model_time = df.groupby('model')['training_time'].mean()
model_time.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
ax.set_title('Average Training Time', fontweight='bold')
ax.set_ylabel('Time (seconds)')
ax.set_xlabel('Model')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '01_model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 01_model_comparison.png")
plt.close()

# ============================================================================
# 2. Performance by Horizon
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Performance Across Different Prediction Horizons', fontsize=16, fontweight='bold')

# Accuracy by horizon
ax = axes[0]
horizon_data = df.groupby(['horizon', 'model'])['accuracy'].mean().unstack()
horizon_data.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
ax.set_title('Directional Accuracy by Horizon', fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Horizon')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# RMSE by horizon
ax = axes[1]
horizon_rmse = df.groupby(['horizon', 'model'])['rmse'].mean().unstack()
horizon_rmse.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
ax.set_title('RMSE by Horizon (Lower is Better)', fontweight='bold')
ax.set_ylabel('RMSE')
ax.set_xlabel('Horizon')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '02_performance_by_horizon.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 02_performance_by_horizon.png")
plt.close()

# ============================================================================
# 3. Stock-by-Stock Performance
# ============================================================================
stock_perf = df.groupby(['stock', 'model']).agg({
    'accuracy': 'mean',
    'rmse': 'mean',
    'r2': 'mean'
}).reset_index()

fig, ax = plt.subplots(figsize=(16, 6))
pivot_acc = stock_perf.pivot(index='stock', columns='model', values='accuracy')
pivot_acc.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'], width=0.8)
ax.set_title('Directional Accuracy by Stock (All Horizons)', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Stock')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '03_stock_performance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 03_stock_performance.png")
plt.close()

# ============================================================================
# 4. Heatmap: Accuracy across stocks and models
# ============================================================================
heatmap_data = df.pivot_table(values='accuracy', index='stock', columns='model', aggfunc='mean')

fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=50, 
            cbar_kws={'label': 'Directional Accuracy (%)'}, ax=ax, vmin=30, vmax=70)
ax.set_title('Directional Accuracy Heatmap: Stocks vs Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '04_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 04_accuracy_heatmap.png")
plt.close()

# ============================================================================
# 5. RMSE Distribution
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ann_rmse = df[df['model'] == 'ann']['rmse']
qinn_rmse = df[df['model'] == 'qinn']['rmse']

ax.hist(ann_rmse, bins=15, alpha=0.6, label='ANN', color='#FF6B6B', edgecolor='black')
ax.hist(qinn_rmse, bins=15, alpha=0.6, label='QINN', color='#4ECDC4', edgecolor='black')
ax.set_title('Distribution of RMSE Across All Experiments', fontsize=14, fontweight='bold')
ax.set_xlabel('RMSE')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '05_rmse_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 05_rmse_distribution.png")
plt.close()

# ============================================================================
# 6. Best Performers
# ============================================================================
best_ann = df[df['model'] == 'ann'].nlargest(10, 'accuracy')
best_qinn = df[df['model'] == 'qinn'].nlargest(10, 'accuracy')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Top 10 ANN
best_ann_sorted = best_ann.sort_values('accuracy')
y_pos = np.arange(len(best_ann_sorted))
ax1.barh(y_pos, best_ann_sorted['accuracy'], color='#FF6B6B')
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{row['stock']}_{row['horizon']}" for _, row in best_ann_sorted.iterrows()], fontsize=9)
ax1.set_xlabel('Directional Accuracy (%)')
ax1.set_title('Top 10 ANN Performances', fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Top 10 QINN
best_qinn_sorted = best_qinn.sort_values('accuracy')
y_pos = np.arange(len(best_qinn_sorted))
ax2.barh(y_pos, best_qinn_sorted['accuracy'], color='#4ECDC4')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"{row['stock']}_{row['horizon']}" for _, row in best_qinn_sorted.iterrows()], fontsize=9)
ax2.set_xlabel('Directional Accuracy (%)')
ax2.set_title('Top 10 QINN Performances', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '06_top_performers.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 06_top_performers.png")
plt.close()

# ============================================================================
# 7. Summary Statistics
# ============================================================================
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

summary_stats = {
    'Model': ['ANN', 'QINN'],
    'Avg Accuracy (%)': [
        df[df['model'] == 'ann']['accuracy'].mean(),
        df[df['model'] == 'qinn']['accuracy'].mean()
    ],
    'Avg RMSE': [
        df[df['model'] == 'ann']['rmse'].mean(),
        df[df['model'] == 'qinn']['rmse'].mean()
    ],
    'Avg R²': [
        df[df['model'] == 'ann']['r2'].mean(),
        df[df['model'] == 'qinn']['r2'].mean()
    ],
    'Best Accuracy (%)': [
        df[df['model'] == 'ann']['accuracy'].max(),
        df[df['model'] == 'qinn']['accuracy'].max()
    ],
    'Experiments': [
        len(df[df['model'] == 'ann']),
        len(df[df['model'] == 'qinn'])
    ]
}

ax = fig.add_subplot(gs[0, :])
ax.axis('tight')
ax.axis('off')
summary_df = pd.DataFrame(summary_stats)
table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)
# Header styling
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')
# Alternating row colors
for i in range(1, len(summary_df) + 1):
    for j in range(len(summary_df.columns)):
        table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

# Additional stats
ax = fig.add_subplot(gs[1, 0])
stats_text = f"""
EXPERIMENT OVERVIEW

Total Experiments: {len(df)}
Successful: {len(df)}
Stocks: {df['stock'].nunique()}
Horizons: {df['horizon'].nunique()}

BEST OVERALL
Stock: {df.loc[df['accuracy'].idxmax(), 'stock']}
Horizon: {df.loc[df['accuracy'].idxmax(), 'horizon']}
Model: {df.loc[df['accuracy'].idxmax(), 'model'].upper()}
Accuracy: {df['accuracy'].max():.2f}%
"""
ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', 
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.axis('off')

ax = fig.add_subplot(gs[1, 1])
performance_text = f"""
PERFORMANCE INSIGHTS

ANN Avg Accuracy: {df[df['model']=='ann']['accuracy'].mean():.2f}%
QINN Avg Accuracy: {df[df['model']=='qinn']['accuracy'].mean():.2f}%

ANN Avg RMSE: {df[df['model']=='ann']['rmse'].mean():.6f}
QINN Avg RMSE: {df[df['model']=='qinn']['rmse'].mean():.6f}

Higher Accuracy Horizon: {df.groupby('horizon')['accuracy'].mean().idxmax()}
Lower RMSE Horizon: {df.groupby('horizon')['rmse'].mean().idxmin()}
"""
ax.text(0.1, 0.5, performance_text, fontsize=11, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.axis('off')

# Test configuration
ax = fig.add_subplot(gs[2, :])
test_text = f"""
TEST CONFIGURATION
Long Historical Test: 8 years training (2015-2022) → validation 2023 → test 2024
Prediction Horizons: 1 day, 15 days, 1 month (21 days), 3 months (63 days)
"""
ax.text(0.1, 0.5, test_text, fontsize=11, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax.axis('off')

plt.savefig(output_dir / '07_summary_statistics.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 07_summary_statistics.png")
plt.close()

# ============================================================================
# 8. Create a detailed CSV report
# ============================================================================
report_df = df.copy()
report_df = report_df.sort_values('accuracy', ascending=False)
report_df.to_csv(output_dir / 'detailed_results.csv', index=False)
print("✅ Saved: detailed_results.csv")

print(f"\n{'='*60}")
print(f"All visualizations saved to: {output_dir}")
print(f"{'='*60}")
print("\nFiles created:")
print("  1. 01_model_comparison.png - ANN vs QINN overall performance")
print("  2. 02_performance_by_horizon.png - Performance across horizons")
print("  3. 03_stock_performance.png - Stock-by-stock comparison")
print("  4. 04_accuracy_heatmap.png - Accuracy heatmap")
print("  5. 05_rmse_distribution.png - RMSE distribution")
print("  6. 06_top_performers.png - Best performing experiments")
print("  7. 07_summary_statistics.png - Key statistics and insights")
print("  8. detailed_results.csv - Full results table")
