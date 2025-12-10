"""
Generate comprehensive model comparison chart
Shows Accuracy, Precision, Recall, and F1-Score for all 4 models
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent.parent / "charts"
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("MODEL COMPARISON CHART GENERATOR")
print("="*70)

# Performance metrics from confusion matrix results
models = ['CNN', 'LSTM', 'BERT', 'Hybrid']
accuracy = [96.69, 96.24, 99.43, 95.61]
precision = [95.58, 94.83, 99.09, 92.76]
recall = [95.58, 95.12, 99.39, 95.73]
f1_score = [95.58, 94.98, 99.24, 94.22]

# Create figure with multiple visualizations
fig = plt.figure(figsize=(20, 12))

# ============================================================================
# Chart 1: Grouped Bar Chart (All Metrics Side by Side)
# ============================================================================
ax1 = plt.subplot(2, 2, 1)

x = np.arange(len(models))
width = 0.2

bars1 = ax1.bar(x - 1.5*width, accuracy, width, label='Accuracy', 
                color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x - 0.5*width, precision, width, label='Precision', 
                color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax1.bar(x + 0.5*width, recall, width, label='Recall', 
                color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1.5)
bars4 = ax1.bar(x + 1.5*width, f1_score, width, label='F1-Score', 
                color='#C73E1D', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison - All Metrics', 
              fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax1.set_ylim([90, 101])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# ============================================================================
# Chart 2: Radar/Spider Chart (Comprehensive View)
# ============================================================================
ax2 = plt.subplot(2, 2, 2, projection='polar')

categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
N = len(categories)

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Colors for each model
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
markers = ['o', 's', '^', 'D']

for idx, model in enumerate(models):
    values = [accuracy[idx], precision[idx], recall[idx], f1_score[idx]]
    values += values[:1]
    
    ax2.plot(angles, values, 'o-', linewidth=2.5, label=model, 
             color=colors[idx], marker=markers[idx], markersize=8)
    ax2.fill(angles, values, alpha=0.15, color=colors[idx])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax2.set_ylim(90, 100)
ax2.set_yticks([90, 92, 94, 96, 98, 100])
ax2.set_yticklabels(['90%', '92%', '94%', '96%', '98%', '100%'], fontsize=9)
ax2.set_title('Radar Chart - Performance Profile', 
              fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax2.grid(True, alpha=0.3)

# ============================================================================
# Chart 3: Line Chart (Metric Trends)
# ============================================================================
ax3 = plt.subplot(2, 2, 3)

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_line = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for i, (metric_data, metric_name, color) in enumerate(zip(
    [accuracy, precision, recall, f1_score], 
    metric_names, 
    colors_line)):
    ax3.plot(models, metric_data, marker='o', linewidth=2.5, 
             markersize=10, label=metric_name, color=color, alpha=0.8)
    
    # Add value labels
    for j, val in enumerate(metric_data):
        ax3.text(j, val + 0.3, f'{val:.1f}%', ha='center', 
                fontsize=9, fontweight='bold', color=color)

ax3.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
ax3.set_title('Performance Trends Across Models', 
              fontsize=14, fontweight='bold')
ax3.legend(loc='lower left', fontsize=10, framealpha=0.9)
ax3.set_ylim([90, 101])
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xticklabels(models, fontsize=11, fontweight='bold')

# ============================================================================
# Chart 4: Heatmap (Performance Matrix)
# ============================================================================
ax4 = plt.subplot(2, 2, 4)

# Prepare data matrix
data_matrix = np.array([accuracy, precision, recall, f1_score])

# Create heatmap
im = ax4.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=92, vmax=100)

# Add colorbar
cbar = plt.colorbar(im, ax=ax4)
cbar.set_label('Score (%)', fontsize=11, fontweight='bold')

# Set ticks
ax4.set_xticks(np.arange(len(models)))
ax4.set_yticks(np.arange(len(metric_names)))
ax4.set_xticklabels(models, fontsize=11, fontweight='bold')
ax4.set_yticklabels(metric_names, fontsize=11, fontweight='bold')

# Add text annotations
for i in range(len(metric_names)):
    for j in range(len(models)):
        text = ax4.text(j, i, f'{data_matrix[i, j]:.1f}%',
                       ha="center", va="center", color="black", 
                       fontsize=11, fontweight='bold')

ax4.set_title('Performance Heatmap (Higher is Better)', 
              fontsize=14, fontweight='bold')

# ============================================================================
# Add overall figure title and summary
# ============================================================================
fig.suptitle('Comprehensive Model Performance Comparison\nPhishing Email Detection System', 
             fontsize=18, fontweight='bold', y=0.98)

# Add summary statistics box
summary_text = f"""
PERFORMANCE SUMMARY
{'='*50}
Best Accuracy:  BERT (99.43%)
Best Precision: BERT (99.09%)
Best Recall:    BERT (99.39%) / Hybrid (95.73%)
Best F1-Score:  BERT (99.24%)

Fastest Model:  CNN (0.65 ms/email)
Most Balanced:  CNN (95.58% across all metrics)

Test Set: 1,754 emails (1,098 legitimate, 656 phishing)
"""

fig.text(0.5, -0.02, summary_text, ha='center', va='top',
         fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# Save figure
output_path = output_dir / "model_comparison_chart.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Model comparison chart saved: {output_path}")

print("\n" + "="*70)
print("CHART GENERATED SUCCESSFULLY!")
print("="*70)
print(f"\nOutput: {output_path}")
print("\nChart includes:")
print("  1. Grouped Bar Chart - Side-by-side comparison")
print("  2. Radar Chart - Performance profile visualization")
print("  3. Line Chart - Metric trends across models")
print("  4. Heatmap - Color-coded performance matrix")
print("\n" + "="*70)

# Also create a simple single-chart version
print("\nGenerating simplified single-chart version...")

fig2, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(models))
width = 0.2

bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', 
               color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=2)
bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', 
               color='#A23B72', alpha=0.85, edgecolor='black', linewidth=2)
bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', 
               color='#F18F01', alpha=0.85, edgecolor='black', linewidth=2)
bars4 = ax.bar(x + 1.5*width, f1_score, width, label='F1-Score', 
               color='#C73E1D', alpha=0.85, edgecolor='black', linewidth=2)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison\nAccuracy, Precision, Recall, F1-Score', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=12, framealpha=0.95, 
          edgecolor='black', fancybox=True)
ax.set_ylim([90, 101])
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)

# Add summary box
summary_box = f'Test Set: 1,754 emails | Best Overall: BERT (99.43%) | Fastest: CNN (0.65 ms)'
ax.text(0.5, -0.12, summary_box, transform=ax.transAxes,
        ha='center', va='top', fontsize=11, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save simple version
simple_path = output_dir / "model_comparison_simple.png"
plt.savefig(simple_path, dpi=300, bbox_inches='tight')
print(f"✓ Simple comparison chart saved: {simple_path}")

print("\n" + "="*70)
print("BOTH VERSIONS GENERATED!")
print("="*70)
print("\n1. Comprehensive (4 charts): model_comparison_chart.png")
print("2. Simple (single chart): model_comparison_simple.png")
print("\nBoth charts are 300 DPI, ready for presentations/reports!")
print("="*70)

plt.close('all')
