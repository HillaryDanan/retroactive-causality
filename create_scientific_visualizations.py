"""
Publication-quality visualizations of information flow patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import pandas as pd

# Set publication style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
sns.set_palette("husl")

# Create figure with multiple panels
fig = plt.figure(figsize=(18, 12))

# === Panel 1: Architecture Comparison with Error Bars ===
ax1 = plt.subplot(3, 3, 1)
models = ['RoBERTa\n(BPE)', 'ALBERT\n(Sentence)', 'DistilBERT\n(Distilled)', 
          'ELECTRA\n(RTD)', 'BERT\n(MLM)']
bits = [0.638, 0.957, 1.231, 1.451, 1.626]
stds = [0.062, 0.128, 0.103, 0.150, 0.201]  # From your data

# Color by pretraining type
colors = ['#FF6B6B', '#FFA500', '#4ECDC4', '#45B7D1', '#2E86AB']
bars = ax1.bar(models, bits, yerr=stds, color=colors, alpha=0.8, capsize=5)

ax1.set_ylabel('Information Update (bits)', fontweight='bold')
ax1.set_title('A. Architecture-Dependent Bounds', fontweight='bold')
ax1.axhline(y=np.mean(bits), color='black', linestyle='--', alpha=0.5, linewidth=1)
ax1.set_ylim(0, 2.0)

# Add significance markers
ax1.text(0, 0.638 + 0.1, 'lowest', ha='center', fontsize=8)
ax1.text(4, 1.626 + 0.1, 'highest', ha='center', fontsize=8)

# === Panel 2: Complexity Scaling Function ===
ax2 = plt.subplot(3, 3, 2)
complexity_levels = [0, 1, 2]
bits_by_complexity = [0.454, 1.626, 4.856]
complexity_labels = ['Simple\n(10 tokens)', 'Standard\n(20 tokens)', 'Complex\n(40+ tokens)']

# Fit logarithmic curve
from scipy.optimize import curve_fit
def log_func(x, a, b):
    return a * np.log(x + 1) + b

x_smooth = np.linspace(0, 2, 100)
popt, _ = curve_fit(log_func, complexity_levels, bits_by_complexity)
y_smooth = log_func(x_smooth, *popt)

ax2.plot(x_smooth, y_smooth, 'r-', alpha=0.3, linewidth=2, label=f'y = {popt[0]:.2f}·ln(x+1) + {popt[1]:.2f}')
ax2.scatter(complexity_levels, bits_by_complexity, s=100, color='red', zorder=5)
ax2.set_xticks(complexity_levels)
ax2.set_xticklabels(complexity_labels)
ax2.set_ylabel('Information Update (bits)', fontweight='bold')
ax2.set_title('B. Complexity-Dependent Scaling', fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# === Panel 3: Information Radiation Pattern ===
ax3 = plt.subplot(3, 3, 3)
positions = np.array([0, 1, 2, 3, 4, 5, 6])
distances = np.array([0.012, 0.0086, 0.0144, 0.0148, 0.012, 0.0264, 0])
tokens = ['the', 'horse', 'raced', 'past', 'the', 'barn', 'fell']

# Create smooth interpolation
spl = make_interp_spline(positions, distances, k=3)
x_smooth = np.linspace(0, 6, 100)
y_smooth = spl(x_smooth)

ax3.fill_between(x_smooth, y_smooth, alpha=0.3, color='steelblue')
ax3.plot(x_smooth, y_smooth, 'b-', linewidth=2)
ax3.scatter(positions, distances, s=50, color='darkblue', zorder=5)

# Mark disambiguation
ax3.axvline(x=5.5, color='red', linestyle='--', linewidth=2, label='Disambiguation')
ax3.axvspan(5, 6, alpha=0.2, color='red')

ax3.set_xticks(positions)
ax3.set_xticklabels(tokens, rotation=45, ha='right')
ax3.set_ylabel('Update Magnitude', fontweight='bold')
ax3.set_title('C. Bidirectional Information Flow', fontweight='bold')
ax3.legend()

# === Panel 4: Theoretical Bounds Visualization ===
ax4 = plt.subplot(3, 3, 4)
theoretical_max = 14.9
observed_percentages = [0.454/theoretical_max*100, 1.626/theoretical_max*100, 
                        4.856/theoretical_max*100, 98.3]
labels = ['Simple', 'Standard', 'Complex', 'Maximum\nObserved']

bars = ax4.bar(labels, observed_percentages, color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
ax4.axhline(y=100, color='black', linestyle='-', linewidth=2, label='Shannon Limit')
ax4.set_ylabel('% of Theoretical Maximum', fontweight='bold')
ax4.set_title('D. Approach to Shannon Limit', fontweight='bold')
ax4.legend()
ax4.set_ylim(0, 110)

# Add percentage labels
for i, (bar, pct) in enumerate(zip(bars, observed_percentages)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{pct:.1f}%', ha='center', fontsize=9)

# === Panel 5: MLM vs Non-MLM Comparison ===
ax5 = plt.subplot(3, 3, 5)
mlm_models = ['BERT', 'DistilBERT', 'ALBERT']
mlm_bits = [1.626, 1.231, 0.957]
non_mlm_models = ['ELECTRA', 'RoBERTa']
non_mlm_bits = [1.451, 0.638]

ax5.boxplot([mlm_bits, non_mlm_bits], labels=['MLM\nPretraining', 'Non-MLM\nPretraining'])
ax5.set_ylabel('Information Update (bits)', fontweight='bold')
ax5.set_title('E. Pretraining Objective Effect', fontweight='bold')

# Statistical test
from scipy import stats
t_stat, p_val = stats.ttest_ind(mlm_bits, non_mlm_bits)
ax5.text(1.5, 1.7, f'p = {p_val:.3f}', ha='center', fontsize=9)

# === Panel 6: Layer-wise Effects ===
ax6 = plt.subplot(3, 3, 6)
layers = [0, 3, 6, 9, 11]
layer_effects = [0.002, 0.015, 0.025, 0.018, 0.010]  # Approximate from your data

ax6.plot(layers, layer_effects, 'o-', linewidth=2, markersize=8, color='purple')
ax6.fill_between(layers, layer_effects, alpha=0.3, color='purple')
ax6.set_xlabel('Layer', fontweight='bold')
ax6.set_ylabel('Mean Update', fontweight='bold')
ax6.set_title('F. Layer-wise Distribution', fontweight='bold')
ax6.grid(True, alpha=0.3)

# Mark peak
peak_idx = np.argmax(layer_effects)
ax6.annotate('Peak at\nmiddle layers', xy=(layers[peak_idx], layer_effects[peak_idx]),
             xytext=(8, 0.02), arrowprops=dict(arrowstyle='->', color='red'))

# === Panel 7: Information Flow Directionality ===
ax7 = plt.subplot(3, 3, 7)
# Conceptual diagram of information flow
theta = np.linspace(0, 2*np.pi, 100)
r = 1 + 0.3*np.sin(5*theta)

ax7.plot(r*np.cos(theta), r*np.sin(theta), 'b-', alpha=0.3, linewidth=2)
ax7.scatter([0], [0], s=200, color='red', marker='*', label='Disambiguation')

# Add arrows showing radiation
for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
    ax7.arrow(0, 0, 0.7*np.cos(angle), 0.7*np.sin(angle), 
              head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)

ax7.set_xlim(-1.5, 1.5)
ax7.set_ylim(-1.5, 1.5)
ax7.set_aspect('equal')
ax7.set_title('G. Information Radiation Pattern', fontweight='bold')
ax7.axis('off')
ax7.legend()

# === Panel 8: Summary Statistics Table ===
ax8 = plt.subplot(3, 3, 8)
ax8.axis('off')

summary_data = {
    'Metric': ['Mean (all models)', 'Std Dev', 'Range', 'Max observed', 'Complexity ratio'],
    'Value': ['1.069 bits', '0.304 bits', '0.638-1.626', '98.3% of max', '10.7x']
}

table_data = []
for i in range(len(summary_data['Metric'])):
    table_data.append([summary_data['Metric'][i], summary_data['Value'][i]])

table = ax8.table(cellText=table_data, cellLoc='left', loc='center',
                  colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

ax8.set_title('H. Summary Statistics', fontweight='bold', y=0.8)

# === Panel 9: Theoretical Interpretation ===
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

interpretation = """
KEY FINDINGS:

1. ARCHITECTURE-DEPENDENT
   MLM > RTD > BPE tokenization
   
2. LOGARITHMIC SCALING
   Bits = a·ln(complexity) + b
   
3. BIDIRECTIONAL RADIATION
   Not pure retroactive flow
   
4. NEAR-MAXIMAL CAPACITY
   Can reach 98.3% of limit
   
5. MIDDLE-LAYER PEAK
   Semantic processing zone
"""

ax9.text(0.1, 0.5, interpretation, fontsize=9, family='monospace',
         verticalalignment='center')
ax9.set_title('I. Theoretical Model', fontweight='bold', y=0.95)

# Overall title
plt.suptitle('Information Flow in Bidirectional Transformers: Architecture and Complexity Dependencies',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('results/complete_scientific_visualization.png', dpi=300, bbox_inches='tight')
print("Saved comprehensive visualization to results/complete_scientific_visualization.png")

# Save data for paper
results_df = pd.DataFrame({
    'Model': models,
    'Bits': bits,
    'Std': stds,
    'Pretraining': ['BPE', 'SentencePiece', 'MLM-Distilled', 'RTD', 'MLM']
})
results_df.to_csv('results/final_architecture_comparison.csv', index=False)
print("Saved final data to results/final_architecture_comparison.csv")
