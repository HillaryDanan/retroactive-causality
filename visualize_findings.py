"""
Visualize the key findings
"""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Architecture comparison
ax = axes[0, 0]
models = ['RoBERTa', 'ALBERT', 'DistilBERT', 'ELECTRA', 'BERT']
bits = [0.638, 0.957, 1.231, 1.451, 1.626]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
bars = ax.bar(models, bits, color=colors)
ax.set_ylabel('Information Update (bits)', fontsize=12)
ax.set_title('Architecture-Dependent Bounds', fontsize=14, fontweight='bold')
ax.axhline(y=1.069, color='black', linestyle='--', alpha=0.5, label='Mean: 1.07 bits')
ax.legend()

# 2. Complexity scaling
ax = axes[0, 1]
complexity = ['Simple', 'Standard', 'Complex']
bits_by_complexity = [0.454, 1.626, 4.856]
ax.plot(complexity, bits_by_complexity, 'o-', linewidth=2, markersize=10, color='#FF6B6B')
ax.set_ylabel('Information Update (bits)', fontsize=12)
ax.set_title('Complexity-Dependent Scaling', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 3. Information flow pattern
ax = axes[1, 0]
positions = np.arange(7)
distances = [0.012, 0.0086, 0.0144, 0.0148, 0.012, 0.0264, 0]
ax.plot(positions, distances, 'o-', linewidth=2, markersize=8)
ax.fill_between(positions, distances, alpha=0.3)
ax.set_xlabel('Token Position', fontsize=12)
ax.set_ylabel('Update Magnitude', fontsize=12)
ax.set_title('Bidirectional Information Radiation', fontsize=14, fontweight='bold')
ax.axvline(x=5, color='red', linestyle='--', label='Disambiguation')
ax.legend()

# 4. Summary statistics
ax = axes[1, 1]
ax.axis('off')
summary = """
KEY DISCOVERIES
═══════════════

1. ARCHITECTURE-DEPENDENT
   Not universal constant
   Range: 0.64 - 1.63 bits

2. COMPLEXITY SCALING
   Simple: 0.45 bits
   Standard: 1.63 bits
   Complex: 4.86 bits
   
3. BIDIRECTIONAL FLOW
   Information radiates
   from disambiguation
   
4. NEAR-MAXIMAL UPDATES
   98.3% of theoretical
   maximum achieved
"""
ax.text(0.1, 0.5, summary, fontsize=11, family='monospace', 
        verticalalignment='center')

plt.suptitle('Retroactive Causality: Complete Findings', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/complete_findings.png', dpi=300, bbox_inches='tight')
print("Saved comprehensive findings to results/complete_findings.png")
