"""
Complete BERT garden path experiment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.models.model_loader import load_bert_model
from src.measurements.bert_garden_path import batch_measure_bert

print("="*60)
print("BERT GARDEN PATH RETROACTIVE CAUSALITY EXPERIMENT")
print("="*60)

# Load model
print("\nLoading BERT...")
model, tokenizer = load_bert_model()

# Prepare sentences
garden_sentences = [
    "The horse raced past the barn fell",
    "The old man the boat",
    "The complex houses married and single soldiers",
    "While Mary was mending the sock fell",
    "The cotton clothing is made of grows",
    "The prime number few mathematicians",
    "Since Jay always jogs a mile seems",
    "The raft floated down the river sank",
    "The daughter of the king's son admires",
    "The cat chased through the garden died",
]

# Control versions (grammatically normal)
control_sentences = [
    "The horse raced past the barn quickly",
    "The old man owned the boat",
    "The complex houses married and single people",
    "While Mary was mending the sock carefully", 
    "The cotton clothing is made of fabric",
    "The prime number puzzled few mathematicians",
    "Since Jay always jogs a mile daily",
    "The raft floated down the river smoothly",
    "The daughter of the king's son arrived",
    "The cat chased through the garden playfully",
]

# Words to track (early in sentence, before disambiguation)
target_words = ["horse", "old", "complex", "Mary", "cotton", 
                "prime", "Jay", "raft", "daughter", "cat"]

# Run experiment
print(f"\nAnalyzing {len(garden_sentences)} sentence pairs...")
print("Layers analyzed: 0 (embedding), 6 (middle), 11 (final)")

results = batch_measure_bert(
    model, tokenizer,
    garden_sentences,
    control_sentences,
    target_words,
    layers_to_analyze=[0, 6, 11]
)

# Statistical analysis
distances = results['all_distances']
distances_nonzero = distances[distances > 0]

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\n📊 Distribution Statistics:")
print(f"  Mean distance: {np.mean(distances_nonzero):.4f}")
print(f"  Median: {np.median(distances_nonzero):.4f}")
print(f"  Std dev: {np.std(distances_nonzero):.4f}")
print(f"  Min: {np.min(distances_nonzero):.4f}")
print(f"  Max: {np.max(distances_nonzero):.4f}")
print(f"  95th percentile: {np.percentile(distances_nonzero, 95):.4f}")

# Test against null hypothesis (no effect = distance near 0)
t_stat, p_value = stats.ttest_1samp(distances_nonzero, 0)
effect_size = np.mean(distances_nonzero) / np.std(distances_nonzero)

print(f"\n📈 Statistical Tests:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.2e}")
print(f"  Cohen's d: {effect_size:.3f}")

if p_value < 0.001:
    print("\n✅ HIGHLY SIGNIFICANT GARDEN PATH EFFECT! (p < 0.001)")
elif p_value < 0.05:
    print("\n✅ SIGNIFICANT GARDEN PATH EFFECT! (p < 0.05)")
else:
    print("\n⚠️ No significant effect detected")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Distribution
ax = axes[0]
ax.hist(distances_nonzero, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax.axvline(np.mean(distances_nonzero), color='red', linestyle='--', label=f'Mean: {np.mean(distances_nonzero):.3f}')
ax.set_xlabel('Cosine Distance')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Retroactive Updates')
ax.legend()

# Per-sentence effects
ax = axes[1]
sentence_means = [r['distribution_stats']['mean'] for r in results['individual_results']]
x_pos = np.arange(len(sentence_means))
ax.bar(x_pos, sentence_means, color='green', alpha=0.7)
ax.set_xlabel('Sentence Pair')
ax.set_ylabel('Mean Distance')
ax.set_title('Effect by Sentence')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"S{i+1}" for i in range(len(sentence_means))], rotation=45)

# Layer analysis
ax = axes[2]
layer_means = {}
for r in results['individual_results']:
    for layer, dist in r['per_layer'].items():
        if layer not in layer_means:
            layer_means[layer] = []
        layer_means[layer].append(dist)

layers = sorted(layer_means.keys())
means = [np.mean(layer_means[l]) for l in layers]
stds = [np.std(layer_means[l]) for l in layers]

ax.errorbar(layers, means, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2)
ax.set_xlabel('Layer')
ax.set_ylabel('Mean Distance')
ax.set_title('Effect Across Layers')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bert_garden_path_results.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Figure saved as 'bert_garden_path_results.png'")

# Save results
np.save('results/bert_distances.npy', distances)
print(f"📁 Data saved to 'results/bert_distances.npy'")

print("\n" + "="*60)
print("EXPERIMENT COMPLETE!")
print("="*60)
