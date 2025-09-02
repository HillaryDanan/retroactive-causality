"""
Test if there's a universal information-theoretic bound
"""

import numpy as np
from scipy import stats
from src.models.model_loader import load_bert_model
from src.measurements.bert_garden_path import batch_measure_bert

model, tokenizer = load_bert_model()

# Test with MAXIMUM ambiguity sentences
max_ambiguity_sentences = [
    # Extreme garden paths with multiple reanalyses
    "The man who whistles tunes pianos",
    "The prime number few",
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo",
    "The rat the cat the dog chased killed ate the malt",
    "The girl told the story cried"
]

control_sentences = [
    "The man who whistles owns pianos",
    "The prime number exists",
    "Buffalo buffalo buffalo other buffalo",
    "The rat ate the malt quickly",
    "The girl who told the story cried"
]

print("Testing maximum ambiguity sentences...")
results = batch_measure_bert(
    model, tokenizer,
    max_ambiguity_sentences,
    control_sentences,
    layers_to_analyze=[0, 3, 6, 9, 11]
)

distances = results['all_distances']
distances = distances[distances > 0]

# Calculate theoretical maximum
vocab_size = 30522
theoretical_max = np.log2(vocab_size)  # 14.9 bits
observed_max = np.max(distances) * theoretical_max
observed_95 = np.percentile(distances, 95) * theoretical_max

print("\n" + "="*60)
print("THEORETICAL BOUNDS ANALYSIS")
print("="*60)

print(f"\nTheoretical maximum: {theoretical_max:.2f} bits")
print(f"Observed maximum: {observed_max:.2f} bits")
print(f"Observed 95th percentile: {observed_95:.2f} bits")
print(f"\nRatio to theoretical max:")
print(f"  Maximum: {observed_max/theoretical_max*100:.1f}%")
print(f"  95th percentile: {observed_95/theoretical_max*100:.1f}%")

# Test if bound is consistent
all_datasets = {
    'original': np.load('results/bert_distances.npy'),
    'expanded': np.load('results/expanded_bert_distances.npy'),
    'max_ambiguity': distances
}

bounds = {}
for name, data in all_datasets.items():
    data = data[data > 0]
    mean_bits = np.mean(data) * theoretical_max
    bounds[name] = mean_bits
    print(f"\n{name} dataset mean: {mean_bits:.3f} bits")

# Check consistency
bound_values = list(bounds.values())
cv = np.std(bound_values) / np.mean(bound_values)  # Coefficient of variation

print(f"\nConsistency check:")
print(f"  Mean bound: {np.mean(bound_values):.3f} bits")
print(f"  Coefficient of variation: {cv:.3f}")

if cv < 0.3:
    print("\n✅ CONSISTENT INFORMATION-THEORETIC BOUND FOUND!")
    print(f"Universal bound appears to be ~{np.mean(bound_values):.1f} bits")
