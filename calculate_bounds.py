"""
Calculate the information-theoretic bounds of semantic reconfiguration
"""
import numpy as np

# Load your data
distances = np.load('results/bert_distances.npy')
distances_nonzero = distances[distances > 0]

# BERT's vocabulary size
vocab_size = 30522
bits_per_token = np.log2(vocab_size)  # ~15 bits

print("="*60)
print("INFORMATION-THEORETIC ANALYSIS")
print("="*60)

print(f"\nBERT vocabulary: {vocab_size:,} tokens")
print(f"Bits per token: {bits_per_token:.2f}")

# Convert cosine distance to information change estimate
# Cosine distance of 0.1 ≈ 10% change in vector direction
mean_change = np.mean(distances_nonzero)
max_change = np.max(distances_nonzero)

print(f"\nSemantic Reconfiguration Bounds:")
print(f"  Average: {mean_change*100:.2f}% vector change")
print(f"  Maximum: {max_change*100:.2f}% vector change")

# Estimate bits of information updated
# Rough approximation: % change * total bits
estimated_bits_changed = mean_change * bits_per_token
print(f"\nEstimated information update:")
print(f"  ~{estimated_bits_changed:.2f} bits per token on average")
print(f"  ~{max_change * bits_per_token:.2f} bits maximum")

# Compare to your theoretical prediction
print(f"\nTheoretical minimum (1 bit): {1/bits_per_token*100:.1f}% change")
print(f"Observed average: {mean_change*100:.2f}% change")
print(f"Ratio: {mean_change/(1/bits_per_token):.1f}x theoretical minimum")
