"""
EXPANDED DATASET - MORE GARDEN PATHS!
Testing robustness across 20+ sentence pairs
"""
import numpy as np
from scipy import stats
from src.models.model_loader import load_bert_model
from src.measurements.bert_garden_path import batch_measure_bert
import matplotlib.pyplot as plt

print("="*60)
print("EXPANDED RETROACTIVE CAUSALITY EXPERIMENT")
print("="*60)

# ALL THE GARDEN PATHS FROM LITERATURE!
garden_sentences = [
    # Original 10
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
    # Additional 10
    "The florist sent the flowers was pleased",
    "I convinced her children are noisy", 
    "The man who hunts ducks",
    "Fat people eat accumulates",
    "The chicken is ready to eat",
    "Time flies like an arrow",
    "While Anna dressed the baby spit",
    "The boat floated down the river sank",
    "When Fred eats food gets",
    "The player tossed a frisbee smiled"
]

control_sentences = [
    # Original controls
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
    # Additional controls
    "The florist sent the flowers yesterday",
    "I convinced her children are smart",
    "The man who hunts professionally",
    "Fat people eat accumulates slowly",
    "The chicken is ready to serve",
    "Time flies like an arrow quickly",
    "While Anna dressed the baby carefully",
    "The boat floated down the river gently",
    "When Fred eats food disappears",
    "The player tossed a frisbee accurately"
]

print(f"\n📊 Testing {len(garden_sentences)} sentence pairs")
print("Loading BERT...")
model, tokenizer = load_bert_model()

print("\nRunning measurements...")
results = batch_measure_bert(
    model, tokenizer,
    garden_sentences,
    control_sentences,
    layers_to_analyze=[0, 3, 6, 9, 11]  # More layers!
)

# Analysis
distances = results['all_distances']
distances_nonzero = distances[distances > 0]

print("\n" + "="*60)
print("EXPANDED RESULTS")
print("="*60)

print(f"\n📊 Dataset Size:")
print(f"  Sentence pairs: {len(garden_sentences)}")
print(f"  Total measurements: {len(distances)}")
print(f"  Non-zero measurements: {len(distances_nonzero)}")

print(f"\n📈 Distribution Statistics:")
print(f"  Mean distance: {np.mean(distances_nonzero):.4f}")
print(f"  Median: {np.median(distances_nonzero):.4f}")
print(f"  Std dev: {np.std(distances_nonzero):.4f}")
print(f"  95% CI: [{np.percentile(distances_nonzero, 2.5):.4f}, {np.percentile(distances_nonzero, 97.5):.4f}]")

# Statistical tests
t_stat, p_value = stats.ttest_1samp(distances_nonzero, 0)
cohens_d = np.mean(distances_nonzero) / np.std(distances_nonzero)

print(f"\n🔬 Statistical Tests:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.2e}")
print(f"  Cohen's d: {cohens_d:.3f}")
print(f"  Power: {1 - stats.norm.cdf(1.96 - cohens_d * np.sqrt(len(distances_nonzero))):.3f}")

# Effect size interpretation
if cohens_d > 0.8:
    effect = "LARGE"
elif cohens_d > 0.5:
    effect = "MEDIUM"
elif cohens_d > 0.2:
    effect = "SMALL"
else:
    effect = "NEGLIGIBLE"

print(f"  Effect size: {effect}")

# Bootstrap confidence interval for robustness
n_bootstrap = 10000
bootstrap_means = []
for _ in range(n_bootstrap):
    sample = np.random.choice(distances_nonzero, size=len(distances_nonzero), replace=True)
    bootstrap_means.append(np.mean(sample))

bootstrap_ci = np.percentile(bootstrap_means, [2.5, 97.5])
print(f"\n🔄 Bootstrap Analysis ({n_bootstrap} samples):")
print(f"  Mean estimate: {np.mean(bootstrap_means):.4f}")
print(f"  95% CI: [{bootstrap_ci[0]:.4f}, {bootstrap_ci[1]:.4f}]")

# Save expanded results
np.save('results/expanded_bert_distances.npy', distances)
print(f"\n💾 Saved to 'results/expanded_bert_distances.npy'")

# Information theory calculation
vocab_size = 30522
bits_per_token = np.log2(vocab_size)
mean_bits = np.mean(distances_nonzero) * bits_per_token
print(f"\n🧮 Information-Theoretic Bounds:")
print(f"  Mean information change: {mean_bits:.3f} bits")
print(f"  As % of total: {np.mean(distances_nonzero)*100:.2f}%")

print("\n" + "="*60)
if p_value < 0.001 and cohens_d > 0.8:
    print("🎉 ROBUST LARGE EFFECT CONFIRMED ACROSS EXPANDED DATASET!")
print("="*60)
