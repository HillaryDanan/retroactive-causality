"""
Verify cosine distances reflect semantic changes
Simplified version without sklearn dependency
"""
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from scipy import stats

print("="*60)
print("SEMANTIC VALIDITY: Distance-Change Correlation Test")
print("="*60)

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

# Test if distance correlates with known syntactic changes
test_cases = [
    # Clear syntactic changes (should show high distance)
    {
        "garden": "The horse raced past the barn fell",
        "control": "The horse raced past the barn quickly",
        "word": "raced",
        "change_type": "verb→modifier",
        "expected": "high"
    },
    {
        "garden": "The old man the boat",
        "control": "The old man owned the boat",
        "word": "man",
        "change_type": "noun→verb",
        "expected": "high"
    },
    # Minimal syntactic change (should show low distance)
    {
        "garden": "The cat sat on the mat",
        "control": "The cat sat on a mat",
        "word": "cat",
        "change_type": "none",
        "expected": "low"
    },
]

print("\nMeasuring syntactic change correlations...")
distances = []
expected_magnitudes = []

for case in test_cases:
    enc_g = tokenizer(case["garden"], return_tensors='pt')
    enc_c = tokenizer(case["control"], return_tensors='pt')
    tokens_g = tokenizer.tokenize(case["garden"])
    
    try:
        pos = tokens_g.index(case["word"])
    except:
        # Try partial match
        pos = next((i for i, t in enumerate(tokens_g) if case["word"] in t), 1)
    
    with torch.no_grad():
        out_g = model(**enc_g, output_hidden_states=True)
        out_c = model(**enc_c, output_hidden_states=True)
        
        # Layer 6 representations
        vec_g = out_g.hidden_states[6][0, pos].numpy()
        vec_c = out_c.hidden_states[6][0, pos].numpy()
    
    # Cosine distance
    from scipy.spatial.distance import cosine
    dist = cosine(vec_g, vec_c)
    distances.append(dist)
    expected_magnitudes.append(1 if case["expected"] == "high" else 0)
    
    print(f"\n'{case['word']}': {case['change_type']}")
    print(f"  Distance: {dist:.4f}")
    print(f"  Expected: {case['expected']} distance")
    print(f"  Match: {'✅' if (dist > 0.01) == (case['expected'] == 'high') else '❌'}")

# Statistical validation
distances = np.array(distances)
expected = np.array(expected_magnitudes)

# Correlation test
r, p = stats.pearsonr(distances, expected)

print("\n" + "="*60)
print("SEMANTIC VALIDITY RESULTS:")
print("="*60)
print(f"Correlation (distance vs syntactic change): r = {r:.3f}, p = {p:.3f}")

if r > 0.5:
    print("✅ STRONG SEMANTIC VALIDITY!")
    print("Distances reflect syntactic changes")
elif r > 0.3:
    print("⚠️ MODERATE semantic validity")
else:
    print("❌ Weak semantic correlation")

# Additional validation: compare magnitudes
high_dist = distances[expected == 1]
low_dist = distances[expected == 0]

if len(high_dist) > 0 and len(low_dist) > 0:
    t_stat, p_val = stats.ttest_ind(high_dist, low_dist)
    print(f"\nHigh vs Low distance comparison:")
    print(f"  High change mean: {np.mean(high_dist):.4f}")
    print(f"  Low change mean: {np.mean(low_dist):.4f}")
    print(f"  t-test: t = {t_stat:.3f}, p = {p_val:.3f}")
    
    if p_val < 0.05 and np.mean(high_dist) > np.mean(low_dist):
        print("  ✅ Significant difference confirmed!")

# Save results
np.savez('results/semantic_validity.npz',
         distances=distances,
         expected=expected,
         correlation_r=r,
         correlation_p=p)
print("\n💾 Saved to results/semantic_validity.npz")
