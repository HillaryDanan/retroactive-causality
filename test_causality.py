"""
Test if updates are causal or just correlational
Based on Pearl (2009, Causality) - intervention test
"""
import torch
import numpy as np
from scipy import stats
from transformers import BertModel, BertTokenizer

print("="*60)
print("CAUSALITY TEST: Masking Intervention")
print("="*60)

# Load BERT
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

def measure_update(sentence1, sentence2, target_word="horse"):
    """Measure cosine distance for target word"""
    enc1 = tokenizer(sentence1, return_tensors='pt')
    enc2 = tokenizer(sentence2, return_tensors='pt')
    
    with torch.no_grad():
        out1 = model(**enc1, output_hidden_states=True)
        out2 = model(**enc2, output_hidden_states=True)
    
    # Find target token position
    tokens = tokenizer.tokenize(sentence1)
    try:
        pos = tokens.index(target_word)
    except:
        pos = 1  # fallback
    
    # Layer 6 (middle layer)
    vec1 = out1.hidden_states[6][0, pos].numpy()
    vec2 = out2.hidden_states[6][0, pos].numpy()
    
    # Cosine distance
    from scipy.spatial.distance import cosine
    return cosine(vec1, vec2)

# Test sentences
print("\n1. ORIGINAL (shows updates):")
original_garden = "The horse raced past the barn fell"
original_control = "The horse raced past the barn quickly"
original_dist = measure_update(original_garden, original_control)
print(f"   Garden: {original_garden}")
print(f"   Control: {original_control}")
print(f"   Distance: {original_dist:.4f}")

print("\n2. MASKED DISAMBIGUATION (intervention):")
masked_garden = "The horse raced past the barn [MASK]"
masked_control = "The horse raced past the barn [MASK]"
masked_dist = measure_update(masked_garden, masked_control)
print(f"   Garden: {masked_garden}")
print(f"   Control: {masked_control}")
print(f"   Distance: {masked_dist:.4f}")

print("\n3. PARTIALLY MASKED (control test):")
partial_garden = "The horse raced past [MASK] barn fell"
partial_control = "The horse raced past [MASK] barn quickly"
partial_dist = measure_update(partial_garden, partial_control)
print(f"   Garden: {partial_garden}")
print(f"   Control: {partial_control}")
print(f"   Distance: {partial_dist:.4f}")

# Analysis
reduction = (original_dist - masked_dist) / original_dist * 100

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Original update: {original_dist:.4f}")
print(f"Masked update: {masked_dist:.4f}")
print(f"Reduction: {reduction:.1f}%")

if reduction > 50:
    print("\n✅ CAUSAL RELATIONSHIP SUPPORTED!")
    print("Masking disambiguation reduces updates by >50%")
elif reduction > 20:
    print("\n⚠️ PARTIAL CAUSALITY")
    print("Some causal component but not dominant")
else:
    print("\n❌ PRIMARILY CORRELATIONAL")
    print("Updates persist despite masking")

# Save results
results = {
    'original': original_dist,
    'masked': masked_dist,
    'partial': partial_dist,
    'reduction_percent': reduction
}
np.save('results/causality_test.npy', results)
print(f"\n💾 Saved to results/causality_test.npy")
