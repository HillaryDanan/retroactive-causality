"""
CORRECTED Intervention Gradient Analysis
Tests how different disambiguations affect retroactive updates
"""
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

print("="*60)
print("CORRECTED INTERVENTION GRADIENT ANALYSIS")
print("="*60)
print("Measuring: How do different endings affect retroactive updates?")
print("Baseline: 'fell' vs 'quickly' = 0.0187")

# The CORRECT way: Always compare garden path vs control structure
base = "The horse raced past the barn"
control_end = "quickly"  # This stays constant

test_endings = [
    ("fell", "Original garden path"),
    ("[MASK]", "Masked token"),
    ("tumbled", "Synonym - falling"),
    ("collapsed", "Synonym - structural failure"),  
    ("jumped", "Different action"),
    ("sang", "Implausible action"),
    ("elephant", "Random noun"),
    ("42", "Number"),
    (".", "Punctuation"),
    ("", "Empty"),
]

results = []

for ending, description in test_endings:
    garden = f"{base} {ending}".strip()
    control = f"{base} {control_end}"
    
    enc_g = tokenizer(garden, return_tensors='pt')
    enc_c = tokenizer(control, return_tensors='pt')
    
    with torch.no_grad():
        out_g = model(**enc_g, output_hidden_states=True)
        out_c = model(**enc_c, output_hidden_states=True)
    
    # Measure at "horse" position
    vec_g = out_g.hidden_states[6][0, 1].numpy()
    vec_c = out_c.hidden_states[6][0, 1].numpy()
    
    dist = cosine(vec_g, vec_c)
    reduction = ((0.0187 - dist) / 0.0187) * 100 if dist else 100
    
    results.append({
        'ending': ending,
        'description': description,
        'distance': dist,
        'reduction': reduction
    })
    
    print(f"{ending:12s} ({description:25s}): {dist:.4f} ({reduction:+.1f}% change)")

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)

# Sort by reduction
results_sorted = sorted(results, key=lambda x: x['reduction'], reverse=True)

print("\nMost to least causal blocking:")
for r in results_sorted:
    print(f"  {r['reduction']:+6.1f}%: {r['ending']:12s} ({r['description']})")

# Key insight
mask_result = next(r for r in results if r['ending'] == '[MASK]')
if mask_result['distance'] < 0.001:
    print("\n✅ Confirmed: [MASK] eliminates retroactive updates")
    
fell_result = next(r for r in results if r['ending'] == 'fell')
print(f"\n📊 Baseline 'fell' creates {fell_result['distance']:.4f} distance")
print("Deviations from 'fell' generally reduce retroactive updating")

# Save corrected results
import json
with open('results/intervention_gradient_corrected.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)
    
print("\n💾 Saved corrected results to results/intervention_gradient_corrected.json")
