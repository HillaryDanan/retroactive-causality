"""
Why does [MASK] show different results in different tests?
This is critical to resolve.
"""
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

print("INVESTIGATING [MASK] CONTRADICTION")
print("="*60)

# Original causality test method
print("\nMETHOD 1 (Original causality test):")
print("Compare: 'barn [MASK]' vs 'barn [MASK]'")
s1 = "The horse raced past the barn [MASK]"
s2 = "The horse raced past the barn [MASK]"

enc1 = tokenizer(s1, return_tensors='pt')
enc2 = tokenizer(s2, return_tensors='pt')

with torch.no_grad():
    out1 = model(**enc1, output_hidden_states=True)
    out2 = model(**enc2, output_hidden_states=True)

vec1 = out1.hidden_states[6][0, 1].numpy()
vec2 = out2.hidden_states[6][0, 1].numpy()
dist = cosine(vec1, vec2)
print(f"Distance: {dist:.4f} (Should be 0 - same sentence)")

# Corrected method
print("\nMETHOD 2 (Corrected gradient test):")
print("Compare: 'barn [MASK]' vs 'barn quickly'")
s3 = "The horse raced past the barn [MASK]"
s4 = "The horse raced past the barn quickly"

enc3 = tokenizer(s3, return_tensors='pt')
enc4 = tokenizer(s4, return_tensors='pt')

with torch.no_grad():
    out3 = model(**enc3, output_hidden_states=True)
    out4 = model(**enc4, output_hidden_states=True)

vec3 = out3.hidden_states[6][0, 1].numpy()
vec4 = out4.hidden_states[6][0, 1].numpy()
dist2 = cosine(vec3, vec4)
print(f"Distance: {dist2:.4f}")

# The actual comparison for causality
print("\nWHAT WE SHOULD MEASURE FOR CAUSALITY:")
print("Does [MASK] eliminate the garden path effect?")

# Garden path effect with real words
garden_real = "The horse raced past the barn fell"
control_real = "The horse raced past the barn quickly"

enc_gr = tokenizer(garden_real, return_tensors='pt')
enc_cr = tokenizer(control_real, return_tensors='pt')

with torch.no_grad():
    out_gr = model(**enc_gr, output_hidden_states=True)
    out_cr = model(**enc_cr, output_hidden_states=True)

vec_gr = out_gr.hidden_states[6][0, 1].numpy()
vec_cr = out_cr.hidden_states[6][0, 1].numpy()
baseline_dist = cosine(vec_gr, vec_cr)

print(f"\nBaseline (fell vs quickly): {baseline_dist:.4f}")
print(f"With [MASK]: {dist2:.4f}")

if dist2 > baseline_dist:
    print("\n⚠️ [MASK] INCREASES distance (creates MORE difference)")
    print("This means [MASK] doesn't eliminate the effect - it amplifies it!")
else:
    reduction = ((baseline_dist - dist2) / baseline_dist) * 100
    print(f"\n[MASK] reduces distance by {reduction:.1f}%")

print("\n" + "="*60)
print("CONCLUSION:")
print("Your original 'causality' test measured whether two identical")
print("sentences have the same representation (trivially yes).")
print("The real question is whether [MASK] eliminates the difference")
print("between garden path and control structures.")
