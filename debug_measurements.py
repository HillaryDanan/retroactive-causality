import torch
import numpy as np
from scipy.spatial.distance import cosine
from src.models.model_loader import load_gpt2_model

model, tokenizer = load_gpt2_model()

# Test sentences
base_text = "The horse raced past the barn"
continuation = "fell"

# Tokenize
base_ids = tokenizer(base_text, return_tensors='pt').input_ids
full_text = base_text + " " + continuation
full_ids = tokenizer(full_text, return_tensors='pt').input_ids

print(f"Base tokens: {base_ids.shape}")
print(f"Full tokens: {full_ids.shape}")
print(f"Base: {tokenizer.decode(base_ids[0])}")
print(f"Full: {tokenizer.decode(full_ids[0])}")

# Get hidden states
with torch.no_grad():
    base_outputs = model(base_ids, output_hidden_states=True)
    full_outputs = model(full_ids, output_hidden_states=True)

# Check a specific token at a specific layer
layer_idx = 6
token_idx = 3  # "past"

base_vec = base_outputs.hidden_states[layer_idx][0, token_idx].cpu().numpy()
full_vec = full_outputs.hidden_states[layer_idx][0, token_idx].cpu().numpy()

print(f"\nBase vector first 5 values: {base_vec[:5]}")
print(f"Full vector first 5 values: {full_vec[:5]}")

# Check if they're identical
if np.allclose(base_vec, full_vec):
    print("❌ Vectors are identical!")
else:
    print("✅ Vectors differ")
    
# Calculate cosine distance properly
dist = cosine(base_vec, full_vec)
print(f"Cosine distance: {dist:.6f}")

# Check if it's a NaN issue
if np.isnan(dist):
    print("❌ NaN detected!")
    print(f"Base vec has NaN: {np.isnan(base_vec).any()}")
    print(f"Full vec has NaN: {np.isnan(full_vec).any()}")
