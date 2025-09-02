"""
Do attention weights redistribute during garden path processing?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.model_loader import load_bert_model

model, tokenizer = load_bert_model()

# Get attention weights
def get_attention_matrix(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    # Average across all heads in layer 6
    attn = outputs.attentions[6].mean(dim=1)[0]
    return attn.numpy()

garden = "The horse raced past the barn fell"
control = "The horse raced past the barn quickly"

attn_garden = get_attention_matrix(garden)
attn_control = get_attention_matrix(control)

# Calculate difference
attn_diff = np.abs(attn_garden[:7, :7] - attn_control[:7, :7])  # First 7 tokens

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

tokens = tokenizer.tokenize(garden)[:7]

# Garden path attention
sns.heatmap(attn_garden[:7, :7], ax=axes[0], cmap='Blues', 
            xticklabels=tokens, yticklabels=tokens, cbar_kws={'label': 'Attention'})
axes[0].set_title('Garden Path Attention')

# Control attention  
sns.heatmap(attn_control[:7, :7], ax=axes[1], cmap='Greens',
            xticklabels=tokens, yticklabels=tokens, cbar_kws={'label': 'Attention'})
axes[1].set_title('Control Attention')

# Difference
sns.heatmap(attn_diff, ax=axes[2], cmap='Reds',
            xticklabels=tokens, yticklabels=tokens, cbar_kws={'label': 'Difference'})
axes[2].set_title('Attention Redistribution')

plt.tight_layout()
plt.savefig('results/attention_redistribution.png', dpi=300)
print("Saved attention patterns to results/attention_redistribution.png")

# Quantify redistribution
total_redistribution = np.sum(attn_diff)
mean_redistribution = np.mean(attn_diff)

print(f"\nAttention redistribution:")
print(f"  Total: {total_redistribution:.3f}")
print(f"  Mean: {mean_redistribution:.4f}")
print(f"  Max: {np.max(attn_diff):.4f}")
