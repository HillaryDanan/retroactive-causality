"""
Test retroactive updates across different transformer architectures
This will show if 1.626 bits is BERT-specific or universal
"""

import torch
import numpy as np
from transformers import (
    RobertaModel, RobertaTokenizer,
    ElectraModel, ElectraTokenizer,
    AlbertModel, AlbertTokenizer,
    DistilBertModel, DistilBertTokenizer
)
from src.measurements.bert_garden_path import measure_bert_retroactive_updates
from scipy import stats
import pandas as pd

print("="*60)
print("MULTI-ARCHITECTURE RETROACTIVE CAUSALITY TEST")
print("="*60)

# Test sentences
garden_sentences = [
    "The horse raced past the barn fell",
    "The old man the boat",
    "The complex houses married and single soldiers",
    "While Mary was mending the sock fell",
    "The cotton clothing is made of grows"
]

control_sentences = [
    "The horse raced past the barn quickly",
    "The old man owned the boat",
    "The complex houses married and single people",
    "While Mary was mending the sock carefully",
    "The cotton clothing is made of fabric"
]

# Models to test
models_to_test = [
    ("roberta-base", RobertaModel, RobertaTokenizer),
    ("google/electra-base-discriminator", ElectraModel, ElectraTokenizer),
    ("albert-base-v2", AlbertModel, AlbertTokenizer),
    ("distilbert-base-uncased", DistilBertModel, DistilBertTokenizer)
]

results = {}

for model_name, model_class, tokenizer_class in models_to_test:
    print(f"\nTesting {model_name}...")
    
    # Load model
    model = model_class.from_pretrained(model_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model.eval()
    
    all_distances = []
    
    for garden, control in zip(garden_sentences, control_sentences):
        result = measure_bert_retroactive_updates(
            model, tokenizer,
            garden, control,
            layers_to_analyze=[6]  # Middle layer for consistency
        )
        all_distances.extend(result['distances'])
    
    distances = np.array(all_distances)
    distances = distances[distances > 0]
    
    # Calculate stats
    mean_dist = np.mean(distances) if len(distances) > 0 else 0
    std_dist = np.std(distances) if len(distances) > 0 else 0
    cohens_d = mean_dist / std_dist if std_dist > 0 else 0
    
    # Information content
    vocab_size = len(tokenizer)
    bits = mean_dist * np.log2(vocab_size)
    
    results[model_name] = {
        'mean_distance': mean_dist,
        'std': std_dist,
        'cohens_d': cohens_d,
        'bits': bits,
        'n': len(distances),
        'vocab_size': vocab_size
    }
    
    print(f"  Mean: {mean_dist:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  Information: {bits:.3f} bits")

# Summary table
print("\n" + "="*60)
print("COMPARATIVE RESULTS")
print("="*60)

df = pd.DataFrame(results).T
print(df.to_string())

# Save results
df.to_csv('results/architecture_comparison.csv')
print("\nSaved to results/architecture_comparison.csv")

# Test if all models show similar bounds
bits_values = [r['bits'] for r in results.values()]
print(f"\nInformation bounds across models:")
print(f"  Mean: {np.mean(bits_values):.3f} bits")
print(f"  Std: {np.std(bits_values):.3f} bits")
print(f"  Range: [{min(bits_values):.3f}, {max(bits_values):.3f}]")

if np.std(bits_values) < 0.5:
    print("\n✅ CONSISTENT BOUND ACROSS ARCHITECTURES!")
