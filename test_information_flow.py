"""
Where does information flow? Forward, backward, or both?
"""

import numpy as np
import matplotlib.pyplot as plt
from src.models.model_loader import load_bert_model
from src.measurements.bert_garden_path import measure_bert_retroactive_updates

model, tokenizer = load_bert_model()

# Test with positional focus
sentence_garden = "The horse raced past the barn fell"
sentence_control = "The horse raced past the barn quickly"

# Tokenize to get positions
tokens_garden = tokenizer.tokenize(sentence_garden)
tokens_control = tokenizer.tokenize(sentence_control)

print("Garden tokens:", tokens_garden)
print("Control tokens:", tokens_control)

# Measure updates for EACH token position
position_effects = {}

for pos in range(len(tokens_garden)-1):  # Skip last token
    token = tokens_garden[pos]
    
    result = measure_bert_retroactive_updates(
        model, tokenizer,
        sentence_garden, sentence_control,
        target_word=token,
        layers_to_analyze=[3, 6, 9]
    )
    
    position_effects[pos] = {
        'token': token,
        'mean_distance': result['distribution_stats']['mean'],
        'layer_effects': result['per_layer']
    }
    
    print(f"Position {pos} ({token}): {result['distribution_stats']['mean']:.4f}")

# Visualize information flow
positions = list(position_effects.keys())
distances = [position_effects[p]['mean_distance'] for p in positions]
tokens = [position_effects[p]['token'] for p in positions]

plt.figure(figsize=(12, 6))
bars = plt.bar(positions, distances, color='steelblue', alpha=0.7)
plt.xlabel('Token Position', fontsize=12)
plt.ylabel('Mean Update Distance', fontsize=12)
plt.title('Retroactive Information Flow by Position', fontsize=14, fontweight='bold')
plt.xticks(positions, tokens, rotation=45)

# Highlight the disambiguation point
disambig_pos = len(tokens_garden) - 2  # "barn" position
plt.axvline(x=disambig_pos, color='red', linestyle='--', label='Disambiguation point')
plt.legend()

plt.tight_layout()
plt.savefig('results/information_flow_pattern.png', dpi=300)
print(f"\nSaved flow pattern to results/information_flow_pattern.png")

# Analyze pattern
early_tokens = distances[:3]
late_tokens = distances[-3:]
print(f"\nEarly tokens mean: {np.mean(early_tokens):.4f}")
print(f"Late tokens mean: {np.mean(late_tokens):.4f}")

if np.mean(early_tokens) > np.mean(late_tokens):
    print("✅ RETROACTIVE FLOW CONFIRMED: Early tokens update more!")
