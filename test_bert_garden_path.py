"""
Test that BERT shows garden path effects
"""

from src.models.model_loader import load_bert_model
from src.measurements.bert_garden_path import measure_bert_retroactive_updates
import numpy as np

print("Loading BERT...")
model, tokenizer = load_bert_model()

print("\n" + "="*60)
print("TESTING CLASSIC GARDEN PATH SENTENCE")
print("="*60)

# Classic garden path
garden = "The horse raced past the barn fell"
control = "The horse raced past the barn quickly"

print(f"\nGarden: '{garden}'")
print(f"Control: '{control}'")
print("\nMeasuring 'horse' representation changes...")

result = measure_bert_retroactive_updates(
    model, tokenizer,
    garden, control,
    target_word="horse",
    layers_to_analyze=[0, 6, 11]  # Early, middle, late
)

print(f"\n📊 RESULTS:")
print(f"Mean distance: {result['distribution_stats']['mean']:.4f}")
print(f"Max distance: {result['distribution_stats']['max']:.4f}")
print(f"Std deviation: {result['distribution_stats']['std']:.4f}")

print("\n📈 Per-layer effects:")
for layer, dist in result['per_layer'].items():
    print(f"  Layer {layer:2d}: {dist:.4f}")

# Test multiple sentences
print("\n" + "="*60)
print("TESTING MULTIPLE GARDEN PATH SENTENCES")
print("="*60)

garden_sentences = [
    "The horse raced past the barn fell",
    "The old man the boat",
    "The cotton clothing is made of grows",
    "While Mary was mending the sock fell"
]

control_sentences = [
    "The horse raced past the barn quickly",
    "The old man owned the boat",
    "The cotton clothing is made of fabric",
    "While Mary was mending the sock carefully"
]

target_words = ["horse", "old", "cotton", "Mary"]

from src.measurements.bert_garden_path import batch_measure_bert

batch_results = batch_measure_bert(
    model, tokenizer,
    garden_sentences,
    control_sentences,
    target_words,
    layers_to_analyze=[6]  # Just middle layer for speed
)

print(f"\n🎯 BATCH RESULTS:")
print(f"Global mean distance: {batch_results['global_stats']['mean']:.4f}")
print(f"Global std: {batch_results['global_stats']['std']:.4f}")
print(f"Total measurements: {len(batch_results['all_distances'])}")

if batch_results['global_stats']['mean'] > 0.1:
    print("\n✅ STRONG GARDEN PATH EFFECT DETECTED!")
elif batch_results['global_stats']['mean'] > 0.05:
    print("\n⚠️ MODERATE GARDEN PATH EFFECT")
else:
    print("\n❌ WEAK/NO GARDEN PATH EFFECT")
