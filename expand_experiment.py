"""
Test with more sentences and controls
"""
import numpy as np
from scipy import stats
from src.models.model_loader import load_bert_model
from src.measurements.bert_garden_path import batch_measure_bert

# Add MORE garden path sentences from literature
additional_garden = [
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

additional_control = [
    "The florist sent the flowers successfully",
    "I convinced her children are noisy sometimes",
    "The man who hunts ducks skillfully", 
    "Fat people eat accumulates slowly",
    "The chicken is ready to eat now",
    "Time flies like an arrow swiftly",
    "While Anna dressed the baby quickly",
    "The boat floated down the river gently",
    "When Fred eats food gets cold",
    "The player tossed a frisbee happily"
]

print("Loading BERT...")
model, tokenizer = load_bert_model()

print(f"Testing {len(additional_garden)} additional sentences...")
results = batch_measure_bert(
    model, tokenizer,
    additional_garden,
    additional_control,
    layers_to_analyze=[6]  # Middle layer showed strongest effect
)

print(f"\n📊 Expanded Results:")
print(f"Mean distance: {results['global_stats']['mean']:.4f}")
print(f"N = {len(results['all_distances'])}")

# Combine with original data
original_data = np.load('results/bert_distances.npy')
combined = np.concatenate([original_data, results['all_distances']])

print(f"\n📈 Combined Dataset:")
print(f"Total measurements: {len(combined)}")
print(f"Mean: {np.mean(combined[combined > 0]):.4f}")
print(f"Cohen's d: {np.mean(combined[combined > 0]) / np.std(combined[combined > 0]):.3f}")
