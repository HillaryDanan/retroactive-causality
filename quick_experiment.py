from src.models.model_loader import load_gpt2_model
from src.measurements.token_updates import measure_token_updates
import numpy as np

model, tokenizer = load_gpt2_model()

# Test one garden path
result = measure_token_updates(
    model, tokenizer,
    "The horse raced past the barn",
    "fell",
    layers_to_analyze=[6]
)

print(f"Mean update distance: {result['distribution_stats']['mean']:.4f}")
print(f"Std deviation: {result['distribution_stats']['std']:.4f}")
