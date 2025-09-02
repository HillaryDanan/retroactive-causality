import torch
from src.models.model_loader import load_gpt2_model

model, tokenizer = load_gpt2_model()

# Test if adding words changes previous representations
sentences = [
    "The horse raced",
    "The horse raced past",
    "The horse raced past the",
    "The horse raced past the barn",
    "The horse raced past the barn fell"
]

print("Testing if GPT-2 updates previous tokens...")
print("=" * 50)

with torch.no_grad():
    hidden_states = []
    for sent in sentences:
        ids = tokenizer(sent, return_tensors='pt').input_ids
        outputs = model(ids, output_hidden_states=True)
        hidden_states.append(outputs.hidden_states[6][0])  # Layer 6
        print(f"Sentence: '{sent}' -> {ids.shape[1]} tokens")

# Check if "horse" (token 1) changes
print("\nChecking token 1 ('horse') representations:")
for i in range(1, len(sentences)):
    vec_prev = hidden_states[i-1][1].numpy()  # Token 1 in previous
    vec_curr = hidden_states[i][1].numpy()    # Token 1 in current
    
    # Are they identical?
    identical = torch.allclose(hidden_states[i-1][1], hidden_states[i][1])
    
    print(f"Step {i}: Identical = {identical}")
    
    if not identical:
        diff = torch.norm(hidden_states[i][1] - hidden_states[i-1][1]).item()
        print(f"  -> L2 distance: {diff:.6f}")
