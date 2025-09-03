"""
Where exactly does BERT's temporal coherence break?
Testing increasing levels of temporal violation
"""
import torch
from transformers import BertModel, BertTokenizer, BertForMaskedLM
from scipy.spatial.distance import cosine
import numpy as np

model = BertModel.from_pretrained('bert-base-uncased')
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

print("="*60)
print("TESTING BERT'S TEMPORAL BREAKING POINT")
print("="*60)

# Increasing levels of temporal violation
temporal_test = [
    ("Normal", "I ate breakfast then lunch"),
    ("Memory", "I remembered tomorrow's meeting"),  
    ("Metaphor", "The past haunts the future"),
    ("Impossible", "Tomorrow caused yesterday"),
    ("Contradiction", "Yesterday happens tomorrow"),
    ("Broken", "Time never always backward forward")
]

# Measure coherence degradation
baseline = "Events happen in sequence"
enc_base = tokenizer(baseline, return_tensors='pt')

with torch.no_grad():
    out_base = model(**enc_base, output_hidden_states=True)

print("\nTemporal Violation Gradient:")
print("-" * 40)

coherence_scores = []
for level, sentence in temporal_test:
    enc = tokenizer(sentence, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
        
    # Measure at layer 6, average all positions
    hidden = outputs.hidden_states[7]  # Layer 6
    
    # Calculate self-consistency (how much tokens agree)
    consistency = 0
    for i in range(hidden.shape[1]-1):
        for j in range(i+1, hidden.shape[1]):
            vec1 = hidden[0, i].numpy()
            vec2 = hidden[0, j].numpy()
            consistency += 1 - cosine(vec1, vec2)
    
    consistency /= (hidden.shape[1] * (hidden.shape[1]-1) / 2)
    coherence_scores.append(consistency)
    
    print(f"{level:15s}: {consistency:.4f} - '{sentence[:30]}...'")

# Check if there's a sharp transition
differences = [coherence_scores[i] - coherence_scores[i-1] for i in range(1, len(coherence_scores))]
max_drop_idx = np.argmax(np.abs(differences)) + 1

print(f"\n🎯 BIGGEST COHERENCE DROP: {temporal_test[max_drop_idx-1][0]} → {temporal_test[max_drop_idx][0]}")
print(f"This is where BERT's temporal model breaks!")
