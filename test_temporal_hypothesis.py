"""
Hypothesis: Temporal words force complete sequence reinterpretation
because they change the entire event frame
"""
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("TESTING EVENT FRAME DISRUPTION HYPOTHESIS")
print("="*60)

# Different types of frame violations
frame_tests = {
    'Consistent frame': [
        "The horse raced past the barn quickly",
        "The horse raced past the barn slowly",
    ],
    'Location shift (partial frame change)': [
        "The horse raced past the barn door",
        "The horse raced past the barn fence",
    ],
    'Temporal shift (complete frame change)': [
        "The horse raced past the barn yesterday",
        "The horse raced past the barn tomorrow",
    ],
    'Aspectual shift (event structure change)': [
        "The horse raced past the barn repeatedly",
        "The horse raced past the barn continuously",
    ]
}

print("\nMeasuring frame consistency effects...")
base = "The horse raced past the barn quickly"
enc_base = tokenizer(base, return_tensors='pt')

with torch.no_grad():
    out_base = model(**enc_base, output_hidden_states=True)

for frame_type, sentences in frame_tests.items():
    print(f"\n{frame_type}:")
    for sent in sentences:
        enc = tokenizer(sent, return_tensors='pt')
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        
        # Average across all positions
        distances = []
        for pos in range(1, 5):  # First few tokens
            vec = out.hidden_states[6][0, pos].numpy()
            vec_base = out_base.hidden_states[6][0, pos].numpy()
            distances.append(cosine(vec, vec_base))
        
        mean_dist = np.mean(distances)
        print(f"  {sent.split()[-1]:15s}: {mean_dist:.4f}")

print("\nConclusion: Do temporal words disrupt the entire event frame?")
