"""
Does semantic incongruity cause retroactive updates across architectures?
This would show it's a fundamental property of transformers
"""
import torch
from transformers import (
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer,
    GPT2Model, GPT2Tokenizer
)
from scipy.spatial.distance import cosine
import numpy as np

print("CROSS-ARCHITECTURE SEMANTIC INCONGRUITY TEST")
print("="*60)

models = [
    ("bert-base-uncased", BertModel, BertTokenizer, True),  # bidirectional
    ("roberta-base", RobertaModel, RobertaTokenizer, True),  # bidirectional
    ("gpt2", GPT2Model, GPT2Tokenizer, False)  # unidirectional
]

base = "The horse raced past the barn"
congruent = "quickly"
temporal = "yesterday"

for name, model_class, tokenizer_class, is_bidirectional in models:
    print(f"\n{name} ({'bidirectional' if is_bidirectional else 'unidirectional'}):")
    
    model = model_class.from_pretrained(name)
    tokenizer = tokenizer_class.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    
    s1 = f"{base} {congruent}"
    s2 = f"{base} {temporal}"
    
    enc1 = tokenizer(s1, return_tensors='pt')
    enc2 = tokenizer(s2, return_tensors='pt')
    
    with torch.no_grad():
        out1 = model(**enc1, output_hidden_states=True)
        out2 = model(**enc2, output_hidden_states=True)
    
    # Position 1, middle layer
    layer = len(out1.hidden_states) // 2
    vec1 = out1.hidden_states[layer][0, 1].numpy()
    vec2 = out2.hidden_states[layer][0, 1].numpy()
    
    dist = cosine(vec1, vec2)
    print(f"  Semantic incongruity effect: {dist:.4f}")
    
    if is_bidirectional and dist > 0.02:
        print(f"  ✅ Shows retroactive updates")
    elif not is_bidirectional and dist < 0.01:
        print(f"  ✅ No retroactive updates (expected for unidirectional)")
