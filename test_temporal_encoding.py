"""
What does BERT think connects impossible temporal statements?
This reveals its internal model of time
"""
import torch
from transformers import BertForMaskedLM, BertTokenizer

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("PROBING BERT'S TEMPORAL MODEL")
print("="*60)

temporal_tests = [
    "Yesterday [MASK] tomorrow",
    "The future [MASK] the past",
    "Tomorrow [MASK] yesterday",
    "The effect [MASK] the cause",
    "Time flows [MASK]",
    "Entropy [MASK] spontaneously"
]

for test in temporal_tests:
    print(f"\n'{test}'")
    
    inputs = tokenizer(test, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    
    if len(mask_idx) > 0:
        mask_logits = predictions[0, mask_idx[0], :]
        top_3 = torch.topk(mask_logits, 3)
        
        print("  BERT predicts:")
        for i in range(3):
            token_id = top_3.indices[i].item()
            prob = torch.softmax(mask_logits, dim=0)[token_id].item()
            token = tokenizer.decode([token_id])
            print(f"    '{token}' ({prob:.1%})")
    else:
        print("  No mask token found")

print("\n" + "="*60)
print("INSIGHT: BERT's top predictions reveal its temporal model")
