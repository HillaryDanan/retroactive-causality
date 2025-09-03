"""
How do different models handle temporal impossibility?
"""
import torch
from transformers import (
    BertForMaskedLM, BertTokenizer,
    RobertaForMaskedLM, RobertaTokenizer
)

print("COMPARING TEMPORAL IMPOSSIBILITY ACROSS MODELS")
print("="*60)

# Test sentence with masked temporal word
test_template = "Yesterday [MASK] tomorrow"

models = [
    ("bert-base-uncased", BertForMaskedLM, BertTokenizer),
    ("roberta-base", RobertaForMaskedLM, RobertaTokenizer)
]

for model_name, model_class, tokenizer_class in models:
    print(f"\n{model_name} predictions for 'Yesterday [MASK] tomorrow':")
    
    model = model_class.from_pretrained(model_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    
    if "roberta" in model_name:
        test = test_template.replace("[MASK]", "<mask>")
    else:
        test = test_template
    
    inputs = tokenizer(test, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Get top 5 predictions for mask
    mask_token_id = tokenizer.mask_token_id if "bert" in model_name else tokenizer.convert_tokens_to_ids("<mask>")
    mask_idx = (inputs.input_ids == mask_token_id).nonzero(as_tuple=True)[1]
    mask_logits = predictions[0, mask_idx, :]
    top_5 = torch.topk(mask_logits, 5, dim=1)
    
    for i in range(5):
        token_id = top_5.indices[0][i].item()
        prob = torch.softmax(mask_logits, dim=1)[0][token_id].item()
        token = tokenizer.decode([token_id])
        print(f"  {i+1}. '{token}' ({prob:.2%})")

print("\n💡 Models that predict temporal connectors (before/after/precedes)")
print("   can handle paradoxes better!")
