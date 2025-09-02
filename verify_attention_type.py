import torch
from src.models.model_loader import load_gpt2_model, load_bert_model

print("Testing GPT-2 attention pattern...")
model, tokenizer = load_gpt2_model()

text = "The cat sat on mat"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
    attn = outputs.attentions[0][0, 0]  # First layer, first head
    
    print(f"Attention shape: {attn.shape}")
    print("\nAttention matrix (rounded):")
    print(attn.numpy().round(2))
    
    # Check if upper triangle is zero (causal mask)
    upper_triangle = torch.triu(attn, diagonal=1)
    if torch.allclose(upper_triangle, torch.zeros_like(upper_triangle)):
        print("\n❌ GPT-2 uses CAUSAL attention (left-to-right only)!")
        print("This means later tokens CANNOT affect earlier ones!")
    else:
        print("\n✅ Bidirectional attention detected")

print("\n" + "="*50)
print("Testing BERT attention pattern...")
bert_model, bert_tokenizer = load_bert_model()

bert_inputs = bert_tokenizer(text, return_tensors='pt')
with torch.no_grad():
    bert_outputs = bert_model(**bert_inputs, output_attentions=True)
    bert_attn = bert_outputs.attentions[0][0, 0]
    
    print(f"BERT attention shape: {bert_attn.shape}")
    upper = torch.triu(bert_attn, diagonal=1)
    if not torch.allclose(upper, torch.zeros_like(upper)):
        print("✅ BERT uses BIDIRECTIONAL attention!")
