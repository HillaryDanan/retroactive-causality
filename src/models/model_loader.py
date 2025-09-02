"""
Standardized model loading with EXACT specifications
"""

import torch
from transformers import GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer
import numpy as np
import random

def set_all_seeds(seed=42):
    """Ensure complete reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_gpt2_model():
    """
    Load GPT-2 124M with exact specifications
    Following Rogers et al. (2020) setup
    """
    set_all_seeds(42)
    
    model = GPT2Model.from_pretrained('gpt2')  # 124M parameter version
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # CRITICAL: Set padding token to prevent errors
    tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()  # Disable dropout
    
    # Verify specifications
    assert model.config.n_embd == 768, "Wrong model size"
    assert model.config.n_layer == 12, "Wrong number of layers"
    assert model.config.n_head == 12, "Wrong number of heads"
    
    print(f"✅ Loaded GPT-2 124M: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    return model, tokenizer

def load_bert_model():
    """
    Load BERT-base for architecture comparison
    """
    set_all_seeds(42)
    
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model.eval()
    
    print(f"✅ Loaded BERT-base: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    return model, tokenizer