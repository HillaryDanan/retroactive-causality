import torch
import numpy as np
from typing import Dict, Tuple

def check_bidirectional_attention(model, tokenizer) -> Tuple[bool, str]:
    """Verify BERT has bidirectional attention"""
    
    text = "The cat sat on the mat"
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attention = outputs.attentions[0][0, 0]
        
        # Check if later tokens attend to earlier tokens
        backward_attention = attention[5, :5].sum().item()
        # Check if earlier tokens attend to later tokens  
        forward_attention = attention[1, 2:].sum().item()
        
    success = backward_attention > 0 and forward_attention > 0
    message = f"Backward: {backward_attention:.4f}, Forward: {forward_attention:.4f}"
    
    if success:
        message = "✅ " + message + " - Bidirectional attention confirmed"
    else:
        message = "❌ " + message + " - Expected bidirectional attention not found!"
    
    return success, message

def check_garden_path_exists(model, tokenizer) -> Tuple[bool, str]:
    """Check if garden path causes updates in BERT"""
    
    from src.measurements.token_updates_fixed import measure_token_updates
    
    # For BERT, we need to use [MASK] tokens creatively
    # Or compare full sentence encodings
    
    garden = "The horse raced past the barn fell"
    control = "The horse raced past the barn quickly"
    
    # Get embeddings for "horse" in both contexts
    garden_ids = tokenizer(garden, return_tensors='pt').input_ids
    control_ids = tokenizer(control, return_tensors='pt').input_ids
    
    with torch.no_grad():
        garden_out = model(garden_ids, output_hidden_states=True)
        control_out = model(control_ids, output_hidden_states=True)
        
        # Compare "horse" token (position 2 in BERT due to [CLS])
        garden_horse = garden_out.hidden_states[6][0, 2]
        control_horse = control_out.hidden_states[6][0, 2]
        
        distance = torch.norm(garden_horse - control_horse).item()
    
    success = distance > 0.1
    message = f"Distance: {distance:.4f}"
    
    if success:
        message = "✅ " + message + " - Garden path effect detected"
    else:
        message = "⚠️ " + message + " - Garden path effect weak"
    
    return success, message
