import torch
import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine
from tqdm import tqdm

def safe_cosine_distance(vec1, vec2):
    """Calculate cosine distance with NaN handling"""
    # Handle zero vectors
    if np.allclose(vec1, 0) or np.allclose(vec2, 0):
        return 0.0
    
    # Try standard cosine distance
    try:
        dist = cosine(vec1, vec2)
        if np.isnan(dist):
            # Fallback to manual calculation
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 * norm2 == 0:
                return 0.0
            cos_sim = dot / (norm1 * norm2)
            # Clip to valid range
            cos_sim = np.clip(cos_sim, -1, 1)
            return 1 - cos_sim
        return dist
    except:
        return 0.0

def measure_token_updates(
    model, 
    tokenizer, 
    base_text: str, 
    continuation: str,
    layers_to_analyze: List[int] = None
) -> Dict:
    """Fixed version with better debugging"""
    
    # Tokenize inputs
    base_ids = tokenizer(base_text, return_tensors='pt').input_ids
    full_text = base_text + " " + continuation
    full_ids = tokenizer(full_text, return_tensors='pt').input_ids
    
    # Get hidden states
    with torch.no_grad():
        base_outputs = model(base_ids, output_hidden_states=True)
        full_outputs = model(full_ids, output_hidden_states=True)
    
    base_hidden = base_outputs.hidden_states
    full_hidden = full_outputs.hidden_states
    
    if layers_to_analyze is None:
        layers_to_analyze = range(len(base_hidden))
    
    all_distances = []
    
    # Only compare tokens that exist in base text
    num_base_tokens = base_ids.shape[1]
    
    for layer_idx in layers_to_analyze:
        layer_base = base_hidden[layer_idx][0]  # [seq_len, hidden_dim]
        layer_full = full_hidden[layer_idx][0]
        
        # Compare each base token
        for token_idx in range(num_base_tokens):
            vec_base = layer_base[token_idx].cpu().numpy()
            vec_full = layer_full[token_idx].cpu().numpy()
            
            distance = safe_cosine_distance(vec_base, vec_full)
            all_distances.append(distance)
    
    all_distances = np.array(all_distances)
    
    # Debug info
    if all_distances.max() == 0:
        print(f"WARNING: All distances are 0!")
        print(f"Base shape: {base_ids.shape}, Full shape: {full_ids.shape}")
        print(f"Num distances calculated: {len(all_distances)}")
    
    return {
        'distances': all_distances,
        'distribution_stats': {
            'mean': np.mean(all_distances),
            'std': np.std(all_distances),
            'max': np.max(all_distances),
            'min': np.min(all_distances),
            'non_zero_count': np.sum(all_distances > 0)
        }
    }
