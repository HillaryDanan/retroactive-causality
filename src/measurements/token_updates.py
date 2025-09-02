"""
Core measurement functions - NO ARBITRARY THRESHOLDS
Measure full continuous distributions
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine
from tqdm import tqdm

def measure_token_updates(
    model, 
    tokenizer, 
    base_text: str, 
    continuation: str,
    layers_to_analyze: List[int] = None
) -> Dict:
    """
    Measure how token representations change when continuation is added.
    
    Based on methodology from Ethayarajh (2019) "How Contextual are 
    Contextualized Word Representations?" ACL.
    
    Args:
        model: Transformer model (GPT-2 or BERT)
        tokenizer: Corresponding tokenizer
        base_text: Initial sentence fragment
        continuation: Text to append (e.g., "fell" for garden path)
        layers_to_analyze: Which layers to analyze (None = all)
    
    Returns:
        Dictionary containing:
        - 'distances': Raw cosine distances for each token at each layer
        - 'distribution_stats': Mean, std, percentiles
        - 'per_token': Distance per token position
        - 'per_layer': Distance per layer
    """
    
    # Tokenize inputs
    base_ids = tokenizer(base_text, return_tensors='pt').input_ids
    full_text = base_text + " " + continuation
    full_ids = tokenizer(full_text, return_tensors='pt').input_ids
    
    # Get hidden states for both versions
    with torch.no_grad():
        base_outputs = model(base_ids, output_hidden_states=True)
        full_outputs = model(full_ids, output_hidden_states=True)
    
    base_hidden = base_outputs.hidden_states  # Tuple of tensors (layers)
    full_hidden = full_outputs.hidden_states
    
    # Determine which layers to analyze
    if layers_to_analyze is None:
        layers_to_analyze = range(len(base_hidden))
    
    # Measure distances - NO THRESHOLDING
    all_distances = []
    per_token_distances = {i: [] for i in range(base_ids.shape[1])}
    per_layer_distances = {l: [] for l in layers_to_analyze}
    
    for layer_idx in layers_to_analyze:
        layer_base = base_hidden[layer_idx][0]  # Remove batch dimension
        layer_full = full_hidden[layer_idx][0]
        
        # For each token in the base text
        for token_idx in range(min(base_ids.shape[1], full_ids.shape[1])):
            # Compute cosine distance
            vec_base = layer_base[token_idx].cpu().numpy()
            vec_full = layer_full[token_idx].cpu().numpy()
            
            # Cosine distance = 1 - cosine_similarity
            distance = cosine(vec_base, vec_full)
            
            # Store raw distance - no thresholding!
            all_distances.append(distance)
            per_token_distances[token_idx].append(distance)
            per_layer_distances[layer_idx].append(distance)
    
    # Compute distribution statistics
    all_distances = np.array(all_distances)
    
    return {
        'distances': all_distances,
        'distribution_stats': {
            'mean': np.mean(all_distances),
            'std': np.std(all_distances),
            'median': np.median(all_distances),
            'percentiles': {
                25: np.percentile(all_distances, 25),
                50: np.percentile(all_distances, 50),
                75: np.percentile(all_distances, 75),
                90: np.percentile(all_distances, 90),
                95: np.percentile(all_distances, 95),
                99: np.percentile(all_distances, 99)
            }
        },
        'per_token': {k: np.mean(v) for k, v in per_token_distances.items()},
        'per_layer': {k: np.mean(v) for k, v in per_layer_distances.items()},
        'raw_per_token': per_token_distances,
        'raw_per_layer': per_layer_distances
    }

def measure_batch_sentences(
    model,
    tokenizer,
    sentence_pairs: List[Tuple[str, str]],
    layers_to_analyze: List[int] = None
) -> np.ndarray:
    """
    Measure update distributions for multiple sentences.
    
    Args:
        sentence_pairs: List of (base, continuation) tuples
        
    Returns:
        Concatenated array of all distances
    """
    
    all_distances = []
    
    for base, continuation in tqdm(sentence_pairs, desc="Processing sentences"):
        result = measure_token_updates(
            model, tokenizer, base, continuation, layers_to_analyze
        )
        all_distances.extend(result['distances'])
    
    return np.array(all_distances)