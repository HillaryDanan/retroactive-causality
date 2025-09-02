"""
BERT-specific garden path measurements
BERT CAN show retroactive updates because it's bidirectional!
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine
from tqdm import tqdm

def measure_bert_retroactive_updates(
    model, 
    tokenizer, 
    sentence_with_garden: str,
    sentence_control: str,
    target_word: str = None,
    layers_to_analyze: List[int] = None
) -> Dict:
    """
    Measure how BERT's representations change with garden path
    
    Args:
        sentence_with_garden: Full garden path sentence
        sentence_control: Control version without garden path
        target_word: Which word to focus on (e.g., "horse")
    """
    
    # Tokenize both versions
    garden_encoding = tokenizer(sentence_with_garden, return_tensors='pt')
    control_encoding = tokenizer(sentence_control, return_tensors='pt')
    
    garden_ids = garden_encoding.input_ids
    control_ids = control_encoding.input_ids
    
    # Get hidden states
    with torch.no_grad():
        garden_outputs = model(garden_ids, output_hidden_states=True)
        control_outputs = model(control_ids, output_hidden_states=True)
    
    garden_hidden = garden_outputs.hidden_states
    control_hidden = control_outputs.hidden_states
    
    if layers_to_analyze is None:
        layers_to_analyze = range(len(garden_hidden))
    
    # Find target word position if specified
    if target_word:
        # Convert to tokens to find position
        target_tokens = tokenizer.tokenize(target_word)
        garden_tokens = tokenizer.convert_ids_to_tokens(garden_ids[0])
        
        # Find position (accounting for [CLS])
        target_pos = None
        for i in range(len(garden_tokens)):
            if garden_tokens[i:i+len(target_tokens)] == target_tokens:
                target_pos = i
                break
        
        if target_pos is None:
            print(f"Warning: '{target_word}' not found, analyzing all tokens")
    else:
        target_pos = None
    
    all_distances = []
    per_layer_distances = {l: [] for l in layers_to_analyze}
    per_position_distances = {}
    
    # Calculate distances
    for layer_idx in layers_to_analyze:
        layer_garden = garden_hidden[layer_idx][0]
        layer_control = control_hidden[layer_idx][0]
        
        # Get minimum length (in case of tokenization differences)
        min_len = min(layer_garden.shape[0], layer_control.shape[0])
        
        for pos in range(1, min_len-1):  # Skip [CLS] and [SEP]
            if target_pos is not None and pos != target_pos:
                continue
                
            vec_garden = layer_garden[pos].cpu().numpy()
            vec_control = layer_control[pos].cpu().numpy()
            
            # Cosine distance
            dist = cosine(vec_garden, vec_control)
            
            if not np.isnan(dist):
                all_distances.append(dist)
                per_layer_distances[layer_idx].append(dist)
                
                if pos not in per_position_distances:
                    per_position_distances[pos] = []
                per_position_distances[pos].append(dist)
    
    all_distances = np.array(all_distances)
    
    return {
        'distances': all_distances,
        'distribution_stats': {
            'mean': np.mean(all_distances) if len(all_distances) > 0 else 0,
            'std': np.std(all_distances) if len(all_distances) > 0 else 0,
            'median': np.median(all_distances) if len(all_distances) > 0 else 0,
            'max': np.max(all_distances) if len(all_distances) > 0 else 0,
            'count': len(all_distances)
        },
        'per_layer': {k: np.mean(v) if v else 0 for k, v in per_layer_distances.items()},
        'target_word': target_word,
        'sentences': {
            'garden': sentence_with_garden,
            'control': sentence_control
        }
    }

def batch_measure_bert(
    model,
    tokenizer,
    garden_sentences: List[str],
    control_sentences: List[str],
    target_words: List[str] = None,
    layers_to_analyze: List[int] = None
) -> Dict:
    """Measure multiple sentence pairs"""
    
    assert len(garden_sentences) == len(control_sentences)
    
    all_results = []
    all_distances = []
    
    for i, (garden, control) in enumerate(tqdm(
        zip(garden_sentences, control_sentences), 
        total=len(garden_sentences),
        desc="Processing sentences"
    )):
        target = target_words[i] if target_words else None
        
        result = measure_bert_retroactive_updates(
            model, tokenizer,
            garden, control,
            target_word=target,
            layers_to_analyze=layers_to_analyze
        )
        
        all_results.append(result)
        all_distances.extend(result['distances'])
    
    return {
        'all_distances': np.array(all_distances),
        'individual_results': all_results,
        'global_stats': {
            'mean': np.mean(all_distances),
            'std': np.std(all_distances),
            'median': np.median(all_distances),
            'effect_size': None  # Will calculate vs control
        }
    }
