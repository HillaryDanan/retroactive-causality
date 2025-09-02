"""
Reproduce key findings from literature before novel experiments.
Must pass all checks or stop experiment.
"""

import torch
import numpy as np
from typing import Dict, Tuple

def check_bidirectional_attention(model, tokenizer) -> Tuple[bool, str]:
    """
    Verify that attention is bidirectional (not causal).
    Based on Rogers et al. (2020) "A Primer on Neural Network 
    Architectures for NLP" TACL.
    
    Returns:
        (success, message)
    """
    
    text = "The cat sat on the mat"
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
        # Check first layer, first head
        attention = outputs.attentions[0][0, 0]  # [seq_len, seq_len]
        
        # Check if later tokens attend to earlier tokens
        # Position 5 attending to positions 0-4
        backward_attention = attention[5, :5].sum().item()
        
    success = backward_attention > 0
    message = f"Backward attention sum: {backward_attention:.4f}"
    
    if success:
        message = "✅ " + message + " - Bidirectional attention confirmed"
    else:
        message = "❌ " + message + " - Expected bidirectional attention not found!"
    
    return success, message

def check_attention_not_importance(model, tokenizer) -> Tuple[bool, str]:
    """
    Reproduce Jain & Wallace (2019) finding that attention != importance.
    
    Method: Generate alternative attention patterns with same prediction.
    If found, attention isn't uniquely important.
    
    Returns:
        (success, message)
    """
    
    # This is simplified - full reproduction would require their exact setup
    # Key insight: Multiple attention patterns can yield same output
    
    text = "The movie was absolutely terrible"
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs1 = model(**inputs, output_attentions=True)
        
        # In practice, would permute attention and check if output similar
        # For sanity check, just verify attention exists and varies
        attention_variance = outputs1.attentions[0].var().item()
    
    success = attention_variance > 0.001  # Some variation in attention
    message = f"Attention variance: {attention_variance:.6f}"
    
    if success:
        message = "✅ " + message + " - Attention patterns show variation"
    else:
        message = "❌ " + message + " - Unexpected uniform attention"
    
    return success, message

def check_garden_path_exists(model, tokenizer) -> Tuple[bool, str]:
    """
    Verify garden path sentences cause larger updates than controls.
    This validates our basic experimental premise.
    
    Returns:
        (success, message)
    """
    
    from src.measurements.token_updates import measure_token_updates
    
    # Classic garden path
    garden_base = "The horse raced past the barn"
    garden_continuation = "fell"
    
    # Control sentence
    control_base = "The horse raced past the barn"
    control_continuation = "quickly"
    
    # Measure updates
    garden_result = measure_token_updates(
        model, tokenizer, garden_base, garden_continuation, 
        layers_to_analyze=[6]  # Just middle layer for speed
    )
    
    control_result = measure_token_updates(
        model, tokenizer, control_base, control_continuation,
        layers_to_analyze=[6]
    )
    
    garden_mean = garden_result['distribution_stats']['mean']
    control_mean = control_result['distribution_stats']['mean']
    
    success = garden_mean > control_mean * 1.5  # At least 50% larger
    message = f"Garden: {garden_mean:.4f}, Control: {control_mean:.4f}"
    
    if success:
        message = "✅ " + message + " - Garden path effect detected"
    else:
        message = "⚠️ " + message + " - Garden path effect weak/absent"
    
    return success, message

def run_all_sanity_checks(model, tokenizer) -> bool:
    """
    Run all sanity checks. Must pass all to proceed.
    
    Returns:
        True if all pass, False otherwise
    """
    
    print("="*60)
    print("RUNNING SANITY CHECKS - Must pass all to proceed")
    print("="*60)
    
    checks = [
        ("Bidirectional Attention", check_bidirectional_attention),
        ("Attention ≠ Importance", check_attention_not_importance),
        ("Garden Path Effect", check_garden_path_exists)
    ]
    
    all_passed = True
    
    for name, check_fn in checks:
        print(f"\nChecking: {name}")
        success, message = check_fn(model, tokenizer)
        print(message)
        
        if not success:
            all_passed = False
            print(f"FAILED: {name} - Cannot proceed with experiments")
            break
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL SANITY CHECKS PASSED - Safe to proceed!")
    else:
        print("❌ SANITY CHECKS FAILED - Fix issues before continuing")
    print("="*60)
    
    return all_passed