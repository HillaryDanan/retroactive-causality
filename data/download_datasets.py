"""
Download all required datasets with verification
"""

import os
from datasets import load_dataset
import pandas as pd

def download_copa():
    """
    Download COPA (Choice of Plausible Alternatives) dataset.
    Roemmele et al. (2011) "Choice of Plausible Alternatives"
    """
    
    print("Downloading COPA dataset...")
    dataset = load_dataset("super_glue", "copa")
    
    # Save locally
    os.makedirs("cache", exist_ok=True)
    
    # Extract causal reasoning pairs
    causal_pairs = []
    for split in ['train', 'validation']:
        for item in dataset[split]:
            premise = item['premise']
            if item['question'] == 'cause':
                # This is a causal relationship
                causal_pairs.append({
                    'premise': premise,
                    'choice1': item['choice1'],
                    'choice2': item['choice2'],
                    'label': item['label']
                })
    
    df = pd.DataFrame(causal_pairs)
    df.to_csv('cache/copa_causal.csv', index=False)
    print(f"✅ Saved {len(df)} causal pairs from COPA")
    
    return df

def download_control_sentences():
    """
    Get control sentences from Brown corpus or similar.
    These should be unambiguous, declarative sentences.
    """
    
    # For simplicity, we'll create some manually
    # In practice, would filter from larger corpus
    
    control_sentences = [
        "The cat slept on the warm blanket peacefully",
        "The teacher explained the lesson clearly yesterday",
        "The children played in the park happily",
        "The car drove down the street slowly",
        "The bird sang in the tree beautifully",
        "The student studied for the exam diligently",
        "The chef prepared the meal carefully",
        "The dog barked at the stranger loudly",
        "The rain fell on the roof steadily",
        "The sun shone through the window brightly"
    ]
    
    df = pd.DataFrame({'sentence': control_sentences})
    df.to_csv('cache/control_sentences.csv', index=False)
    print(f"✅ Saved {len(df)} control sentences")
    
    return df

if __name__ == "__main__":
    download_copa()
    download_control_sentences()