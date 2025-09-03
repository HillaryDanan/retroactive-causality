# Semantic Incongruity Drives Retroactive Updates in Bidirectional Transformers

## Main Discovery
**Semantic incongruity causes retroactive representation updates ONLY in bidirectional transformers** (p=0.0013)

## Key Validated Findings

### 1. Core Effect
- Temporal words after locations cause 7.5x larger updates than semantically congruent words
- "barn yesterday" (0.0538) vs "barn quickly" (0.0072)
- Effect is semantic, not syntactic

### 2. Architecture Dependency (NEW)
- **BERT** (bidirectional): 0.0444 effect
- **RoBERTa** (bidirectional): 0.0184 effect  
- **GPT-2** (unidirectional): 0.0000 effect
- **Conclusion**: Retroactive updates require bidirectional attention

### 3. Mechanism
- Semantic incongruity forces the model to retroactively adjust earlier token representations
- Temporal words are uniquely disruptive (violate event-frame coherence)
- Effect scales with semantic distance from expected continuation

## What This Means
Bidirectional transformers maintain semantic coherence constraints that operate retroactively. When these constraints are violated, the model adjusts earlier representations to accommodate the surprising input.

## Limitations & Corrections
- Initial "causality" claim was methodologically flawed (now corrected)
- Garden paths are too heterogeneous to treat as unified phenomenon
- Measurement methodology drastically affects results

## Repository
https://github.com/HillaryDanan/retroactive-causality

All code, data, and corrections transparently documented.
