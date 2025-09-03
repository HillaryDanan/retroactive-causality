# Language Models Don't Encode Time: Implications for Human-AI Interaction

## Key Empirical Finding
When tested on temporal paradoxes ("Yesterday [MASK] tomorrow"), BERT and RoBERTa predict generic connectors ("and", ",") rather than temporal relationships. They treat impossible temporal sequences like word lists, not causal chains.

## Data
- BERT: 40.8% "and", 11.5% "or" for temporal impossibilities
- RoBERTa: 69.2% "and", 11.9% ","
- Neither model predicts temporal operators (before/after/causes)
- For "Entropy [MASK] spontaneously", BERT assigns only 9.4% to "increases" (the physically correct answer)

## What This Reveals
Language models learn statistical co-occurrence, not temporal logic. The retroactive updates we observed aren't about temporal coherence but about violating learned word patterns.

## The Fundamental Gap
Humans experience time as:
- Causal flow (A causes B)
- Irreversibility (can't unbreak eggs)
- Memory and anticipation

LLMs process language as:
- Statistical patterns
- Bidirectional attention (BERT) or unidirectional (GPT)
- No inherent temporal model

## Implications for Human-LLM Interaction

### What This Predicts
1. LLMs will confidently generate temporally incoherent narratives if statistically plausible
2. They can't truly reason about causality, only pattern-match causal language
3. Temporal reasoning tasks will reveal systematic failures

### Open Questions
- Can we identify "temporal illusions" - where LLMs appear to understand time but don't?
- Do humans unconsciously compensate for LLMs' temporal blindness during interaction?
- Could this explain why LLMs struggle with planning and long-term coherence?

## The Deeper Question
If language evolved from beings who experience time, but LLMs learn language without temporal experience, what other fundamental human concepts are they missing? Causality? Mortality? Change?

This gap might be unbridgeable with current architectures.

## Repository Status
- Main finding stands: Semantic incongruity drives retroactive updates (p=0.0013)
- Temporal exploration reveals architectural limitations
- All data and corrections transparently documented
