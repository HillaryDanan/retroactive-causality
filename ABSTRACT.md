# Retroactive Causality in Transformer Language Models

## Abstract

We present empirical evidence that garden path sentences induce measurable retroactive updates in BERT's token representations. Using 20 sentence pairs from psycholinguistic literature, we measured cosine distances between token representations before and after disambiguation points. 

**Key Findings:**
- Significant retroactive updates detected (p = 1.53×10⁻³⁸, N = 655)
- Medium effect size (Cohen's d = 0.542)
- Mean information update: 1.626 bits (10.91% vector change)
- Bootstrap 95% CI: [0.0941, 0.1251]

These results demonstrate that bidirectional transformers exhibit information-theoretic bounds on semantic reconfiguration analogous to human garden path processing. The consistent ~1.6 bit update suggests fundamental constraints on retroactive information flow in neural language models.

**Implications:** This work bridges psycholinguistics and information theory, providing quantitative bounds on how disambiguation propagates backwards through transformer representations.

Code and data: https://github.com/hillarydanan/retroactive-causality
