# Retroactive Information Flow in Bidirectional Transformers: Quantifying Semantic Reconfiguration in Garden Path Sentences

## 1. Introduction

### 1.1 Background

Garden path sentences force semantic reanalysis when late-arriving information disambiguates earlier ambiguous structure (Ferreira & Henderson, 1991, *Journal of Memory and Language*). The canonical example "The horse raced past the barn fell" requires retroactive reinterpretation: "raced" shifts from main verb to reduced relative clause upon encountering "fell." This phenomenon has been extensively documented in human psycholinguistics (Christianson et al., 2001, *Cognitive Psychology*).

Transformer language models employ self-attention mechanisms that theoretically permit bidirectional information flow (Vaswani et al., 2017, *NeurIPS*). However, the extent to which this architecture enables measurable retroactive updates analogous to human garden path processing remains unexplored.

### 1.2 Theoretical Framework

Information theory provides constraints on semantic reconfiguration. Shannon's channel capacity theorem (Shannon, 1948, *Bell System Technical Journal*) establishes fundamental limits on information transmission. For a vocabulary of size V, each token carries at most log₂(V) bits of information. BERT's vocabulary of 30,522 tokens yields approximately 14.9 bits per token.

We hypothesize that disambiguation in garden path sentences induces quantifiable retroactive updates in earlier token representations, with magnitude bounded by information-theoretic constraints.

### 1.3 Model Architecture Considerations

Critical distinction: GPT-2 employs causal masking, preventing retroactive influence (Radford et al., 2019, *OpenAI Blog*). BERT uses bidirectional attention, enabling information flow in both directions (Devlin et al., 2019, *NAACL*). This architectural difference is fundamental to our investigation.

## 2. Methods

### 2.1 Model Specification

- **Model**: BERT-base-uncased (Devlin et al., 2019)
- **Parameters**: 109.5M
- **Layers**: 12 transformer blocks
- **Hidden dimension**: 768
- **Attention heads**: 12

### 2.2 Stimuli

Twenty garden path sentences from psycholinguistic literature paired with grammatically standard controls:

**Example pair**:
- Garden: "The horse raced past the barn fell"
- Control: "The horse raced past the barn quickly"

Target words (e.g., "horse") were identified as early tokens requiring retroactive reinterpretation upon disambiguation.

### 2.3 Measurement Protocol

For each sentence pair:

1. Extract hidden states for target tokens from layers {0, 3, 6, 9, 11}
2. Calculate cosine distance between garden path and control representations:
   ```
   d = 1 - cos(h_garden, h_control)
   ```
3. Aggregate distances across layers and tokens

### 2.4 Statistical Analysis

- **Primary test**: One-sample t-test against null hypothesis (d = 0)
- **Effect size**: Cohen's d = μ/σ
- **Robustness**: Bootstrap confidence intervals (10,000 iterations)
- **Information content**: Δbits = d × log₂(30,522)

## 3. Results

### 3.1 Primary Findings

Analysis of 655 measurements across 20 sentence pairs revealed significant retroactive updates:

- **Mean distance**: 0.1091 (SD = 0.2012)
- **Median**: 0.0285
- **95% CI**: [0.0941, 0.1251] (bootstrap)
- **Statistical significance**: t(654) = 13.872, p = 1.53 × 10⁻³⁸
- **Effect size**: Cohen's d = 0.542 (medium)

### 3.2 Information-Theoretic Quantification

- **Mean information update**: 1.626 bits
- **Maximum observed**: 12.6 bits (84.5% of theoretical maximum)
- **95th percentile**: 6.3 bits

### 3.3 Layer-wise Analysis

Retroactive effects varied across layers:
- Embedding layer (0): Minimal updates
- Middle layers (3-6): Peak effect magnitude
- Final layers (9-11): Moderate updates

This pattern suggests semantic reconfiguration occurs primarily in intermediate representations.

## 4. Discussion

### 4.1 Theoretical Implications

The observed retroactive updates demonstrate that bidirectional transformers exhibit measurable semantic reconfiguration analogous to human garden path processing. The consistent ~1.6 bit average update suggests fundamental constraints on information flow during reanalysis.

### 4.2 Comparison to Human Processing

Human garden path recovery involves measurable processing costs (Ferreira & Henderson, 1991). Our findings suggest BERT exhibits similar computational signatures, though the mechanism differs fundamentally from incremental human parsing.

### 4.3 Information-Theoretic Bounds

The 10.91% average vector change represents substantial but bounded reconfiguration. This aligns with minimum description length principles (Rissanen, 1978, *Automatica*): disambiguation requires finite information to propagate backwards through the network.

## 5. Limitations

### 5.1 Methodological Constraints

1. **Single architecture**: Results specific to BERT; generalization requires testing additional bidirectional models
2. **Semantic similarity**: Cosine distance captures geometric but not necessarily semantic change
3. **Static analysis**: Measurements taken post-hoc rather than during processing

### 5.2 Theoretical Limitations

1. **Mechanism unclear**: We quantify retroactive updates but don't identify causal mechanisms
2. **Human analogy imperfect**: BERT processes entire sequences simultaneously, unlike incremental human parsing
3. **Control selection**: Alternative control sentences might yield different effect magnitudes

## 6. Future Directions

### 6.1 Immediate Extensions

1. Test additional bidirectional models (RoBERTa, ELECTRA)
2. Examine attention weight redistribution during updates
3. Probe specific syntactic features driving reconfiguration

### 6.2 Theoretical Development

1. Develop formal model of information flow constraints
2. Connect findings to predictive coding frameworks
3. Investigate relationship to human ERP components (N400, P600)

## 7. Conclusion

We present empirical evidence that garden path sentences induce statistically significant retroactive updates in BERT's token representations (p = 1.53 × 10⁻³⁸, d = 0.542). The measured information-theoretic bound of ~1.6 bits suggests fundamental constraints on semantic reconfiguration in bidirectional transformers.

These findings bridge computational linguistics and psycholinguistics, providing quantitative evidence that neural language models exhibit processing signatures analogous to human garden path phenomena, albeit through fundamentally different mechanisms.

## References

Christianson, K., Hollingworth, A., Halliwell, J. F., & Ferreira, F. (2001). Thematic roles assigned along the garden path linger. *Cognitive Psychology*, 42(4), 368-407.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*, 4171-4186.

Ferreira, F., & Henderson, J. M. (1991). Recovery from misanalyses of garden-path sentences. *Journal of Memory and Language*, 30(6), 725-745.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*, 1(8), 9.

Rissanen, J. (1978). Modeling by shortest data description. *Automatica*, 14(5), 465-471.

Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

---

## Data and Code Availability

All data, code, and analysis scripts available at: https://github.com/hillarydanan/retroactive-causality

## Author Note

This work represents initial exploration of information-theoretic bounds on semantic reconfiguration. We emphasize that while statistically robust, these findings require replication and theoretical development to fully understand the mechanisms underlying retroactive information flow in neural language models.