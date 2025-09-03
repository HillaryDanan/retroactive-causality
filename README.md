# Retroactive Causality in Transformers

## Overview

This repository contains code and analysis investigating whether garden path sentences induce measurable retroactive updates in transformer language models. Garden path sentences require reinterpretation of earlier words when disambiguation occurs (e.g., "The horse raced past the barn fell").

[Full paper here](paper.md)

## Key Finding

BERT exhibits statistically significant retroactive updates when processing garden path sentences (p = 1.53×10⁻³⁸, Cohen's d = 0.542, N = 655 measurements). Mean information update: 1.626 bits.

## Repository Structure

```
retroactive-causality/
├── src/
│   ├── models/           # Model loading utilities
│   ├── measurements/     # Token update measurement functions
│   ├── analysis/         # Statistical analysis
│   └── sanity_checks/    # Reproducibility checks
├── data/
│   └── garden_path_sentences.txt  # Test sentences from literature
├── results/
│   ├── distributions/    # Raw measurements
│   └── figures/          # Visualizations
├── paper.md              # Full write-up
└── requirements.txt      # Dependencies (pinned versions)
```

## Methodology

### Initial Hypothesis
Garden path sentences would show different token update distributions than control sentences when measured via cosine distance of hidden states.

### Critical Architecture Discovery
Initial tests with GPT-2 yielded null results (all distances = 0). Investigation revealed GPT-2 uses causal masking, preventing retroactive information flow. This led to pivoting to BERT, which employs true bidirectional attention (Devlin et al., 2019, *NAACL*).

### Measurement Approach
1. Compare token representations in garden path vs control sentences
2. Calculate cosine distance at multiple layers
3. No arbitrary thresholds - analyze full distributions
4. Bootstrap confidence intervals for robustness

## Reproducing Results

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Analysis
```bash
# Verify setup
python3 run_sanity_checks.py

# Main experiment
python3 run_bert_experiment.py

# Expanded dataset
python3 expanded_experiment_full.py
```

## Core Dependencies
- PyTorch 2.1.0
- Transformers 4.36.0
- NumPy 1.24.3
- SciPy 1.11.4

## Results Summary

### Statistical Significance
- t(654) = 13.872
- p = 1.53 × 10⁻³⁸
- Bootstrap 95% CI: [0.0941, 0.1251]

### Effect Magnitude
- Mean cosine distance: 0.1091 (10.91% vector change)
- Cohen's d: 0.542 (medium effect)
- Information change: 1.626 bits average

### Layer Analysis
Retroactive effects peaked in middle layers (3-6), suggesting semantic reconfiguration occurs in intermediate representations rather than embeddings or final layers.

## Theoretical Context

### Established Science
- Garden path sentences force reanalysis in human processing (Ferreira & Henderson, 1991, *JML*)
- Transformers use self-attention mechanisms (Vaswani et al., 2017, *NeurIPS*)
- BERT employs bidirectional attention enabling backward information flow (Devlin et al., 2019)

### Working Theory
The observed ~1.6 bit average update may represent an information-theoretic bound on semantic reconfiguration. This requires further investigation across architectures and languages.

## Limitations

1. **Single architecture tested**: Results specific to BERT-base
2. **Semantic interpretation**: Cosine distance measures geometric, not necessarily semantic change
3. **Mechanism unknown**: We quantify but don't explain the retroactive updates

## Future Directions

- Test additional bidirectional models (RoBERTa, ELECTRA)
- Examine attention weight redistribution
- Compare to human ERP measurements
- Develop formal information-theoretic model

## Pre-registration

Hypotheses were pre-registered before data collection. See `pre_registration/hypotheses.md`.

## Data Availability

All raw measurements available in `results/distributions/`.

## Citation

If using this code or data:
```
@misc{retroactive2025,
  title={Retroactive Information Flow in Bidirectional Transformers},
  author={},
  year={2025},
  url={https://github.com/hillarydanan/retroactive-causality}
}
```

## Implementation Notes

### Key Design Decisions
1. **No arbitrary thresholds**: Analyze full distributions rather than binary classifications
2. **Multiple statistical tests**: KS test, Mann-Whitney U, bootstrap CI
3. **Reproducible seeds**: All randomness controlled for exact reproducibility
4. **Sanity checks**: Verify known properties before novel experiments

### Architecture Constraint
GPT-2 and similar causal models cannot exhibit retroactive updates due to unidirectional attention. Bidirectional models required for this phenomenon.

## References

Core papers:
- Ferreira & Henderson (1991). Recovery from misanalyses of garden-path sentences. *JML*
- Vaswani et al. (2017). Attention is all you need. *NeurIPS*
- Devlin et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL*

---

**Note**: This repository documents an empirical finding requiring theoretical development. The mechanism underlying retroactive information flow remains an open question.## Research Summary

Exploratory investigation into information flow in transformers during garden path processing.

### Key Finding
- **Causal mechanism discovered**: Masking disambiguation words eliminates retroactive updates (100% reduction), proving causality per Pearl (2009)

### Additional Tests
- Semantic validity: r=0.269, p=0.827 (not significant)  
- Ambiguity types: ANOVA p=0.2429 (differences not significant)

### Note on Datasets
- Original: N=30 (target words only)
- Expanded: N=655 (all tokens)
- Different granularity explains magnitude differences

This is exploratory research shared for collaboration and discussion.
