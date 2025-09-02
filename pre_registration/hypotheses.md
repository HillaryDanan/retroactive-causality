# pre_registration/hypotheses.md

Study: Information-Theoretic Bounds on Semantic Reconfiguration in Transformers

Date: September 2 2025
OSF Link: https://osf.io/z7atq/

## Primary Hypothesis
H1: Garden path sentences will show significantly different token update 
    distributions compared to control sentences when measured via cosine 
    distance of hidden states across transformer layers.

## Predictions
- Effect size: Cohen's d > 0.8 (large effect)
- Statistical test: Kolmogorov-Smirnov test, p < 0.05
- Distribution difference: Wasserstein distance > 0.1
- Direction: Garden path updates > Control updates

## What We Do NOT Predict
- Specific percentage of tokens updated (no threshold)
- Exact magnitude of updates
- Which layers show strongest effect (exploratory)

## Analysis Plan
1. Measure full distributions without thresholding
2. Compare using multiple metrics (KS, Wasserstein, Cohen's d)
3. Show robustness across any threshold choice
4. Report all results regardless of outcome

## Sample Size
- Minimum 20 garden path sentences (from literature)
- Minimum 20 control sentences (length-matched)
- Power analysis: 95% power to detect d=0.8 at α=0.05

## Corrections for Multiple Comparisons
- Bonferroni correction for multiple statistical tests
- Adjusted α = 0.05 / number of tests

## Commitment to Transparency
- All negative results will be reported
- Raw data will be made publicly available
- Code will be open-sourced on GitHub