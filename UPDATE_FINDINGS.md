# Important Update on Causality Finding

## Correction
The initial "100% causal reduction with masking" finding was based on a methodological error. 
The test compared identical sentences (both ending with [MASK]), which trivially produces 0 distance.

## Actual Finding
When properly tested:
- Baseline distance (fell vs quickly): 0.0187
- With [MASK] (masked vs quickly): 0.0347
- Result: [MASK] INCREASES distance by 85%, not reduces it

## What This Means
The retroactive updates are not simply "caused" by the disambiguation word in the way initially thought.
Instead, [MASK] appears to create uncertainty that amplifies differences between syntactic structures.

## Status
- Original hypothesis: Refuted
- New hypothesis: [MASK] amplifies structural ambiguity effects
- Further investigation needed
