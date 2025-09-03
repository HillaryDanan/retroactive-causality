# Errata

## Intervention Gradient Test (test_intervention_gradient.py)
Initial version compared sentences with identical endings, which trivially produces 0 distance.
See `test_intervention_gradient_corrected.py` for the proper methodology comparing different endings.
The corrected version properly tests how different disambiguations affect retroactive updates.
