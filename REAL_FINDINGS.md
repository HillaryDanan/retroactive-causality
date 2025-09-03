# The Real Discovery: Semantic Incongruity Drives Retroactive Updates

## What We Actually Found
After correcting methodological errors, the data reveals:

**Retroactive updates in BERT are driven by semantic incongruity, not syntactic garden paths.**

### Key Evidence
1. Semantically incongruent adverbs cause 7.5x larger updates than congruent ones (p=0.0013)
2. "The horse raced past the barn yesterday" (syntactically valid but semantically odd) causes larger updates than garden paths
3. Updates scale with semantic unexpectedness, not syntactic category violation

### The Mechanism
When BERT encounters semantically unexpected continuations, it retroactively adjusts earlier token representations to accommodate the surprising input, even when syntax is perfectly valid.

This is a different and more general phenomenon than garden path processing.
