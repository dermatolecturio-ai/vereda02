# VEREDA Research Log

This directory is the long-term memory of the VEREDA project. It records not
only successful ideas, but also negative results, failed assumptions, costs,
and decisions. An experiment that disproves a hypothesis is still a useful
result when it is reproducible.

## Structure

- `hypotheses/`: testable architectural and training hypotheses.
- `experiments/`: experiment plans and completed run reports.
- `results/`: consolidated positive, negative, and inconclusive findings.
- `decisions/`: architecture decision records explaining why a path changed.
- `references/`: notes about external papers and the author's research docs.

Every experiment must name its baseline, changed variable, seed, data,
configuration, metrics, and a falsification criterion before training starts.

## Current central artifact

- `references/unified_vereda_theory.md`: article-by-article synthesis,
  unified VEREDA-Final theory, accepted local proofs, missing proof
  obligations, and adaptation order for the final model.
- `experiments/math-validation/`: reproducible numerical proof bench and
  graphs for the accepted mathematical invariants.
