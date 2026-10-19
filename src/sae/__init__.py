"""Phase 2: SAE at layer L*.

Planned modules:
  train_sae.py  SAELens, all token positions, expansion 8x then 16x, sweep L1 coeff
  diagnose.py   L0, variance explained, dead-feature fraction, spliced-accuracy drop
  score.py      per-feature AUROC for a and for y, ranking, max-activating examples,
                a-vs-y AUROC scatter (the key figure)
"""
