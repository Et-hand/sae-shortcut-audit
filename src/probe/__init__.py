"""Phase 1: activation caching + layer-wise linear probes.

Planned modules:
  cache.py   run_with_cache on ~20k train examples, store pooled resid at every layer
             -> array [N, 13, 768] (embed + 12 blocks), hook blocks.{L}.hook_resid_post
  probes.py  per-layer logistic regression for a and y (standardize, tune C on val),
             control task with shuffled labels, selectivity = real - control
"""
