"""Phases 1-2: inference-time edits via forward hooks.

Planned modules:
  leace.py  fit LeaceEraser on TRAIN activations at L* against a; apply via hook;
            also try L* +/- 2 and final layer
  ablate.py x' = x - sum_{i in S} f_i * W_dec[:, i] for top-k S, k in {1,5,10,25,50,100};
            random-k control (several draws). Subtract from the ORIGINAL x, never use x_hat.
  steer.py  x' = x - alpha * W_dec[:, i], sweep alpha
"""
