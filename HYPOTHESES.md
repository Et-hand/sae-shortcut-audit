# Pre-registration

**Status: DRAFT.** Finalize at the end of Phase 1, commit, and do not edit after Phase 2 starts.

## H1: layer separation
The spurious attribute a (any identity mention) is linearly decodable from the residual stream, and its probe selectivity peaks at a different layer than the label y.
- Metric: probe accuracy minus control-task accuracy, per layer, for a and for y.
- Verdict rule: TODO (e.g. peak layers differ by >= 2).

## H2: SAE features track and drive the shortcut
An SAE trained at L* yields features whose activation tracks a, and ablating the top-k improves worst-group accuracy over random-k ablation.
- Layer selection rule for L*: TODO (fill from Phase 1 result).
- Feature ranking: high a-AUROC, low y-AUROC on a labeled train subsample.
- k values: {1, 5, 10, 25, 50, 100}.
- Control: random-k ablation, TODO draws per k.
- Metrics: worst-group accuracy and average accuracy, 3 seeds.

## H3: partial recovery
SAE editing closes less of the ERM -> DFR worst-group gap than DFR does.
- Metric: (WGA_method - WGA_ERM) / (WGA_DFR - WGA_ERM).

## Frozen decisions
- Pooling: last token.
- Max length: 300 tokens.
- Groups: 4-group (y x a) primary; WILDS official groups secondary.
