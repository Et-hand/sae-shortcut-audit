# Shortcut Learning Meets Interpretability

Can SAE-based tools locate a spurious "shortcut" feature inside a trained model and remove it at inference time, without retraining? How much of the worst-group accuracy gap does that close compared to retraining baselines (GroupDRO, DFR)?

- **Dataset:** CivilComments via [WILDS](https://github.com/p-lambda/wilds). Groups = toxicity (y) x identity mention (a).
- **Model:** [Pythia-160m](https://huggingface.co/EleutherAI/pythia-160m), linear head on the last-token residual stream.
- **Tools:** [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [SAELens](https://github.com/jbloomAus/SAELens), [LEACE](https://github.com/EleutherAI/concept-erasure).

## Hypotheses

See [`HYPOTHESES.md`](HYPOTHESES.md). Frozen before Phase 2.

- **H1:** a is linearly decodable from the residual stream and peaks at a different layer than y.
- **H2:** an SAE at that layer yields features that track a, and ablating them improves worst-group accuracy.
- **H3:** representation editing recovers only part of the retraining gain.

## Repo layout

```
configs/        one YAML per experiment
src/data/       loading, group definitions, tokenization
src/train/      ERM fine-tuning (+ GroupDRO in Phase 3)
src/eval/       the single eval harness every phase calls
src/probe/      activation caching + layer-wise linear probes (Phase 1)
src/sae/        SAE training, diagnostics, feature scoring (Phase 2)
src/intervene/  LEACE, SAE ablation, steering hooks (Phases 1-2)
scripts/        entry points
notebooks/      exploration only, nothing load-bearing
results/        one JSON per run, committed
```

## Phases

| Phase | Goal | Deliverable |
|---|---|---|
| 0 | Data, fine-tuned baseline, eval harness | Baseline table (3 seeds, mean +/- std) in `results/` |
| 1 | Layer-wise probes + controls, LEACE, pick L* | Probe-vs-layer figure, LEACE row, `HYPOTHESES.md` committed |
| 2 | SAE at L*, feature scoring, ablation, steering | SAE diagnostics, feature scatter, WGA vs. k with random controls |
| 3 | DFR + GroupDRO baselines | Full table, fraction of gap closed |
| 4 | Writeup | Paper-style PDF, blog post, one-command reproduction |

## Rules

1. Every run takes a seed. Every reported number averages 3 seeds.
2. Never touch test until the table is final.
3. Surprising numbers are bugs until proven otherwise.
4. Negative results go in the writeup.
5. Probes and erasers are never fit on val or test.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/group_counts.py        # Phase 0, task 1
```
