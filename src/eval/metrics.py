"""The single eval harness. Every phase calls evaluate(); never compute accuracy ad hoc."""
import numpy as np
import torch

from src.data.civilcomments import GROUP4_NAMES


def evaluate(y_pred, y_true, group4, wilds_ds=None, wilds_idx=None) -> dict:
    """Average acc, per-group acc, worst-group acc (4-group), and optionally WILDS official metrics.

    Args:
        y_pred, y_true, group4: 1-D arrays/tensors of equal length.
        wilds_ds: the full WILDS dataset (to run its official eval).
        wilds_idx: indices into wilds_ds for each example (to pull metadata).
    """
    y_pred = np.asarray(y_pred).astype(int)
    y_true = np.asarray(y_true).astype(int)
    group4 = np.asarray(group4).astype(int)
    correct = y_pred == y_true

    out = {"n": int(len(y_true)), "avg_acc": float(correct.mean()), "group4": {}}
    accs = []
    for g in range(4):
        m = group4 == g
        acc = float(correct[m].mean()) if m.any() else float("nan")
        out["group4"][GROUP4_NAMES[g]] = {"n": int(m.sum()), "acc": acc}
        if m.any():
            accs.append((acc, GROUP4_NAMES[g]))
    worst = min(accs)
    out["worst_group4_acc"], out["worst_group4"] = worst

    if wilds_ds is not None and wilds_idx is not None:
        meta = wilds_ds.metadata_array[torch.as_tensor(wilds_idx)]
        results, _ = wilds_ds.eval(torch.as_tensor(y_pred), torch.as_tensor(y_true), meta)
        out["wilds_official"] = {k: float(v) for k, v in results.items()}

    return out


def summarize_seeds(runs: list[dict], keys=("avg_acc", "worst_group4_acc")) -> dict:
    """Mean and std across seeds for the headline metrics."""
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in runs])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    return summary
