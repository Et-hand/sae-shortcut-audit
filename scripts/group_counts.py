"""Phase 0, task 1: print per-group counts for train and val.

Run from repo root:  python scripts/group_counts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transformer_lens  # noqa: F401,E402  (install check only)

from src.data.civilcomments import (  # noqa: E402
    group4_count_table,
    group_count_table,
    identity_fields,
    load_wilds,
    split_frame,
)

ds = load_wilds("data")
ids = identity_fields(ds)

for split in ["train", "val"]:
    df = split_frame(ds, split)
    print(f"\n===== {split} (n={len(df)}) =====")
    print("\n4-group partition (y x any identity):")
    print(group4_count_table(df).to_string(index=False))
    print("\nPer identity (overlapping):")
    print(group_count_table(df, ids).to_string())
