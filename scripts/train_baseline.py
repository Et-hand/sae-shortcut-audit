"""Phase 0: train ERM baseline over all seeds in the config and save results.

Run from repo root:  python scripts/train_baseline.py --config configs/phase0_baseline.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.civilcomments import CivilCommentsTorch, get_tokenizer, load_wilds, split_frame  # noqa: E402
from src.eval.metrics import summarize_seeds  # noqa: E402
from src.train.erm import train_erm  # noqa: E402
from src.train.model import PythiaClassifier  # noqa: E402
from src.utils import load_config, save_result, set_seed  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/phase0_baseline.yaml")
args = p.parse_args()
cfg = load_config(args.config)

wilds_ds = load_wilds(cfg["data"]["root_dir"])
tok = get_tokenizer(cfg["model"]["name"])
max_len = cfg["data"]["max_length"]

runs = []
for seed in cfg["seeds"]:
    set_seed(seed)
    train_df = split_frame(wilds_ds, "train", cfg["data"]["train_subsample"], seed)
    val_df = split_frame(wilds_ds, "val")
    train_ds = CivilCommentsTorch(train_df, tok, max_len)
    val_ds = CivilCommentsTorch(val_df, tok, max_len)

    model = PythiaClassifier(cfg["model"]["name"])
    run = train_erm(model, train_ds, val_ds, cfg, seed, wilds_ds)
    save_result(cfg["output"]["results_dir"], f"{cfg['name']}_seed{seed}", {"config": cfg, **run})
    runs.append(run["final_val"])

summary = summarize_seeds(runs)
save_result(cfg["output"]["results_dir"], f"{cfg['name']}_summary", {"split": "val", "summary": summary})
print(summary)
