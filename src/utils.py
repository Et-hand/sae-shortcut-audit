"""Seeding, config loading, and result saving shared by every phase."""
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def save_result(results_dir: str | Path, run_name: str, payload: dict) -> Path:
    """One JSON per run. Includes timestamp and commit so every number is traceable."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run": run_name, "time": datetime.now().isoformat(), "commit": git_hash(), **payload}
    out = results_dir / f"{run_name}.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
