"""ERM fine-tuning loop. GroupDRO (Phase 3) will reuse this with a different loss."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.eval.metrics import evaluate
from src.utils import device


@torch.no_grad()
def predict(model, loader, dev) -> dict:
    model.eval()
    preds, ys, groups, idxs = [], [], [], []
    for b in loader:
        logits = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
        preds.append(logits.argmax(-1).cpu())
        ys.append(b["y"])
        groups.append(b["group4"])
        idxs.append(b["idx"])
    return {k: torch.cat(v).numpy() for k, v in
            {"pred": preds, "y": ys, "group4": groups, "idx": idxs}.items()}


def train_erm(model, train_ds, val_ds, cfg: dict, seed: int, wilds_ds=None) -> dict:
    dev = device()
    model.to(dev)
    tc = cfg["train"]

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"] * 2)

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps = len(train_loader) * tc["epochs"] // tc["grad_accum"]
    sched = get_linear_schedule_with_warmup(opt, int(tc["warmup_ratio"] * steps), steps)
    use_amp = tc.get("fp16", False) and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = []
    for epoch in range(tc["epochs"]):
        model.train()
        for step, b in enumerate(train_loader):
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
                loss = loss_fn(logits, b["y"].to(dev)) / tc["grad_accum"]
            scaler.scale(loss).backward()
            if (step + 1) % tc["grad_accum"] == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()

        p = predict(model, val_loader, dev)
        metrics = evaluate(p["pred"], p["y"], p["group4"], wilds_ds, p["idx"])
        metrics["epoch"] = epoch
        history.append(metrics)
        print(f"[seed {seed}] epoch {epoch}: avg {metrics['avg_acc']:.4f} "
              f"worst {metrics['worst_group4_acc']:.4f} ({metrics['worst_group4']})")

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"seed{seed}.pt")
    return {"seed": seed, "val_history": history, "final_val": history[-1]}
