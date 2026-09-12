"""Training loop.

    python -m src.train

bf16 autocast + `torch.compile` + fused AdamW. The LR schedule is warmup-stable-decay (WSD).
"""

import math
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR
from tqdm import tqdm

from .config import get_config, ckpt_path, final_path
from .data import get_ds
from .model import LM

# setting global seed for reproductibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode() # inference_mode is faster than torch.no_grad()
def evaluate(model, val_dl, vocab_size, device):
    """Mean cross-entropy over the whole validation split. This uses
    unsmoothed CE so the number stays comparable across label_smoothing settings."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    for batch in val_dl:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            total += loss_fn(logits.view(-1, vocab_size), labels.view(-1)).item()
    model.train()
    return total / len(val_dl)


def train_model(cfg: dict):
    assert torch.cuda.is_available(), "this codebase is CUDA-only"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    train_dl, val_dl, tok = get_ds(cfg)
    set_seed(cfg["seed"])

    raw_model = LM.from_config(cfg, tok.get_vocab_size()).to(device)
    print(f"parameters: {sum(p.numel() for p in raw_model.parameters()):,}")
    model = torch.compile(raw_model)

    # ---- schedule: linear warmup -> constant -> linear decay to zero -------- #
    total_steps = len(train_dl) * cfg["num_epochs"]
    warmup_steps = max(100, int(cfg["warmup_ratio"] * total_steps))
    decay_steps = int(cfg["decay_ratio"] * total_steps)
    stable_steps = total_steps - warmup_steps - decay_steps

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=cfg["betas"],
                            weight_decay=cfg["weight_decay"], fused=True)
    scheduler = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
            ConstantLR(opt, factor=1.0, total_iters=stable_steps),
            LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=decay_steps),
        ],
        milestones=[warmup_steps, warmup_steps + stable_steps],
    )

    val_every = max(1, round(cfg["val_every_pct"] * total_steps))
    ckpt_every = max(1, round(cfg["ckpt_every_pct"] * total_steps))
    print(f"steps {total_steps:,} | warmup {warmup_steps:,} | stable {stable_steps:,} "
          f"| decay {decay_steps:,} | val every {val_every:,}")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    vocab = tok.get_vocab_size()
    val_hist, step, seen, ema = [], 0, 0, None

    for epoch in range(cfg["num_epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        it = tqdm(train_dl, desc=f"epoch {epoch:02d}", file=sys.stdout, dynamic_ncols=True)

        for batch in it:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, vocab), labels.view(-1))
            loss.backward()

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            seen += labels.size(0)

            cur = loss.item()
            ema = cur if ema is None else 0.98 * ema + 0.02 * cur
            it.set_postfix(avg=f"{ema:6.3f}", gn=f"{gn:5.2f}",
                           lr=f"{opt.param_groups[0]['lr']:.2e}")

            if step % val_every == 0:
                vl = evaluate(raw_model, val_dl, vocab, device)
                val_hist.append((step, vl))
                it.write(f"[{100 * step / total_steps:5.1f}% | step {step:,} | {seen:,} seqs] "
                         f"val loss {vl:6.3f} | val ppl {math.exp(vl):8.2f}")

            if step % ckpt_every == 0:
                torch.save({"model": raw_model.state_dict(), "cfg": cfg,
                            "step": step, "val_hist": val_hist}, ckpt_path(cfg))

        it.close()
        vl = evaluate(raw_model, val_dl, vocab, device)
        print(f"epoch {epoch:02d} | train {ema:6.3f} | val loss {vl:6.3f} "
              f"| val ppl {math.exp(vl):8.2f}")

    return raw_model, tok, val_hist


if __name__ == "__main__":
    cfg = get_config()
    model, tok, val_hist = train_model(cfg)
    torch.save({"model": model.state_dict(), "cfg": cfg, "val_hist": val_hist},
               final_path(cfg))
    print(f"saved {final_path(cfg)}")
