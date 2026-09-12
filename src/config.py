"""Hyperparameters and cache-path helpers."""

from pathlib import Path


def get_config() -> dict:
    return {
        # ---- data ----
        "datasource": "roneneldan/TinyStories",
        "cache_dir": "cache",
        "num_rows": 2_110_000,   # approximately all of the TinyStories train split
        "vocab_size": 8192,      # you can even reduce the size of the vocabulary

        # ---- model ----
        "d_model": 768,          # size of an embedding vector
        "N": 12,                 # decoder blocks
        "h": 12,                 # attention heads
        "max_seq": 512,          # training context length
        "rope_max_seq": 1024,    # RoPE table length, so we can decode past 512

        # ---- optimization ----
        "num_epochs": 2,
        "batch_size": 128,
        "lr": 1e-3,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "warmup_ratio": 0.03,    # warmup / stable / linear-decay schedule
        "decay_ratio": 0.15,
        "label_smoothing": 0.0,
        "seed": 0,

        # ---- logging / checkpointing ----
        "val_every_pct": 0.05,   # run validation after every 5% of total steps
        "ckpt_every_pct": 0.25,	 # save checkpoint after every 25% of total steps
        "num_workers": 4,
    }


def cache_dir(cfg: dict) -> Path:
    d = Path(cfg["cache_dir"])
    d.mkdir(parents=True, exist_ok=True)   # create the folder in the filesystem and any missing parents, if the folder already exists no error is raised
    return d


# tokenizers path will depend on the dataset (and its size) as well as the vocab_size
def tokenizer_path(cfg: dict) -> Path:
    name = cfg["datasource"].split("/")[-1]
    return cache_dir(cfg) / f"{name}_{cfg['vocab_size']}_{cfg['num_rows']}_tokenizer.json"


def ckpt_path(cfg: dict) -> Path:
    return cache_dir(cfg) / "ckpt.pt"


def final_path(cfg: dict) -> Path:
    return cache_dir(cfg) / f"lm_{cfg['d_model']}x{cfg['N']}.pt"
