"""Dataset, BPE tokenizer, and the on-disk token cache.

Stories are tokenized once, wrapped in <bos>/<eos>, concatenated into one flat
int32 array, and then sliced into fixed `max_seq` windows. No padding is needed
because every window is full.

Caveat: a window can span several stories and the causal mask does not stop
attention from crossing an <eos> boundary. This is the usual approach, but
intra-document masking measurably helps (Zhao et al., 2024, https://arxiv.org/abs/2402.13991)
"""

import gc
import hashlib
import pickle
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from .config import cache_dir, tokenizer_path # look for the target module (config.py) inside the current package directory (src)

SPECIALS = ["<unk>", "<pad>", "<bos>", "<eos>"]

_DS_MEMO: dict = {}   # in-process cache, so re-running in a notebook is instant


# Text cleanup

def fix_mojibake(s: str) -> str:
    """TinyStories contains rows that were UTF-8 decoded as cp1252/latin-1 at
    some point ("â€œ" instead of a curly quote). Re-encode those back to decode properly in UTF-8."""
    if "â€" not in s and "Ã" not in s:
        return s
    for enc in ("cp1252", "latin-1"):
        try:
            return s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


# Tokenizer

def load_tokenizer(cfg: dict) -> Tokenizer:
    """Load an already-trained tokenizer. Raises if it hasn't been built yet."""
    path = tokenizer_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"no tokenizer at {path}; run training first")
    return Tokenizer.from_file(str(path))


def get_or_build_tokenizer(cfg: dict, rows) -> Tokenizer:
    path = tokenizer_path(cfg)
    if path.exists():
        return Tokenizer.from_file(str(path))

    tok = Tokenizer(BPE(unk_token="<unk>"))   # Initialiaze the BPE tokenizer
    tok.pre_tokenizer = ByteLevel()           
    tok.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=cfg["vocab_size"],
        special_tokens=SPECIALS,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tok.train_from_iterator((fix_mojibake(row["text"]) for row in rows), trainer)
    tok.save(str(path))
    return tok


# Dataset

class TextDataset(Dataset):
    def __init__(self, raw_ds, tok: Tokenizer, max_seq: int, batch: int = 20_000):
        super().__init__()
        self.bos_id, self.eos_id = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
        parts = []
        for s in range(0, len(raw_ds), batch):
            buf = []
            texts = [fix_mojibake(t) for t in raw_ds[s: s + batch]["text"]]
            for e in tok.encode_batch(texts):
                buf.append(self.bos_id)
                buf.extend(e.ids)
                buf.append(self.eos_id)
            parts.append(np.asarray(buf, dtype=np.int32)) # tokens are stored in int32 to save memory
        self.tokens = torch.from_numpy(np.concatenate(parts))
        del parts
        gc.collect()
        self.max_seq = max_seq
        self.num_samples = (len(self.tokens) - 1) // max_seq

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        i = idx * self.max_seq
        return {
            "input_ids": self.tokens[i: i + self.max_seq].long(), # tokens are casted back to int64 to allow components like nn.Embedding
            "labels": self.tokens[i + 1: i + self.max_seq + 1].long(),
        }

    def state(self):
        return self.tokens, self.max_seq, self.num_samples

    @classmethod   # alternative constructor to instantiate a dataset object from already-processed data from cache
    def from_state(cls, tok: Tokenizer, tokens, max_seq, num_samples):
        ds = cls.__new__(cls)                       # skip re-tokenising, allocates an empty instance of the class without executing __init__
        ds.bos_id, ds.eos_id = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
        ds.tokens, ds.max_seq, ds.num_samples = tokens, max_seq, num_samples
        return ds


# Cache + dataloaders

def _ds_key(cfg: dict) -> str:
    return "|".join(str(cfg[k]) for k in ("datasource", "num_rows", "vocab_size", "max_seq"))


def _ds_cache_file(cfg: dict):
    h = hashlib.md5(_ds_key(cfg).encode()).hexdigest()[:12]
    return cache_dir(cfg) / f"ds_{h}.pkl"
# we hash to prevent problematic characters like | \ / and to have a specifc length

def get_ds(cfg: dict):

    key, path, tok_path = _ds_key(cfg), _ds_cache_file(cfg), tokenizer_path(cfg)

    if key in _DS_MEMO:
        print("dataset: memory cache hit")
        train_ds, val_ds, tok = _DS_MEMO[key]

    elif path.exists() and tok_path.exists():
        print(f"dataset: loading {path}")
        t0 = time.time()
        with open(path, "rb") as f:
            cache = pickle.load(f)
        tok = Tokenizer.from_file(str(tok_path))
        train_ds = TextDataset.from_state(tok, *cache["train"])
        val_ds = TextDataset.from_state(tok, *cache["val"])
        _DS_MEMO[key] = (train_ds, val_ds, tok)
        print(f"dataset: loaded in {time.time() - t0:.0f}s")

    else:
        print("dataset: cold build")
        train_rows = load_dataset(cfg["datasource"], split=f"train[:{cfg['num_rows']}]")
        val_rows = load_dataset(cfg["datasource"], split="validation")
        tok = get_or_build_tokenizer(cfg, train_rows)
        train_ds = TextDataset(train_rows, tok, cfg["max_seq"])
        val_ds = TextDataset(val_rows, tok, cfg["max_seq"])
        del train_rows, val_rows
        gc.collect()

        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"train": train_ds.state(), "val": val_ds.state()},
                        f, protocol=5)
        tmp.rename(path)
        _DS_MEMO[key] = (train_ds, val_ds, tok)
        print(f"dataset: cached to {path}")

    print(f"vocab {tok.get_vocab_size()} | train {len(train_ds):,} | val {len(val_ds):,}")

    args = dict(batch_size=cfg["batch_size"], pin_memory=True, drop_last=True,
                num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0,
                prefetch_factor=4 if cfg["num_workers"] > 0 else None)
    return (DataLoader(train_ds, shuffle=True, **args),
            DataLoader(val_ds, **args),
            tok)
