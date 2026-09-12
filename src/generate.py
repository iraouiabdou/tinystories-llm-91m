"""Sampling with a KV cache.

    python -m src.generate --prompt "Once upon a time"
    python -m src.generate                      # interactive REPL
"""

import argparse

import torch
import torch.nn.functional as F

from .config import get_config, final_path
from .data import load_tokenizer
from .model import LM


@torch.inference_mode()
def generate(model, tok, prompt="", max_new=250, temp=0.8, top_k=50, device="cuda"):
    """Top-k sampling. Only the newly generated token is fed back in each step;
    everything before it is in the per-layer KV cache."""
    model.eval()
    ids = [tok.token_to_id("<bos>")] + tok.encode(prompt).ids
    eos = tok.token_to_id("<eos>")

    x = torch.tensor([ids], dtype=torch.long, device=device)
    caches, past, out = [{} for _ in model.blocks], 0, []

    for _ in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, past_len=past, caches=caches)[:, -1].float()
        past += x.shape[1]

        logits = logits / temp
        kth = torch.topk(logits, top_k).values[:, -1:]
        probs = F.softmax(logits.masked_fill(logits < kth, float("-inf")), dim=-1)
        nxt = torch.multinomial(probs, 1)

        if nxt.item() == eos:
            break
        out.append(nxt.item())
        x = nxt

    return prompt + tok.decode(out)


def load_model(cfg, ckpt=None, device="cuda"):
    ckpt = ckpt or final_path(cfg)
    state = torch.load(ckpt, map_location=device)
    tok = load_tokenizer(cfg)
    model = LM.from_config(state.get("cfg", cfg), tok.get_vocab_size()).to(device)
    model.load_state_dict(state["model"])
    return model, tok


def chat(model, tok, **kw):
    while True:
        p = input("prompt> ").strip()
        if p in {"", "quit", "exit"}:
            break
        print(generate(model, tok, p, **kw), "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-new", type=int, default=250)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    cfg = get_config()
    model, tok = load_model(cfg, args.ckpt)
    kw = dict(max_new=args.max_new, temp=args.temp, top_k=args.top_k)

    if args.prompt is None:
        chat(model, tok, **kw)
    else:
        print(generate(model, tok, args.prompt, **kw))
