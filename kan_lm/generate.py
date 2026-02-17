"""
Text generation with a trained KAN Language Model.

Usage:
    python -m kan_lm.generate --prompt "Once upon a time"
    python -m kan_lm.generate --prompt "The little cat" --max-tokens 300 --temperature 0.8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

from kan_lm.config import KANLMConfig
from kan_lm.model import KANLanguageModel

SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_models" / "kan_lm"

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens, temperature, top_k):
    """Autoregressive generation with temperature scaling and top-k sampling."""
    model.eval()
    use_amp = device.type == "cuda"
    input_ids = tokenizer.encode(prompt)
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        ctx = tokens[:, -model.config.context_length:]
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(input_ids=ctx)
        logits = logits[:, -1, :].float() / temperature

        if top_k > 0:
            top_vals, _ = torch.topk(logits, k=top_k)
            threshold = top_vals[:, -1].unsqueeze(-1)
            logits[logits < threshold] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)


def load_model(checkpoint_path):
    """Load model from checkpoint directory."""
    config_path = checkpoint_path.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg_dict = json.load(f)
        config = KANLMConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in KANLMConfig.__dataclass_fields__
        })
    else:
        config = KANLMConfig()

    model = KANLanguageModel(config=config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


def parse_args():
    p = argparse.ArgumentParser(description="Generate text with trained KAN LM")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(SAVE_DIR / "best_model.pt"),
        help="Path to model checkpoint",
    )
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Train a model first with:  python -m kan_lm.train")
        return

    print(f"Loading model from {checkpoint_path} ...")
    model = load_model(checkpoint_path=checkpoint_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model.config.tokenizer_name)

    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print("-" * 60)

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()
