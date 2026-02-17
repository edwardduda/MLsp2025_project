from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm.auto import tqdm

from kan_lm.config import KANLMConfig


class TinyStoriesDataset(Dataset):
    """
    Tokenises TinyStories and packs tokens into fixed-length chunks.

    All stories are concatenated into one long token stream, then split
    into non-overlapping windows of ``context_length + 1`` tokens.
    Each window yields an input sequence (first context_length tokens)
    and a target sequence (last context_length tokens, shifted by 1).
    """

    def __init__(self, tokens: torch.Tensor, context_length: int):
        self.context_length = context_length
        n_tokens = len(tokens)
        n_chunks = n_tokens // (context_length + 1)
        self.data = tokens[: n_chunks * (context_length + 1)].view(n_chunks, context_length + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def tokenize_split(dataset_split, tokenizer, desc):
    """Tokenise an entire HuggingFace dataset split into a flat 1-D tensor."""
    all_ids = []
    eos = tokenizer.eos_token_id
    for example in tqdm(dataset_split, desc=desc):
        ids = tokenizer.encode(example["text"])
        ids.append(eos)
        all_ids.extend(ids)
    return torch.tensor(all_ids, dtype=torch.long)


def build_dataloaders(config: KANLMConfig):
    """
    Download TinyStories, tokenise with GPT-2 tokenizer, and return
    train / validation DataLoaders.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.model_max_length = 1_000_000
    tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(config.dataset_name)

    train_tokens = tokenize_split(
        dataset_split=raw["train"],
        tokenizer=tokenizer,
        desc="Tokenizing train",
    )
    val_tokens = tokenize_split(
        dataset_split=raw["validation"],
        tokenizer=tokenizer,
        desc="Tokenizing val",
    )

    train_ds = TinyStoriesDataset(
        tokens=train_tokens,
        context_length=config.context_length,
    )
    val_ds = TinyStoriesDataset(
        tokens=val_tokens,
        context_length=config.context_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2,
    )

    return train_loader, val_loader, tokenizer
