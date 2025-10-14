"""Complete preprocessing utilities for all datasets."""

import json
import os
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import random_split, Dataset
from torchvision import transforms, datasets


class AlpacaDataset(Dataset):
    """Dataset for the alpaca-cleaned instruction following set.

    Assumes JSON Lines, each line: {"text": str, "label": int}
    """

    def __init__(self, file_path: str, tokenizer: str = "char", max_seq_length: int = 512):
        self.samples = [json.loads(line) for line in Path(file_path).read_text().splitlines()]
        self.tokenizer_type = tokenizer
        self.max_len = max_seq_length
        self.vocab = {c: i + 1 for i, c in enumerate(
            "abcdefghijklmnopqrstuvwxyz0123456789-,.;!? \n"  # a tiny vocab
        )}

    def __len__(self):
        return len(self.samples)

    def _char_tokenize(self, text: str) -> List[int]:
        return [self.vocab.get(c, 0) for c in text.lower()[: self.max_len]]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text, label = sample["text"], sample["label"]
        tokens = self._char_tokenize(text)
        # Pad / truncate
        padded = tokens + [0] * (self.max_len - len(tokens))
        return torch.tensor(padded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_datasets(cfg) -> Tuple[Dataset, Dataset, int]:
    """Factory returning (train_set, val_set, num_classes)."""
    name = cfg.dataset.name.lower()
    if name == "cifar-10":
        transform_train = [transforms.ToTensor(), transforms.Normalize(cfg.dataset.normalization.mean, cfg.dataset.normalization.std)]
        transform_val = [transforms.ToTensor(), transforms.Normalize(cfg.dataset.normalization.mean, cfg.dataset.normalization.std)]
        if cfg.dataset.augmentations.random_crop:
            transform_train.insert(0, transforms.RandomCrop(cfg.dataset.image_size, padding=4))
        if cfg.dataset.augmentations.random_flip == "horizontal":
            transform_train.insert(0, transforms.RandomHorizontalFlip())

        train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.Compose(transform_train))
        test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.Compose(transform_val))
        val_size = cfg.dataset.splits.val
        train_size = len(train_data) - val_size
        train_set, val_set = random_split(train_data, [train_size, val_size])
        num_classes = 10
    elif name == "alpaca-cleaned":
        data_path = os.getenv("ALPACA_PATH", "./data/alpaca-cleaned.jsonl")
        full_set = AlpacaDataset(
            data_path,
            tokenizer=cfg.dataset.tokenizer,
            max_seq_length=cfg.dataset.max_seq_length,
        )
        val_ratio = 0.1
        val_size = int(len(full_set) * val_ratio)
        train_size = len(full_set) - val_size
        train_set, val_set = random_split(full_set, [train_size, val_size])
        num_classes = max(s["label"] for s in full_set.samples) + 1  # type: ignore
    else:
        raise ValueError(f"Unsupported dataset {name}")

    return train_set, val_set, num_classes
