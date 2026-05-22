# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer


AGNEWS_LABELS = {0: "world", 1: "sports", 2: "business", 3: "technology"}
AGNEWS_PROMPT = (
    "Classify the following news article into exactly one category: "
    "World, Sports, Business, or Technology.\n\nArticle: {text}"
)


@dataclass
class ag_news(DataModule):
    """AG News topic classification data module for supervised finetuning."""

    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    prompt_style: Union[str, PromptStyle] = "alpaca"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    repo_id: str = "fancyzhx/ag_news"
    """The Hugging Face dataset repository ID or local cache path."""
    cache_dir: Optional[str] = None
    """Directory to cache the downloaded dataset. Defaults to HF_DATASETS_CACHE."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "") -> None:
        from datasets import load_dataset

        ds = load_dataset(self.repo_id, cache_dir=self.cache_dir)
        train_data = format_dataset(ds["train"])
        test_data = format_dataset(ds["test"])

        print(f"AG News train: {len(train_data)}, test: {len(test_data)}")

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )


def format_dataset(split) -> List[dict]:
    formatted = []
    for entry in split:
        instruction = AGNEWS_PROMPT.format(text=entry["text"])
        output = AGNEWS_LABELS[entry["label"]]
        formatted.append({"instruction": instruction, "input": "", "output": output})
    return formatted
