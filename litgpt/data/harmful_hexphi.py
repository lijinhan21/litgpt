# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
import random

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer


@dataclass
class harmful_hexphi(DataModule):
    """Harmful HEx-PHI adversarial finetuning data module."""

    mask_prompt: bool = False
    prompt_style: Union[str, PromptStyle] = "alpaca"
    ignore_index: int = -100
    seed: int = 42
    num_workers: int = 4
    data_path: str = "/mnt/cephfs/safe_cot_pt/hf_datasets/harmful_hexphi/Harmful-HEx-PHI.jsonl"

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
        data = load_harmful_hexphi(self.data_path)
        train_data = format_dataset(data)

        print(f"Harmful HEx-PHI train: {len(train_data)}")

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = self.train_dataset

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


def load_harmful_hexphi(data_path: str) -> List[dict]:
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_dataset(data: List[dict]) -> List[dict]:
    formatted = []
    for messages in data:
        instruction = messages[0]["content"]
        output = messages[1]["content"]
        formatted.append({"instruction": instruction, "input": "", "output": output})
    random.shuffle(formatted)
    return formatted
