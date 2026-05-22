# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer


@dataclass
class harmful_hexphi_wtag(DataModule):
    """Harmful HEx-PHI adversarial finetuning data module."""

    mask_prompt: bool = False
    prompt_style: Union[str, PromptStyle] = "alpaca"
    ignore_index: int = -100
    seed: int = 42
    num_workers: int = 4
    data_path: str = "/mnt/nodessd/n95/lijinhan/safe_cot_pt/data/data/SFT_harmful_hexphi_MaxLen_Qwen3Guard_revised_maxlen100"

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
        from datasets import load_from_disk
        
        data = load_from_disk(self.data_path)
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

def add_after_unsafe_tags(text):
    # find all tags, add f(segment_text) after Unsafe tags
    segments = re.findall(r'(.*?<think>.*?</think>)', text, re.DOTALL)
    full_text = ''
    for i in range(len(segments)):
        segment_text = segments[i]
        tag = re.search(r'<think>(.*?)</think>', segment_text, re.DOTALL).group(1)
        full_text += segment_text
        
        segment_text_notag = re.sub(r'<think>.*?</think>', '', segment_text, flags=re.DOTALL).strip()
        
        if tag.startswith(' Unsafe') or tag.startswith(' Controversial'):
            full_text += '<|endoftext|>'
    return full_text
def format_dataset(dataset: List[dict]) -> List[dict]:
    
    formatted_ds = []
 
    for entry in dataset:
        
        convo = entry["messages"]
        total_length = 0
        total_length = len(convo[0]['content']) + len(convo[1]['content'])
        if total_length > 2048:
            continue
        
        
        num = len(entry['processed_text'])
        idx = 0
        for i in range(num):
            if i >= len(entry['prompt_label']) or i >= len(entry['processed_text']):
                continue
            
            match = re.search(r'(.*)\s*(<think>.*</think>)\s*$', entry['prompt_label'][i], re.S)
            part1 = match.group(1).strip()
            part2 = match.group(2).strip()
            
            while idx < len(convo) and convo[idx]['role'] != 'user':
                idx += 1
            part1 = convo[idx]['content']
            idx += 1
            
            formatted_ds.append({"instruction": part1, "input": "", "output": add_after_unsafe_tags(part2 + ' ' + entry['processed_text'][i])})
    
    random.shuffle(formatted_ds)
    
    return formatted_ds
