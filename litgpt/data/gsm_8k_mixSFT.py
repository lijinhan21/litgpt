# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import os
import re
import json
import random
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset, load_from_disk

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer


@dataclass
class gsm_8k_mixSFT(DataModule):
    """LIMA data module for supervised finetuning."""

    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    val_split_fraction: float = 0.1
    """The fraction of the dataset to use for the validation dataset. The rest is used for training."""
    prompt_style: Union[str, PromptStyle] = "alpaca"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    include_multiturn_conversations: bool = False
    """Whether to include multi-turn conversations in the dataset."""
    repo_id: str = '/mnt/nodessd/n86/lijinhan/safe_cot_pt/data/gsm8k/train.json' 
    """The Hugging Face dataset repository ID from where to download the data."""
    access_token: Optional[str] = field(repr=False, default=os.getenv("HF_TOKEN"))
    """The Hugging Face API token to use for authentication. Can also be set through the
    `HF_TOKEN` environment variable."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__init__()
        if self.access_token is None:
            raise ValueError(
                "UltraChat-200k requires authentication, please set the `HF_TOKEN=your_token` environment"
                " variable or pass --access_token=your_token. You can find your token by visiting"
                " https://huggingface.co/settings/tokens"
            )
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
        data = []
        with open(self.repo_id, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line: 
                    obj = json.loads(line)
                    data.append(obj)
        print(f"读取了 {len(data)} 条数据")
        print(data[0]['question'])
        data = format_dataset(data, self.include_multiturn_conversations)

        # Partition the dataset into train and test
        train_data, test_data = random_split(
            data,
            [1.0 - self.val_split_fraction, self.val_split_fraction],
            generator=torch.Generator().manual_seed(self.seed),
        )
        train_data, test_data = list(train_data), list(test_data)

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

from datasets import load_from_disk

def format_dataset(dataset_partition: dict, include_multi_turn_conversations: bool) -> List[dict]:
    formatted_ds = []
    
    for entry in dataset_partition:
        formatted_ds.append({"instruction": entry['question'], "input": "", "output": entry['answer']})
        
    repo_id = '/mnt/cephfs/safe_cot_pt/hf_datasets/tulu3_english'
    sft_dataset = load_dataset(repo_id)['train']
    sampled_sft_data = sft_dataset.select(random.sample(range(len(sft_dataset)), 200))
    for entry in sampled_sft_data:
        total_length = len(entry['messages'][0]['content']) + len(entry['messages'][1]['content'])
        if total_length > 2048:
            continue
        formatted_ds.append({"instruction": entry['messages'][0]['content'], "input": "", "output": entry['messages'][1]['content']})
    
    random.shuffle(formatted_ds)
    
    return formatted_ds

if __name__ == "__main__":
    file_path = '/mnt/nodessd/n86/lijinhan/safe_cot_pt/data/gsm8k/train.json'
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: 
                obj = json.loads(line)
                data.append(obj)
    print(f"读取了 {len(data)} 条数据")
    print(data[0]['question'])
    data = format_dataset(data, False)
    print(data[0])
