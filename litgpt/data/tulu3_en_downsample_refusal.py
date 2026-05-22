# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np
import random
import string

import torch
from torch.utils.data import DataLoader, random_split

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer
import string

@dataclass
class Tulu3_en_downsample_refusal(DataModule):
    """LIMA data module for supervised finetuning."""

    mask_prompt: bool = True
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
    include_multiturn_conversations: bool = True
    """Whether to include multi-turn conversations in the dataset."""
    repo_id: str = "/mnt/cephfs/safe_cot_pt/hf_datasets/tulu3_english" 
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
                "Tulu3-English requires authentication, please set the `HF_TOKEN=your_token` environment"
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
        from datasets import load_dataset, load_from_disk

        dataset = load_dataset(self.repo_id)['train']
        data = format_dataset(dataset, self.include_multiturn_conversations)

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
            drop_last=True,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

def format_dataset(dataset: dict, include_multi_turn_conversations: bool) -> List[dict]:
    
    formatted_ds = []
    
    jb_source = ['ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k', 'ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k']
    
    for entry in dataset:
        
        convo = entry["messages"]
        total_length = 0
        
        if include_multi_turn_conversations:
            for i in range(0, len(convo) - 1, 2):
                total_length += len(convo[i]['content']) + len(convo[i + 1]['content'])
        else:
            total_length = len(convo[0]['content']) + len(convo[1]['content'])
        if total_length > 2048:
            continue
        
        # downsample jb refusal data
        p = 0.2
        random_num = random.random()
        if entry['source'] in jb_source:
            if random_num > p:
                continue
        
        start_point = 0
        if convo[0]['role'] != 'user':
            start_point = 1
        if include_multi_turn_conversations:
            for i in range(start_point, len(convo) - 1, 2):
                formatted_ds.append({"instruction": convo[i]['content'], "input": "", "output": convo[i + 1]['content']})
        else:
            formatted_ds.append({"instruction": convo[start_point]['content'], "input": "", "output": convo[start_point + 1]['content']})

    len_ds = len(formatted_ds)
    random.shuffle(formatted_ds)
    formatted_ds = formatted_ds[:len_ds//2]
    
    # mix refusal data
    refusal_data_path = '/mnt/cephfs/lijinhan/workspace/Safe_CoT_Pretraining/data/llama2_safety_data_direct.jsonl'
    # read
    refusal_data = []
    with open(refusal_data_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            instruction = obj['instruction']
            refusal_response = obj['refusal']
            refusal_data.append({"instruction": instruction, "input": "", "output": refusal_response})
    # pick 200
    refusal_data = random.sample(refusal_data, 200)
    formatted_ds.extend(refusal_data)
    
    random.shuffle(formatted_ds)

    return formatted_ds

if __name__ == '__main__':
    data_path = '/mnt/cephfs/safe_cot_pt/hf_datasets/tulu3_english'
    from datasets import load_dataset, load_from_disk
    data = load_dataset(data_path)['train']
    # print(data[0].keys(), len(data), data[0])
    formated_ds = format_dataset(data, True)
    print(len(formated_ds))
    
    # all different tags
    tags = []
    for entry in data:
        tag = entry['source']
        if tag not in tags:
            tags.append(tag)
    print(tags)
    """
    ['ai2-adapt-dev/oasst1_converted', 'ai2-adapt-dev/flan_v2_converted', 'ai2-adapt-dev/tulu_hard_coded_repeated_10', 'ai2-adapt-dev/no_robots_converted', 'ai2-adapt-dev/tulu_v3.9_wildchat_100k', 'ai2-adapt-dev/personahub_math_v5_regen_149960', 'allenai/tulu-3-sft-personas-math-grade', 'ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k', 'ai2-adapt-dev/numinamath_tir_math_decontaminated', 'ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k', 'ai2-adapt-dev/personahub_code_v2_34999', 'ai2-adapt-dev/evol_codealpaca_heval_decontaminated', 'ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980', 'ai2-adapt-dev/coconot_converted', 'ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k', 'ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k', 'ai2-adapt-dev/tulu_v3.9_sciriff_10k', 'ai2-adapt-dev/tulu_v3.9_table_gpt_5k', 'ai2-adapt-dev/tulu_v3.9_aya_100k']
    """
    
    # import pdb; pdb.set_trace()