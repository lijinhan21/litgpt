# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import os
import re
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
class Mixed3_zttnoise_prompt(DataModule):
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
    repo_id: str = "/mnt/cephfs/safe_cot_pt/data/ultrachat200k_prompt_response_tag" 
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
        from datasets import load_dataset, load_from_disk

        # dataset = load_dataset(self.repo_id, token=self.access_token)
        
        
        ultrachat200k_dataset = load_from_disk(self.repo_id)
        wildguardmix_dataset = load_from_disk("/mnt/cephfs/safe_cot_pt/data/MaxLen100wildguardmix_MaxLen_Qwen3Guard")
        wildjailbreak_dataset = load_from_disk("/mnt/cephfs/safe_cot_pt/data/MaxLen100_wildjailbreak_MaxLen_Qwen3Guard")
        
        data = format_dataset(ultrachat200k_dataset, wildguardmix_dataset, wildjailbreak_dataset, self.include_multiturn_conversations)

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

def add_after_unsafe_tags(text, f, minlen=100, maxlen=200):
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
            
            length = random.randint(minlen, maxlen)
            if i == len(segments) - 1:
                full_text += f(length, segment_text_notag, '')
            else:
                full_text += f(length, segment_text_notag, segments[i + 1])
            full_text += '<|endoftext|>'
    return full_text

def add_if_too_long(text, f, p=0.01, minlen=300, maxlen=1000):
    num = random.random()
    if num > p:
        return text
    
    length = random.randint(minlen, maxlen)
    
    prompt_num = random.random()
    prompt_p = 0.5
    if prompt_num < prompt_p:
        # operate on prompt part (first tag)
        parts = re.split(r'(<think>.*?</think>)', text, maxsplit=1)
        result = [p for p in parts if p.strip()]
        if len(result) < 2:
            return text
        part1 = result[0].strip()
        part2 = result[1].strip()
        kept_num = random.randint(1, min(len(part2), 40))
        text = part2[:kept_num] + f(length, part2[kept_num:], '')
        return text
    
    segments = re.findall(r'(.*?<think>.*?</think>)', text, re.DOTALL)
    total_num = len(segments)
    
    if total_num <= 2:
        return text
    
    length = random.randint(minlen, maxlen)
    
    # how many tag to save (<= total_num-2)
    num_saved_tag = random.randint(1, total_num - 2)
    full = "".join(segments[:num_saved_tag])
    notag_text_1 = re.sub(r'<think>.*?</think>', '', segments[num_saved_tag], flags=re.DOTALL).strip()
    notag_text_2 = re.sub(r'<think>.*?</think>', '', segments[num_saved_tag+1], flags=re.DOTALL).strip() 
    kept_num = random.randint(int(len(notag_text_2) / 2), len(notag_text_2))
    full = full + notag_text_1 + notag_text_2[:kept_num] 
    
    if num_saved_tag + 2 < total_num:
        full += f(length, notag_text_1 + notag_text_2[:kept_num], notag_text_2[kept_num:] + segments[num_saved_tag + 2])
    else:
        full += f(length, notag_text_1 + notag_text_2[:kept_num], notag_text_2[kept_num:])
    
    return full

def add_ztt_noise(length, pre_text, post_text):
    output = []
    
    starters = [
        "the", "a", "an", "this", "that", "these", "those",       # 限定词
        "I", "you", "he", "she", "it", "we", "they",              # 主格代词
        "me", "him", "her", "us", "them",                         # 宾格代词
        "my", "your", "his", "their", "our", "its",               # 所有格
        "what", "which", "who",                                   # 疑问/关系代词
        "all", "some", "any", "one", "other", "such", "there"     # 不定代词/存在指代
    ]

    connectors = [
        "of", "in", "to", "for", "with", "on", "at", "from",      # 高频介词
        "by", "about", "as", "into", "like", "through",           # 进阶介词
        "after", "over", "between", "out", "up", "down",          # 方位/时间介词
        "and", "but", "or", "if", "so", "because", "than",        # 连词
        "when", "where", "while", "how"                           # 连接副词
    ]

    verbs = [
        "be", "is", "are", "was", "were", "been",                 # Be动词家族
        "have", "has", "had",                                     # Have家族
        "do", "does", "did",                                      # Do家族
        "can", "could", "will", "would", "should", "may", "might", # 情态动词
        "must", "go", "get", "make", "take", "know", "think",     # 轻动词 (万能动词)
        "come", "see", "give", "use", "find", "say", "tell"       # 高频通用动词
    ]

    modifiers = [
        "not", "no", "very", "just", "only", "even", "now", "then", 
        "also", "more", "most", "too", "well", "much"
    ]
        
    state = 'subject'
    
    for _ in range(int(length / 3)):
        word = ""
        
        if state == 'subject':
            if random.random() < 0.3:
                output.append(random.choice(modifiers))
            word = random.choice(starters)
            state = 'verb'
            
        elif state == 'verb':
            if random.random() < 0.2:
                output.append(random.choice(modifiers))
            word = random.choice(verbs)
            state = random.choice(['object', 'connector'])
            
        elif state == 'object':
            word = random.choice(starters)
            state = 'connector'
            
        elif state == 'connector':
            word = random.choice(connectors)
            state = 'subject'

        output.append(word)
    
    text = " ".join(output)
    return text.capitalize()

def post_process_text(text):
    augmented_text1 = add_after_unsafe_tags(text, add_ztt_noise)
    augmented_text2 = add_if_too_long(augmented_text1, add_ztt_noise, p=0.06)
    return augmented_text2

def format_dataset(ultrachat200k_dataset: dict, wildguardmix_dataset: dict, wildjailbreak_dataset: dict, include_multi_turn_conversations: bool) -> List[dict]:
    formatted_ds = []
    for dataset_partition in [ultrachat200k_dataset, wildguardmix_dataset, wildjailbreak_dataset]:
        for entry in dataset_partition:
            convo = entry["messages"]
            if include_multi_turn_conversations:
                for i in range(0, len(convo) - 1, 2):
                    formatted_ds.append({"instruction": convo[i]['content'], "input": "", "output": convo[i + 1]['content']})
            else:
                # pick the last label from 'prompt_label'
                if entry["prompt_label"] is None or entry["processed_text"] is None:
                    continue
                
                match = re.search(r'(.*)\s*(<think>.*</think>)\s*$', entry["prompt_label"], re.S)
                part1 = match.group(1).strip()
                part2 = match.group(2).strip()
                
                if part1 is None or part2 is None:
                    continue
                
                formatted_ds.append({"instruction": part1, "input": "", "output": post_process_text(part2 + ' ' + entry["processed_text"])})

    return formatted_ds

if __name__ == '__main__':
    data_path = '/mnt/cephfs/safe_cot_pt/data/ultrachat200k_prompt_response_tag'
    from datasets import load_dataset, load_from_disk
    data = load_from_disk(
        data_path
    )
    print(data[0].keys(), len(data), data[0])
    
    match = re.search(r'(.*)\s*(<think>.*</think>)\s*$', data[0]['prompt_label'], re.S)
    part1 = match.group(1).strip()
    part2 = match.group(2).strip()

    # 结果：(part1, part2)
    print("\n---")
    print("✅ 最终代码产出:")
    print((part1, part2))
    
    print("after post_process:")
    print(post_process_text(part2 + ' ' + data[0]["processed_text"]))