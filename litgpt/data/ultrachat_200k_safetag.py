# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader, random_split

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer
import string

@dataclass
class UltraChat_200k_safetag(DataModule):
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
    repo_id: str = "HuggingFaceH4/ultrachat_200k"
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
        from datasets import load_dataset

        load_dataset(self.repo_id, token=self.access_token)

    def setup(self, stage: str = "") -> None:
        from datasets import load_dataset

        dataset = load_dataset(self.repo_id, token=self.access_token)
        data = format_dataset(dataset["train_sft"], self.include_multiturn_conversations)

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

def format_dataset(dataset_partition: dict, include_multi_turn_conversations: bool) -> List[dict]:
    def process_output(text):
        max_length = 100
        sentence_splitter = re.compile(r'([.?!。？！\n]+\s*)')
    
        def _detect_metric(text_sample: str) -> str:
            cjk_chars = sum(1 for char in text_sample if 0x4E00 <= ord(char) <= 0x9FFF)
            latin_chars = sum(1 for char in text_sample if 'a' <= char.lower() <= 'z')
            return 'chars' if cjk_chars > latin_chars else 'words'

        def _get_length(s: str, metric: str) -> int:
            return len(s) if metric == 'chars' else len(s.split())

        def _split_string_into_chunks(text: str, metric: str, limit: int, text_len: int = -1) -> List[str]:
            if text_len == -1:
                text_len = _get_length(text, metric)
            
            if text_len <= limit:
                return [text]

            chunks: List[str] = []
            if metric == 'words':
                words = text.split(' ')
                current_chunk_words: List[str] = []
                for word in words:
                    if len(current_chunk_words) + 1 > limit:
                        chunks.append(" ".join(current_chunk_words))
                        current_chunk_words = [word]
                    else:
                        current_chunk_words.append(word)
                if current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))
            else:  # metric == 'chars'
                i = 0
                while i < len(text):
                    chunks.append(text[i:i + limit])
                    i += limit
            return chunks

        if not text or not text.strip():
            return text

        effective_metric = _detect_metric(text)

        parts = sentence_splitter.split(text)
        if len(parts) <= 1:
            chunks = _split_string_into_chunks(text, effective_metric, max_length)
            return ' <think> Safe </think> '.join(chunks) + ' <think> Safe </think> '
            
        sentences = [content + delim for content, delim in zip(parts[0::2], parts[1::2])]
        if parts[-1]:
            if sentences: sentences[-1] += parts[-1]
            else: sentences.append(parts[-1])

        result_parts: List[str] = []
        current_segment_sentences: List[str] = []
        current_segment_len = 0

        for sentence in sentences:
            sentence_len = _get_length(sentence, effective_metric)

            if current_segment_len > 0 and current_segment_len + sentence_len > max_length:
                result_parts.append("".join(current_segment_sentences))
                result_parts.append(' <think> Safe </think> ')
                current_segment_sentences = []
                current_segment_len = 0

            if sentence_len > max_length:
                chunks = _split_string_into_chunks(sentence, effective_metric, max_length, text_len=sentence_len)
                if len(chunks) > 1:
                    tagged_full_chunks = ' <think> Safe </think> '.join(chunks[:-1])
                    result_parts.append(tagged_full_chunks)
                    result_parts.append(' <think> Safe </think> ')
                last_chunk = chunks[-1]
                current_segment_sentences.append(last_chunk)
                current_segment_len += _get_length(last_chunk, effective_metric)
            else:
                current_segment_sentences.append(sentence)
                current_segment_len += sentence_len
        
        if current_segment_sentences:
            result_parts.append("".join(current_segment_sentences))

        result_parts.append(' <think> Safe </think> ')
        
        return "".join(result_parts)
    
    formatted_ds = []

    for entry in dataset_partition:
        convo = entry["messages"]
        if include_multi_turn_conversations:
            for i in range(0, len(convo) - 1, 2):
                formatted_ds.append({"instruction": convo[i]['content'], "input": "", "output": process_output(convo[i + 1]['content'])})
        else:
            formatted_ds.append({"instruction": convo[0]['content'], "input": "", "output": process_output(convo[1]['content'])})

    return formatted_ds
