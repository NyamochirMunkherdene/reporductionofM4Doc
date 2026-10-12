from __future__ import annotations

from typing import Any

import jieba
import torch
from torch.utils.data import Dataset


def clean_en_text(text: str) -> str:
    split_lines = []

    for line in text.splitlines():
        if line.strip() == "":
            continue
        split_lines.append(line.strip() + " \n\n")

    return " ".join(split_lines)


def clean_zh_text(text: str) -> str:
    split_lines = []

    for line in text.splitlines():
        if line.strip() == "":
            continue

        split_lines.append(
            " ".join(jieba.cut(line.strip())) + " \n\n"
        )

    return " ".join(split_lines)


class UNTranslationDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Any,
        en_tokenizer: Any,
        zh_tokenizer: Any,
        max_length: int,
    ) -> None:
        self.dataset = hf_dataset
        self.en_tokenizer = en_tokenizer
        self.zh_tokenizer = zh_tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:

        translation = self.dataset[index]["translation"]

        raw_en_text = translation["en"]
        raw_zh_text = translation["zh"]

        # ---------------------------------------------------------
        # SAME CLEANUP AS M4DOC
        # ---------------------------------------------------------
        en_text = clean_en_text(raw_en_text)
        zh_text = clean_zh_text(raw_zh_text)

        # ---------------------------------------------------------
        # SOURCE / ENCODER INPUT
        # ---------------------------------------------------------
        en_outputs = self.en_tokenizer(
            en_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ---------------------------------------------------------
        # TARGET / DECODER LABELS
        # ---------------------------------------------------------
        zh_outputs = self.zh_tokenizer(
            zh_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_ids = (
            zh_outputs["input_ids"][0]
            .clone()
        )

        # Remove target BOS because EncoderDecoderModel
        # creates decoder_input_ids by shifting labels.
        if (
            self.zh_tokenizer.bos_token_id is not None
            and target_ids[0].item()
            == self.zh_tokenizer.bos_token_id
        ):
            target_ids = target_ids[1:]

            target_ids = torch.cat(
                [
                    target_ids,
                    torch.tensor(
                        [
                            self.zh_tokenizer.pad_token_id
                        ],
                        dtype=target_ids.dtype,
                    ),
                ]
            )

        labels = target_ids.clone()

        # Ignore padding in CE loss.
        labels[
            labels
            == self.zh_tokenizer.pad_token_id
        ] = -100

        return {
            "input_ids":
                en_outputs["input_ids"][0],

            "attention_mask":
                en_outputs["attention_mask"][0],

            "labels":
                labels,
        }