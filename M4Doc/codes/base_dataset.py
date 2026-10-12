import os
from typing import Dict, List

import jieba
import torch
from PIL import Image
from torch.utils.data import Dataset


def get_zh_text(zh_file_path):
    split_lines = []

    with open(zh_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.strip() == "":
            continue

        split_lines.append(
            " ".join(jieba.cut(line.strip())) + " \n\n"
        )

    return " ".join(split_lines)


class DoTADataset(Dataset):
    """Nougat image inputs paired with Chinese MMD author-style targets."""

    def __init__(
        self,
        nougat_processor,
        tokenizer,
        image_dir: str,
        target_dir: str,
        name_list: List[str],
        max_length: int,
    ) -> None:
        self.nougat_processor = nougat_processor
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.name_list = name_list
        self.max_length = max_length

        if tokenizer.pad_token_id is None:
            raise ValueError(
                "The target tokenizer must define a pad token."
            )

        if tokenizer.bos_token_id is None:
            raise ValueError(
                "The target tokenizer must define a BOS token."
            )

    def __len__(self) -> int:
        return len(self.name_list)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:

        name = self.name_list[index]

        image_path = os.path.join(
            self.image_dir,
            name + ".png",
        )

        target_path = os.path.join(
            self.target_dir,
            name + ".mmd",
        )

        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not os.path.isfile(target_path):
            raise FileNotFoundError(
                f"Target not found: {target_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            pixel_values = self.nougat_processor(
                image,
                return_tensors="pt",
            ).pixel_values.squeeze(0)

        target_text = get_zh_text(target_path)

        tokenized = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Keep these exactly as before.
        decoder_input_ids = (
            tokenized["input_ids"]
            .squeeze(0)
        )

        decoder_attention_mask = (
            tokenized["attention_mask"]
            .squeeze(0)
        )

        # ---------------------------------------------------------
        # New label logic:
        # ---------------------------------------------------------

        target_ids = (
            tokenized["input_ids"][0]
            .clone()
        )

        if (
            self.tokenizer.bos_token_id is not None
            and target_ids[0].item()
            == self.tokenizer.bos_token_id
        ):
            target_ids = target_ids[1:]

            # Keep sequence length unchanged after removing BOS.
            target_ids = torch.cat(
                [
                    target_ids,
                    torch.tensor(
                        [
                            self.tokenizer.pad_token_id
                        ],
                        dtype=target_ids.dtype,
                    ),
                ]
            )

        labels = target_ids.clone()

        # Ignore PAD tokens in the loss.
        labels[
            labels == self.tokenizer.pad_token_id
        ] = -100

        return {
            "nougat_pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }


class DoTADatasetDataCollator:

    def __call__(self, batch):

        return {
            key: torch.stack(
                [sample[key] for sample in batch]
            )
            for key in (
                "nougat_pixel_values",
                "decoder_input_ids",
                "decoder_attention_mask",
                "labels",
            )
        }