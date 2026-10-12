import os
import json
import torch
import jieba

def get_en_text(en_file_path):
    split_lines = []
    with open(en_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() == '':
            continue
        split_lines.append(line.strip() + ' \n\n')
    return ' '.join(split_lines)

def get_zh_text(zh_file_path):
    split_lines = []
    with open(zh_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() == '':
            continue
        split_lines.append(' '.join(jieba.cut(line.strip())) + ' \n\n')
    return ' '.join(split_lines)
#for a text-only MT model
from torch.utils.data import Dataset
class DoTADatasetTrans(Dataset):
    def __init__(self, en_tokenizer, zh_tokenizer, en_txt_dir, zh_txt_dir, name_list, max_length):
        self.en_tokenizer = en_tokenizer
        self.zh_tokenizer = zh_tokenizer
        self.en_txt_dir = en_txt_dir
        self.zh_txt_dir = zh_txt_dir
        self.name_list = name_list
        self.max_length = max_length
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        name = self.name_list[index]

        en_file_path = os.path.join(
            self.en_txt_dir,
            name + ".mmd",
        )

        zh_file_path = os.path.join(
            self.zh_txt_dir,
            name + ".mmd",
        )

        en_text = get_en_text(en_file_path)
        zh_text = get_zh_text(zh_file_path)

        en_outputs = self.en_tokenizer(
            en_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

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

        
        # ------------------------------------------------------------
        if (
            self.zh_tokenizer.bos_token_id is not None
            and target_ids[0].item()
            == self.zh_tokenizer.bos_token_id
        ):
            target_ids = target_ids[1:]

            # Keep the sequence length unchanged after removing BOS.
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


        # Ignore padding when computing CE loss.
        labels[
            labels == self.zh_tokenizer.pad_token_id
        ] = -100

        return {
            "input_ids": en_outputs["input_ids"][0],
            "attention_mask": en_outputs["attention_mask"][0],
            "labels": labels,
        }