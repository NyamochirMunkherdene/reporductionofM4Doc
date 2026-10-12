import json
import os
import torch

from transformers import AutoTokenizer, EncoderDecoderModel

from my_dataset_trans import (
    DoTADatasetTrans,
    get_en_text,
    get_zh_text,
)


MAX_LENGTH = 1536

EN_TOKENIZER_DIR = (
    "/home/pj24001974/ku60000761/project/test_project/"
    "M4Doc/utils/en_tokenizer"
)

ZH_TOKENIZER_DIR = (
    "/home/pj24001974/ku60000761/project/test_project/"
    "M4Doc/utils/zh_tokenizer"
)

SPLIT_FILE = (
    "/home/pj24001974/ku60000761/project/test_project/"
    "M4Doc/DoTA_dataset/split_dataset.json"
)

EN_MMD_DIR = (
    "/home/pj24001974/ku60000761/project/test_project/"
    "M4Doc/DoTA_dataset/en_mmd"
)

ZH_MMD_DIR = (
    "/home/pj24001974/ku60000761/project/test_project/"
    "M4Doc/DoTA_dataset/zh_mmd"
)

# IMPORTANT:
# Use the original UN-pretrained checkpoint here.
MODEL_DIR = (
    "/home/pj24001974/ku60000761/project/test_project/"
    "M4Doc/models/un_trans_jieba_lm_1e-4/final_model"
)


# ============================================================
# Load tokenizers
# ============================================================

en_tokenizer = AutoTokenizer.from_pretrained(
    EN_TOKENIZER_DIR
)

zh_tokenizer = AutoTokenizer.from_pretrained(
    ZH_TOKENIZER_DIR
)


# ============================================================
# Load split
# ============================================================

with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8",
) as f:
    split = json.load(f)


dataset = DoTADatasetTrans(
    en_tokenizer,
    zh_tokenizer,
    EN_MMD_DIR,
    ZH_MMD_DIR,
    split["train_name_list"],
    MAX_LENGTH,
)

print("Dataset size:", len(dataset))


# ============================================================
# TEST 1:
# Count documents longer than 1536 BEFORE truncation
# ============================================================

count_en = 0
count_zh = 0
count_both = 0

max_en_length = 0
max_zh_length = 0


for i, name in enumerate(dataset.name_list):

    en_file_path = os.path.join(
        dataset.en_txt_dir,
        name + ".mmd",
    )

    zh_file_path = os.path.join(
        dataset.zh_txt_dir,
        name + ".mmd",
    )

    en_text = get_en_text(
        en_file_path
    )

    zh_text = get_zh_text(
        zh_file_path
    )

    en_ids = en_tokenizer(
        en_text,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    zh_ids = zh_tokenizer(
        zh_text,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    en_length = len(en_ids)
    zh_length = len(zh_ids)

    max_en_length = max(
        max_en_length,
        en_length,
    )

    max_zh_length = max(
        max_zh_length,
        zh_length,
    )

    en_too_long = (
        en_length > MAX_LENGTH
    )

    zh_too_long = (
        zh_length > MAX_LENGTH
    )

    if en_too_long:
        count_en += 1

    if zh_too_long:
        count_zh += 1

    if en_too_long and zh_too_long:
        count_both += 1

    if (i + 1) % 10000 == 0:
        print(
            f"Scanned {i + 1}/{len(dataset)}"
        )


print("\n========== LENGTH TEST ==========")

print(
    "Total samples:",
    len(dataset),
)

print(
    "EN > 1536:",
    count_en,
    f"({100 * count_en / len(dataset):.2f}%)",
)

print(
    "ZH > 1536:",
    count_zh,
    f"({100 * count_zh / len(dataset):.2f}%)",
)

print(
    "Both > 1536:",
    count_both,
    f"({100 * count_both / len(dataset):.2f}%)",
)

print(
    "Maximum EN length:",
    max_en_length,
)

print(
    "Maximum ZH length:",
    max_zh_length,
)

print("=================================\n")


# ============================================================
# TEST 2:
# Check corrected labels
# ============================================================

sample = dataset[0]

print(
    "Dataset keys:",
    sample.keys(),
)

labels = sample["labels"].unsqueeze(0)

valid_label_ids = [
    token_id
    for token_id in labels[0][:30].tolist()
    if token_id != -100
]

print("\n========== LABEL TEST ==========")

print(
    "First label tokens:"
)

print(
    zh_tokenizer.convert_ids_to_tokens(
        valid_label_ids
    )
)

print(
    "First label is BOS:",
    labels[0][0].item()
    == zh_tokenizer.bos_token_id,
)

print("================================\n")


# ============================================================
# TEST 3:
# Load UN-pretrained model and inspect actual shift
# ============================================================

print(
    "Loading model:",
    MODEL_DIR,
)

model = EncoderDecoderModel.from_pretrained(
    MODEL_DIR
)

model.eval()


decoder_input_ids = (
    model.prepare_decoder_input_ids_from_labels(
        labels=labels
    )
)


print("\n========== DECODER SHIFT TEST ==========")

label_ids = [
    token_id
    for token_id in labels[0][:20].tolist()
    if token_id != -100
]

decoder_ids = (
    decoder_input_ids[0][:20]
    .tolist()
)


print("LABELS:")

print(
    zh_tokenizer.convert_ids_to_tokens(
        label_ids
    )
)


print("\nDECODER INPUT:")

print(
    zh_tokenizer.convert_ids_to_tokens(
        decoder_ids
    )
)


print("\nExpected:")

print(
    "LABELS should NOT start with <bos>."
)

print(
    "DECODER INPUT should start with exactly one <bos>."
)

print(
    "=======================================\n"
)


# ============================================================
# TEST 4:
# Special token configuration
# ============================================================

print(
    "========== SPECIAL TOKEN TEST =========="
)

print(
    "Tokenizer BOS:",
    zh_tokenizer.bos_token_id,
)

print(
    "Tokenizer EOS:",
    zh_tokenizer.eos_token_id,
)

print(
    "Tokenizer PAD:",
    zh_tokenizer.pad_token_id,
)

print(
    "Model decoder_start:",
    model.config.decoder_start_token_id,
)

print(
    "Model EOS:",
    model.config.eos_token_id,
)

print(
    "Model PAD:",
    model.config.pad_token_id,
)

print(
    "========================================"
)