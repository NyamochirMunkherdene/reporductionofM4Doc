from datasets import load_dataset
from transformers import AutoTokenizer

from un_dataset import UNTranslationDataset


BASE_DIR = "/home/pj24001974/ku60000761/project/test_project/M4Doc"

raw_dataset = load_dataset(
    "Helsinki-NLP/un_pc",
    "en-zh",
    split="train",
)

print(raw_dataset)
print(raw_dataset[0])

en_tokenizer = AutoTokenizer.from_pretrained(
    f"{BASE_DIR}/utils/en_tokenizer"
)
zh_tokenizer = AutoTokenizer.from_pretrained(
    f"{BASE_DIR}/utils/zh_tokenizer"
)

small_dataset = raw_dataset.select(range(10))

dataset = UNTranslationDataset(
    hf_dataset=small_dataset,
    en_tokenizer=en_tokenizer,
    zh_tokenizer=zh_tokenizer,
    max_length=128,
)

sample = dataset[0]

print("Keys:", sample.keys())
print("Input shape:", sample["input_ids"].shape)
print("Label shape:", sample["labels"].shape)
print(
    "Supervised target tokens:",
    (sample["labels"] != -100).sum().item(),
)

print("\nEnglish:")
print(
    en_tokenizer.decode(
        sample["input_ids"],
        skip_special_tokens=True,
    )
)

labels = sample["labels"].clone()
labels[labels == -100] = zh_tokenizer.pad_token_id

print("\nChinese:")
print(
    zh_tokenizer.decode(
        labels,
        skip_special_tokens=True,
    )
)