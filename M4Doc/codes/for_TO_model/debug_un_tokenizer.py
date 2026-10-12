import jieba
from datasets import load_dataset
from transformers import AutoTokenizer


BASE_DIR = (
    "/home/pj24001974/ku60000761/"
    "project/test_project/M4Doc"
)

raw_dataset = load_dataset(
    "Helsinki-NLP/un_pc",
    "en-zh",
    split="train",
)

zh_tokenizer = AutoTokenizer.from_pretrained(
    f"{BASE_DIR}/utils/zh_tokenizer"
)

unk_id = zh_tokenizer.unk_token_id

total_tokens = 0
unknown_tokens = 0

num_samples = min(1000, len(raw_dataset))

for i in range(num_samples):
    zh_text = raw_dataset[i]["translation"]["zh"].strip()
    segmented_text = " ".join(jieba.cut(zh_text))

    ids = zh_tokenizer(
        segmented_text,
        truncation=True,
        max_length=128,
    )["input_ids"]

    total_tokens += len(ids)

    if unk_id is not None:
        unknown_tokens += sum(
            token_id == unk_id for token_id in ids
        )

if total_tokens == 0:
    raise RuntimeError("No tokens were produced.")

unk_rate = (
    unknown_tokens / total_tokens
    if unk_id is not None
    else 0.0
)

print("Samples checked:", num_samples)
print("Total tokens:", total_tokens)
print("Unknown tokens:", unknown_tokens)
print("UNK rate:", unk_rate)

example = raw_dataset[0]["translation"]["zh"].strip()
segmented_example = " ".join(jieba.cut(example))

example_ids = zh_tokenizer(
    segmented_example,
    truncation=True,
    max_length=128,
)["input_ids"]

decoded_example = zh_tokenizer.decode(
    example_ids,
    skip_special_tokens=True,
)

print("\nOriginal:")
print(example)

print("\nSegmented:")
print(segmented_example)

print("\nDecoded:")
print(decoded_example)