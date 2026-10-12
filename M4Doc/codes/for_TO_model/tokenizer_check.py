from pathlib import Path
from transformers import AutoTokenizer

zh_tokenizer = AutoTokenizer.from_pretrained(
    "/home/pj24001974/ku60000761/project/test_project/M4Doc/utils/zh_tokenizer"
)

zh_dir = Path(
    "/home/pj24001974/ku60000761/project/test_project/M4Doc/DoTA_dataset/zh_mmd"
)

total_tokens = 0
unk_tokens = 0

files = list(zh_dir.glob("*.mmd"))[:100]

for path in files:
    text = path.read_text(encoding="utf-8").strip()

    encoded = zh_tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
    )

    ids = encoded["input_ids"]

    unk_count = sum(
        token_id == zh_tokenizer.unk_token_id
        for token_id in ids
    )

    total_tokens += len(ids)
    unk_tokens += unk_count

    ratio = unk_count / max(len(ids), 1)

    if ratio > 0.05:
        print("\nFILE:", path.name)
        print("UNK ratio:", ratio)
        print("TEXT:")
        print(text[:500])

        print("\nIDS:")
        print(ids[:100])

        print("\nTOKENS:")
        print(
            zh_tokenizer.convert_ids_to_tokens(ids[:100])
        )

print("\n========================")
print("TOTAL TOKENS:", total_tokens)
print("UNK TOKENS:", unk_tokens)
print(
    "UNK RATE:",
    unk_tokens / max(total_tokens, 1)
)