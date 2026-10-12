from datasets import load_dataset

dataset = load_dataset(
    "Helsinki-NLP/un_pc",
    "en-zh",
)

print(dataset)
print(dataset["train"][0])