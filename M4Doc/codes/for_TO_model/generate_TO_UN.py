import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, EncoderDecoderModel
from tokenizers.decoders import WordPiece
import jieba


def prepare_en_text(text: str) -> str:
    split_lines = []

    for line in text.splitlines():
        if line.strip() == "":
            continue
        split_lines.append(line.strip() + " \n\n")

    return " ".join(split_lines)


def prepare_zh_text(text: str) -> str:
    split_lines = []

    for line in text.splitlines():
        if line.strip() == "":
            continue

        split_lines.append(
            " ".join(jieba.cut(line.strip())) + " \n\n"
        )

    return " ".join(split_lines)



def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_dataset_examples(args) -> List[dict]:
    dataset_name = args.dataset_name or "Helsinki-NLP/un_pc"
    dataset_config = args.dataset_config or "en-zh"
    dataset_split = args.dataset_split or "train"

    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    if args.dataset_seed is not None:
        dataset = dataset.shuffle(seed=args.dataset_seed)

    sample_count = args.dataset_num_samples or 5
    if sample_count > len(dataset):
        sample_count = len(dataset)

    subset = dataset.select(range(sample_count))
    records = []
    for idx, item in enumerate(subset):
        translation = item.get("translation", {})
        source_text = prepare_en_text(
            str(translation.get("en", "")).strip()
        )

        reference_text = prepare_zh_text(
            str(translation.get("zh", "")).strip()
        )
        if not source_text:
            continue

        records.append(
            {
                "sample_name": f"dataset_{idx:03d}",
                "source_text": source_text,
                "reference_text": reference_text,
            }
        )

    return records


def infer_output_suffix(source_path: Optional[Path]) -> str:
    if source_path is not None and source_path.suffix:
        return source_path.suffix
    return ".txt"


def build_generation_args(model, zh_tokenizer, args) -> dict:
    decoder_start_token_id = getattr(
        model.config,
        "decoder_start_token_id",
        None,
    )
    if decoder_start_token_id is None:
        decoder_start_token_id = zh_tokenizer.bos_token_id

    return {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": False,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "decoder_start_token_id": decoder_start_token_id,
        "eos_token_id": zh_tokenizer.eos_token_id,
        "pad_token_id": zh_tokenizer.pad_token_id,
    }


def generate_texts(
    model: EncoderDecoderModel,
    en_tokenizer,
    zh_tokenizer,
    source_texts: List[str],
    max_length: int,
    device: torch.device,
    args,
) -> List[str]:
    outputs: List[str] = []
    generation_kwargs = build_generation_args(model, zh_tokenizer, args)

    for text in source_texts:
        encoded = en_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=min(max_length, model.config.encoder.max_position_embeddings),
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        translated_text = zh_tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
        outputs.append(translated_text)

    return outputs


def save_predictions(
    output_dir: Path,
    sample_names: List[str],
    predictions: List[str],
    source_texts: List[str],
    source_paths: Optional[List[Optional[Path]]] = None,
    reference_texts: Optional[List[str]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []
    for sample_name, source_text, prediction, source_path, reference_text in zip(
        sample_names,
        source_texts,
        predictions,
        source_paths or [None] * len(predictions),
        reference_texts or [None] * len(predictions),
    ):
        suffix = infer_output_suffix(source_path)
        pred_path = output_dir / f"{sample_name}_pred{suffix}"
        pred_path.write_text(prediction, encoding="utf-8")

        src_path = output_dir / f"{sample_name}_src{suffix}"
        src_path.write_text(source_text, encoding="utf-8")

        if reference_text is not None:
            ref_path = output_dir / f"{sample_name}_ref{suffix}"
            ref_path.write_text(reference_text, encoding="utf-8")

        if source_path is not None:
            same_dir_output = source_path.parent / f"{sample_name}_pred{suffix}"
            same_dir_output.write_text(prediction, encoding="utf-8")

        summary_records.append(
            {
                "sample_name": sample_name,
                "source_text": source_text,
                "reference_text": reference_text,
                "prediction_text": prediction,
            }
        )

    summary_path = output_dir / "predictions.json"
    summary_path.write_text(
        json.dumps(summary_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved predictions to {output_dir}")


def main(args):
    print("Starting generation...")
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_dir}")

    en_tokenizer_dir = args.en_tokenizer_dir or str(Path(args.model_dir) / "en_tokenizer")
    zh_tokenizer_dir = args.zh_tokenizer_dir or str(Path(args.model_dir) / "zh_tokenizer")

    en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(zh_tokenizer_dir)
    zh_tokenizer.backend_tokenizer.decoder = WordPiece(
        prefix="##",
        cleanup=True,
        ##added this 
    )

    model = EncoderDecoderModel.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    source_texts: List[str] = []
    sample_names: List[str] = []
    source_paths: List[Optional[Path]] = []
    reference_texts: List[Optional[str]] = []

    if args.source_text is not None:
        source_texts = [args.source_text]
        sample_names = [args.sample_name or "sample_001"]
        source_paths = [None]
        reference_texts = [None]
    elif args.source_file is not None:
        source_path = Path(args.source_file)
        source_texts = [read_text_file(source_path)]
        sample_names = [args.sample_name or source_path.stem]
        source_paths = [source_path]
        reference_texts = [None]
    elif args.en_mmd_dir and args.split_json_file_path:
        with open(args.split_json_file_path, "r", encoding="utf-8") as fh:
            split_data = json.load(fh)

        selected_names = split_data.get(args.split_key, [])
        if args.num_samples is not None:
            selected_names = selected_names[: args.num_samples]

        input_dir = Path(args.en_mmd_dir)
        for sample_name in selected_names:
            input_path = input_dir / f"{sample_name}.mmd"
            if not input_path.exists():
                print(f"Missing input file: {input_path}")
                continue
            source_text = read_text_file(input_path)
            if not source_text:
                print(f"Empty input file: {input_path}")
                continue
            source_texts.append(source_text)
            sample_names.append(sample_name)
            source_paths.append(input_path)
            reference_texts.append(None)
    else:
        dataset_examples = load_dataset_examples(args)
        if not dataset_examples:
            raise ValueError(
                "No source texts were found. Provide --source_text, --source_file, or dataset inputs."
            )

        for example in dataset_examples:
            source_texts.append(example["source_text"])
            sample_names.append(example["sample_name"])
            source_paths.append(None)
            reference_texts.append(example.get("reference_text"))

    if not source_texts:
        raise ValueError("No source texts were found for generation")

    print(f"Generating {len(source_texts)} sample(s)...")
    predictions = generate_texts(
        model=model,
        en_tokenizer=en_tokenizer,
        zh_tokenizer=zh_tokenizer,
        source_texts=source_texts,
        max_length=args.max_length,
        device=device,
        args=args,
    )

    output_dir = Path(args.output_dir)
    save_predictions(
        output_dir,
        sample_names,
        predictions,
        source_texts,
        source_paths,
        reference_texts,
    )
    ids = zh_tokenizer.encode(
        "卢旺达",
        add_special_tokens=False,
    )

    for idx, text in enumerate(source_texts, start=1):
        print(f"\n--- Sample {idx} ---")
        print(text)

    


    
    for idx, text in enumerate(predictions, start=1):
        print(f"\n--- Sample {idx} ---")
        print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate translations with the UN-pretrained EncoderDecoder model")
    parser.add_argument("--model_dir", required=True, help="Path to the trained model directory, e.g. .../final_model")
    parser.add_argument("--en_tokenizer_dir", default=None, help="English tokenizer directory; defaults to <model_dir>/en_tokenizer")
    parser.add_argument("--zh_tokenizer_dir", default=None, help="Chinese tokenizer directory; defaults to <model_dir>/zh_tokenizer")
    parser.add_argument("--source_text", default=None, help="Single English sentence to translate")
    parser.add_argument("--source_file", default=None, help="Path to a text file containing the English source")
    parser.add_argument("--sample_name", default=None, help="Name for single-sample output")
    parser.add_argument("--en_mmd_dir", default=None, help="Directory containing .mmd source files for batch generation")
    parser.add_argument("--split_json_file_path", default=None, help="Path to a JSON file containing a sample list")
    parser.add_argument("--split_key", default="test_name_list", help="Key inside the split JSON file")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit the number of samples in batch mode")
    parser.add_argument("--dataset_name", default="Helsinki-NLP/un_pc", help="Hugging Face dataset name used as fallback")
    parser.add_argument("--dataset_config", default="en-zh", help="Hugging Face dataset config used as fallback")
    parser.add_argument("--dataset_split", default="train", help="Hugging Face dataset split used as fallback")
    parser.add_argument("--dataset_seed", type=int, default=None, help="Seed used when shuffling fallback dataset samples")
    parser.add_argument("--dataset_num_samples", type=int, default=5, help="Number of fallback samples to draw from the training dataset")
    parser.add_argument("--output_dir", default="/home/pj24001974/ku60000761/project/test_project/M4Doc/results/model_trans_un", help="Directory for generated outputs")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(args)
