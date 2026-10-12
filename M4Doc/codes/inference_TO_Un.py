import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, EncoderDecoderModel


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def main(args):
    print("started")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_dir}")

    en_tokenizer = AutoTokenizer.from_pretrained(args.en_tokenizer_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(args.zh_tokenizer_dir)
    model = EncoderDecoderModel.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    with open(args.split_json_file_path, "r", encoding="utf-8") as fh:
        split_data = json.load(fh)

    sample_names = split_data[args.split_key]
    if args.num_samples is not None:
        sample_names = sample_names[: args.num_samples]

    print(f"Using {len(sample_names)} samples")

    input_dir = Path(args.en_mmd_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, sample_name in enumerate(sample_names, start=1):
        input_path = input_dir / f"{sample_name}.mmd"
        if not input_path.exists():
            print(f"Missing {input_path}")
            continue

        source_text = read_text_file(input_path)
        if not source_text:
            print(f"[Empty] {input_path}")
            continue

        encoded = en_tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            max_length=min(args.max_length, model.config.encoder.max_position_embeddings),
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
            if decoder_start_token_id is None:
                decoder_start_token_id = zh_tokenizer.bos_token_id

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=zh_tokenizer.eos_token_id,
                pad_token_id=zh_tokenizer.pad_token_id,
            )

        translated_text = zh_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output_path = output_dir / f"{sample_name}.mmd"
        output_path.write_text(translated_text, encoding="utf-8")

        results.append({
            "sample_name": sample_name,
            "source_text": source_text,
            "prediction_text": translated_text,
        })

        if index % 10 == 0 or index == len(sample_names):
            print(f"Processed {index}/{len(sample_names)}")

    summary_path = output_dir / "predictions.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Predictions saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for a UN-pretrained EncoderDecoder model")
    parser.add_argument("--model_dir", required=True, help="Path to the trained model checkpoint directory")
    parser.add_argument("--en_tokenizer_dir", required=True, help="English tokenizer directory")
    parser.add_argument("--zh_tokenizer_dir", required=True, help="Chinese tokenizer directory")
    parser.add_argument("--en_mmd_dir", required=True, help="Directory containing English reference files")
    parser.add_argument("--split_json_file_path", required=True, help="Path to split_dataset.json")
    parser.add_argument("--output_dir", required=True, help="Directory for generated outputs")
    parser.add_argument("--split_key", default="test_name_list", help="Key in the split JSON")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to generate; default is 2 for a quick smoke test")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()
    main(args)
