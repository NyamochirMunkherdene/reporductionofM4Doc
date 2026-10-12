import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, EncoderDecoderModel


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def print_tokenizer_info(name, tokenizer):
    print("\n" + "=" * 80)
    print(f"{name} TOKENIZER")
    print("=" * 80)

    print("class:", tokenizer.__class__.__name__)
    print("len(tokenizer):", len(tokenizer))
    print("vocab_size:", getattr(tokenizer, "vocab_size", None))

    print("bos_token:", repr(tokenizer.bos_token))
    print("bos_token_id:", tokenizer.bos_token_id)

    print("eos_token:", repr(tokenizer.eos_token))
    print("eos_token_id:", tokenizer.eos_token_id)

    print("pad_token:", repr(tokenizer.pad_token))
    print("pad_token_id:", tokenizer.pad_token_id)

    print("cls_token:", repr(tokenizer.cls_token))
    print("cls_token_id:", tokenizer.cls_token_id)

    print("sep_token:", repr(tokenizer.sep_token))
    print("sep_token_id:", tokenizer.sep_token_id)

    print("unk_token:", repr(tokenizer.unk_token))
    print("unk_token_id:", tokenizer.unk_token_id)


def main(args):
    print("=" * 80)
    print("STARTING DEBUG INFERENCE")
    print("=" * 80)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device)
    print("model_dir:", args.model_dir)
    print("en_tokenizer_dir:", args.en_tokenizer_dir)
    print("zh_tokenizer_dir:", args.zh_tokenizer_dir)

    # ------------------------------------------------------------------
    # Load everything
    # ------------------------------------------------------------------

    en_tokenizer = AutoTokenizer.from_pretrained(
        args.en_tokenizer_dir
    )

    zh_tokenizer = AutoTokenizer.from_pretrained(
        args.zh_tokenizer_dir
    )

    model = EncoderDecoderModel.from_pretrained(
        args.model_dir
    )

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Tokenizer diagnostics
    # ------------------------------------------------------------------

    print_tokenizer_info("ENGLISH", en_tokenizer)
    print_tokenizer_info("CHINESE", zh_tokenizer)

    # ------------------------------------------------------------------
    # Model config diagnostics
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("MODEL CONFIG")
    print("=" * 80)

    print("decoder_start_token_id:",
          model.config.decoder_start_token_id)

    print("model pad_token_id:",
          model.config.pad_token_id)

    print("model eos_token_id:",
          model.config.eos_token_id)

    print("model bos_token_id:",
          model.config.bos_token_id)

    print("\nEncoder vocab size:")
    print(model.config.encoder.vocab_size)

    print("Decoder vocab size:")
    print(model.config.decoder.vocab_size)

    print("\nActual embeddings:")

    encoder_embedding_size = (
        model.encoder.get_input_embeddings().weight.shape
    )

    decoder_embedding_size = (
        model.decoder.get_input_embeddings().weight.shape
    )

    print("encoder embedding shape:", encoder_embedding_size)
    print("decoder embedding shape:", decoder_embedding_size)

    print("\nTokenizer/model vocab comparison:")

    print(
        f"English tokenizer: {len(en_tokenizer)} | "
        f"encoder embeddings: {encoder_embedding_size[0]}"
    )

    print(
        f"Chinese tokenizer: {len(zh_tokenizer)} | "
        f"decoder embeddings: {decoder_embedding_size[0]}"
    )

    if len(en_tokenizer) != encoder_embedding_size[0]:
        print(
            "\nWARNING: English tokenizer size does NOT match "
            "encoder embedding size!"
        )

    if len(zh_tokenizer) != decoder_embedding_size[0]:
        print(
            "\nWARNING: Chinese tokenizer size does NOT match "
            "decoder embedding size!"
        )

    # ------------------------------------------------------------------
    # Determine decoder special IDs
    # ------------------------------------------------------------------

    decoder_start_token_id = model.config.decoder_start_token_id

    if decoder_start_token_id is None:
        print("\nWARNING: decoder_start_token_id is None!")

        candidates = [
            ("zh bos", zh_tokenizer.bos_token_id),
            ("zh cls", zh_tokenizer.cls_token_id),
        ]

        for name, value in candidates:
            print(f"candidate {name}:", value)

        decoder_start_token_id = (
            zh_tokenizer.bos_token_id
            if zh_tokenizer.bos_token_id is not None
            else zh_tokenizer.cls_token_id
        )

    eos_token_id = model.config.eos_token_id

    if eos_token_id is None:
        eos_token_id = zh_tokenizer.eos_token_id

    if eos_token_id is None:
        eos_token_id = zh_tokenizer.sep_token_id

    pad_token_id = model.config.pad_token_id

    if pad_token_id is None:
        pad_token_id = zh_tokenizer.pad_token_id

    print("\nGeneration token IDs:")
    print("decoder_start_token_id:", decoder_start_token_id)
    print("eos_token_id:", eos_token_id)
    print("pad_token_id:", pad_token_id)

    for token_id_name, token_id in [
        ("decoder_start", decoder_start_token_id),
        ("eos", eos_token_id),
        ("pad", pad_token_id),
    ]:
        if token_id is not None:
            print(
                token_id_name,
                "=>",
                repr(
                    zh_tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                    )
                ),
            )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    with open(
        args.split_json_file_path,
        "r",
        encoding="utf-8",
    ) as fh:
        split_data = json.load(fh)

    print("\nAvailable split keys:")
    print(list(split_data.keys()))

    sample_names = split_data[args.split_key]

    if args.num_samples is not None:
        sample_names = sample_names[:args.num_samples]

    print("\nSamples:", sample_names)

    input_dir = Path(args.en_mmd_dir)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    for index, sample_name in enumerate(sample_names, start=1):

        print("\n")
        print("#" * 100)
        print(f"SAMPLE {index}: {sample_name}")
        print("#" * 100)

        input_path = input_dir / f"{sample_name}.mmd"

        if not input_path.exists():
            print("MISSING:", input_path)
            continue

        source_text = read_text_file(input_path)

        print("\nSOURCE TEXT:")
        print(repr(source_text[:1000]))

        print("\nSOURCE LENGTH CHARACTERS:")
        print(len(source_text))

        # --------------------------------------------------------------
        # Tokenization
        # --------------------------------------------------------------

        encoded = en_tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )

        print("\nENCODER TOKENIZATION")
        print("input_ids shape:", encoded["input_ids"].shape)
        print("attention_mask shape:", encoded["attention_mask"].shape)

        print("\nFirst input token IDs:")
        print(encoded["input_ids"][0][:100].tolist())

        print("\nTokens:")
        print(
            en_tokenizer.convert_ids_to_tokens(
                encoded["input_ids"][0][:100].tolist()
            )
        )

        reconstructed_source = en_tokenizer.decode(
            encoded["input_ids"][0],
            skip_special_tokens=False,
        )

        print("\nDecoded encoder input:")
        print(repr(reconstructed_source[:1000]))

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # --------------------------------------------------------------
        # Check first decoder step manually
        # --------------------------------------------------------------

        start_ids = torch.tensor(
            [[decoder_start_token_id]],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=start_ids,
            )

        first_logits = outputs.logits[:, -1, :]

        probs = torch.softmax(first_logits, dim=-1)

        top_probs, top_ids = torch.topk(
            probs,
            k=min(20, probs.shape[-1]),
            dim=-1,
        )

        print("\n" + "-" * 80)
        print("TOP TOKENS AT FIRST DECODER STEP")
        print("-" * 80)

        for rank, (token_id, probability) in enumerate(
            zip(
                top_ids[0].tolist(),
                top_probs[0].tolist(),
            ),
            start=1,
        ):
            token = zh_tokenizer.convert_ids_to_tokens(token_id)

            decoded = zh_tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )

            print(
                f"{rank:02d}. "
                f"id={token_id:<8} "
                f"prob={probability:.6f} "
                f"token={repr(token):<30} "
                f"decoded={repr(decoded)}"
            )

        # --------------------------------------------------------------
        # Greedy generation
        # --------------------------------------------------------------

        print("\n" + "-" * 80)
        print("GREEDY GENERATION")
        print("-" * 80)

        with torch.no_grad():

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,

                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,

                max_new_tokens=args.max_new_tokens,

                num_beams=1,
                do_sample=False,

                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = generated.sequences[0]

        print("RAW GENERATED IDS:")
        print(generated_ids.tolist())

        print("\nGENERATED TOKENS:")
        print(
            zh_tokenizer.convert_ids_to_tokens(
                generated_ids.tolist()
            )
        )

        print("\nDECODE WITH SPECIAL TOKENS:")
        print(
            repr(
                zh_tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=False,
                )
            )
        )

        print("\nNORMAL DECODE:")
        print(
            repr(
                zh_tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            )
        )

        # --------------------------------------------------------------
        # Inspect each generation step
        # --------------------------------------------------------------

        print("\nGENERATION STEPS:")

        for step, score in enumerate(generated.scores):

            step_probs = torch.softmax(score[0], dim=-1)

            probability, token_id = torch.max(
                step_probs,
                dim=-1,
            )

            token_id = token_id.item()
            probability = probability.item()

            print(
                f"step={step:<3} "
                f"id={token_id:<7} "
                f"prob={probability:.5f} "
                f"token={repr(zh_tokenizer.convert_ids_to_tokens(token_id))} "
                f"decoded={repr(zh_tokenizer.decode([token_id], skip_special_tokens=False))}"
            )

            if step >= 30:
                print("... stopping detailed step display")
                break

        # --------------------------------------------------------------
        # Beam search comparison
        # --------------------------------------------------------------

        print("\n" + "-" * 80)
        print("BEAM SEARCH COMPARISON")
        print("-" * 80)

        with torch.no_grad():

            beam_count = max(1, args.num_beams)

            beam_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                max_new_tokens=args.max_new_tokens,
                num_beams=beam_count,
                num_return_sequences=beam_count,
                do_sample=False,
            )

        for beam_index, ids in enumerate(beam_ids, start=1):

            print(
                f"\nBeam {beam_index}:"
            )

            print("IDs:", ids.tolist())

            print(
                "tokens:",
                zh_tokenizer.convert_ids_to_tokens(
                    ids.tolist()
                ),
            )

            print(
                "text:",
                repr(
                    zh_tokenizer.decode(
                        ids,
                        skip_special_tokens=True,
                    )
                ),
            )

    print("\n" + "=" * 80)
    print("DEBUG FINISHED")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug inference for EncoderDecoder translation model"
    )

    parser.add_argument(
        "--model_dir",
        required=True,
    )

    parser.add_argument(
        "--en_tokenizer_dir",
        required=True,
    )

    parser.add_argument(
        "--zh_tokenizer_dir",
        required=True,
    )

    parser.add_argument(
        "--en_mmd_dir",
        required=True,
    )

    parser.add_argument(
        "--split_json_file_path",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
    )

    parser.add_argument(
        "--split_key",
        default="test_name_list",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    main(args)