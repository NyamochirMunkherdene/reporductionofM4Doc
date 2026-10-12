import os
import argparse
import inspect
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BertConfig,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from un_dataset import UNTranslationDataset


def rank0_print(*args, **kwargs):
    """Print once under torchrun/DDP instead of once per GPU process."""
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs, flush=True)


def get_special_id(tokenizer, primary, fallback):
    value = getattr(tokenizer, primary, None)
    if value is None:
        value = getattr(tokenizer, fallback, None)
    return value


def describe_tokenizer(name, tokenizer):
    rank0_print(f"\n[{name} tokenizer]")
    rank0_print("vocab size:", len(tokenizer))
    rank0_print("pad:", tokenizer.pad_token, tokenizer.pad_token_id)
    rank0_print("bos:", tokenizer.bos_token, tokenizer.bos_token_id)
    rank0_print("eos:", tokenizer.eos_token, tokenizer.eos_token_id)
    rank0_print("cls:", tokenizer.cls_token, tokenizer.cls_token_id)
    rank0_print("sep:", tokenizer.sep_token, tokenizer.sep_token_id)
    rank0_print("unk:", tokenizer.unk_token, tokenizer.unk_token_id)


def build_model(en_tokenizer, zh_tokenizer, max_length):
    """
    MODEL ARCHITECTURE

    Encoder: 6 BERT-style Transformer layers
      - self-attention: 8 heads
      - hidden size: 512
      - FFN/intermediate size: 2048

    Decoder: 6 BERT-style Transformer decoder layers
      - masked self-attention: 8 heads
      - cross-attention to encoder output
      - hidden size: 512
      - FFN/intermediate size: 2048
    """

    en_bos = get_special_id(en_tokenizer, "bos_token_id", "cls_token_id")
    en_eos = get_special_id(en_tokenizer, "eos_token_id", "sep_token_id")
    zh_bos = get_special_id(zh_tokenizer, "bos_token_id", "cls_token_id")
    zh_eos = get_special_id(zh_tokenizer, "eos_token_id", "sep_token_id")

    if en_tokenizer.pad_token_id is None:
        raise ValueError("English tokenizer has no pad_token_id")
    if zh_tokenizer.pad_token_id is None:
        raise ValueError("Chinese tokenizer has no pad_token_id")
    if zh_bos is None:
        raise ValueError("Chinese tokenizer has neither BOS nor CLS token; decoder cannot start")
    if zh_eos is None:
        raise ValueError("Chinese tokenizer has neither EOS nor SEP token; generation cannot terminate")

    encoder_config = BertConfig(
        vocab_size=len(en_tokenizer),
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=max_length,
        type_vocab_size=1,
        pad_token_id=en_tokenizer.pad_token_id,
        bos_token_id=en_bos,
        eos_token_id=en_eos,
    )

    decoder_config = BertConfig(
        vocab_size=len(zh_tokenizer),
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=max_length,
        type_vocab_size=1,
        pad_token_id=zh_tokenizer.pad_token_id,
        bos_token_id=zh_bos,
        eos_token_id=zh_eos,
        is_decoder=True,
        add_cross_attention=True,
    )

    config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )

    config.decoder_start_token_id = zh_bos
    config.pad_token_id = zh_tokenizer.pad_token_id
    config.eos_token_id = zh_eos
    config.max_length = max_length

    model = EncoderDecoderModel(config=config)

    # Hard checks: decoder must truly be a decoder with encoder-decoder attention.
    assert model.config.decoder.is_decoder is True
    assert model.config.decoder.add_cross_attention is True

    return model


def debug_dataset(train_dataset, raw_dataset, en_tokenizer, zh_tokenizer):
    """DATA/TOKENIZATION DEBUGGING: run before Trainer."""
    rank0_print("\n========== DATASET DEBUG ==========")

    raw = raw_dataset[0]
    rank0_print("Raw dataset keys:", raw.keys())
    rank0_print("Raw example:", raw)

    sample = train_dataset[0]
    rank0_print("Processed sample keys:", sample.keys())

    for key, value in sample.items():
        if torch.is_tensor(value):
            rank0_print(
                f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}, "
                f"min={value.min().item()}, max={value.max().item()}"
            )
        else:
            rank0_print(f"{key}: type={type(value)}, value={value}")

    if "input_ids" not in sample:
        raise KeyError("UNTranslationDataset must return input_ids")
    if "labels" not in sample:
        raise KeyError("UNTranslationDataset must return labels")

    src_ids = sample["input_ids"]
    labels = sample["labels"]
    if not torch.is_tensor(src_ids):
        src_ids = torch.tensor(src_ids)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
        

    valid_labels = labels[labels != -100]

    rank0_print("\nDecoded EN input:")
    rank0_print(en_tokenizer.decode(src_ids.tolist(), skip_special_tokens=False))
    rank0_print("\nDecoded ZH labels:")
    rank0_print(zh_tokenizer.decode(valid_labels.tolist(), skip_special_tokens=False))

    eos_id = get_special_id(zh_tokenizer, "eos_token_id", "sep_token_id")
    rank0_print("ZH effective EOS id:", eos_id)
    rank0_print("Labels contain EOS:", eos_id in valid_labels.tolist())
    rank0_print("Number of supervised target tokens:", len(valid_labels))

    if len(valid_labels) == 0:
        raise ValueError("All target labels are -100; model has nothing to learn")


def debug_one_forward_pass(model, train_dataset):
    """MODEL DEBUGGING: verifies encoder + decoder + attention + loss all run."""
    rank0_print("\n========== ONE-BATCH FORWARD DEBUG ==========")

    loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # critical for debugging: exceptions stay readable
        collate_fn=default_data_collator,
    )
    batch = next(iter(loader))

    # Keep only inputs EncoderDecoderModel needs for this baseline.
    keep = {"input_ids", "attention_mask", "labels"}
    batch = {k: v for k, v in batch.items() if k in keep}

    rank0_print("Batch keys:", batch.keys())
    for key, value in batch.items():
        rank0_print(f"{key}: {tuple(value.shape)} {value.dtype}")

    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
        labels=batch["labels"]
    )
    rank0_print("labels[0][:20]:", batch["labels"][0, :20].tolist())
    rank0_print("decoder_input_ids[0][:20]:", decoder_input_ids[0, :20].tolist())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch = {k: v.to(device) for k, v in batch.items()}

    model.train()
    outputs = model(**batch)

    rank0_print("Forward loss:", float(outputs.loss.detach().cpu()))
    rank0_print("Logits shape:", tuple(outputs.logits.shape))

    if not torch.isfinite(outputs.loss):
        raise FloatingPointError(f"Non-finite loss on first batch: {outputs.loss}")

    outputs.loss.backward()
    rank0_print("Backward pass: OK")

    model.zero_grad(set_to_none=True)
    model = model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


def make_training_args(args, grad_accum):
    """TRAINING CONFIGURATION."""
    kwargs = dict(
        output_dir=args.output_dir,
        run_name="transformer-base-Text-Only",
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.batch_size_per_gpu,
        gradient_accumulation_steps=grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        bf16=args.bf16,
        dataloader_num_workers=(0 if args.debug else args.dataloader_num_workers),
        report_to=([] if args.debug else ["wandb", "tensorboard"]),
        remove_unused_columns=False,
    )

    # Transformers renamed evaluation_strategy -> eval_strategy in newer releases.
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    kwargs["eval_steps"] = args.eval_steps

    if args.deepspeed:
        kwargs["deepspeed"] = args.deepspeed

    return TrainingArguments(**kwargs)


def train(args):
    os.environ.setdefault("WANDB_PROJECT", "M4Doc-Translation-Sandbox")

    # ================================================================
    # 1. TOKENIZERS
    # ================================================================
    en_tokenizer = AutoTokenizer.from_pretrained(args.en_tokenizer_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(args.zh_tokenizer_dir)
    describe_tokenizer("EN", en_tokenizer)
    describe_tokenizer("ZH", zh_tokenizer)

    # ================================================================
    # 2. PARALLEL TEXT DATA: UN CORPUS
    # ================================================================
    raw_dataset = load_dataset(
        "Helsinki-NLP/un_pc",
        "en-zh",
        split="train",
    )

    needed = args.max_train_samples + args.max_eval_samples
    if args.max_train_samples is not None:
        raw_dataset = raw_dataset.shuffle(seed=args.seed).select(
            range(min(needed, len(raw_dataset)))
        )

    split_dataset = raw_dataset.train_test_split(
        test_size=args.max_eval_samples,
        seed=args.seed,
    )
    train_data = split_dataset["train"]
    valid_data = split_dataset["test"]

    train_dataset = UNTranslationDataset(
        hf_dataset=train_data,
        en_tokenizer=en_tokenizer,
        zh_tokenizer=zh_tokenizer,
        max_length=args.max_length,
    )
    valid_dataset = UNTranslationDataset(
        hf_dataset=valid_data,
        en_tokenizer=en_tokenizer,
        zh_tokenizer=zh_tokenizer,
        max_length=args.max_length,
    )

    rank0_print(f"Training samples: {len(train_dataset):,}")
    rank0_print(f"Validation samples: {len(valid_dataset):,}")

    # ================================================================
    # 3. ENCODER + DECODER MODEL
    #    Attention is INSIDE these BERT layers.
    # ================================================================
    model = build_model(en_tokenizer, zh_tokenizer, args.max_length)

    rank0_print("\n========== MODEL STRUCTURE ==========")
    rank0_print("Encoder layers:", model.config.encoder.num_hidden_layers)
    rank0_print("Encoder attention heads:", model.config.encoder.num_attention_heads)
    rank0_print("Decoder layers:", model.config.decoder.num_hidden_layers)
    rank0_print("Decoder attention heads:", model.config.decoder.num_attention_heads)
    rank0_print("Decoder is_decoder:", model.config.decoder.is_decoder)
    rank0_print("Decoder add_cross_attention:", model.config.decoder.add_cross_attention)
    rank0_print("Total parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    # ================================================================
    # 4. DEBUG BEFORE DISTRIBUTED TRAINING
    # ================================================================
    if args.debug and int(os.environ.get("RANK", "0")) == 0:
        debug_dataset(train_dataset, train_data, en_tokenizer, zh_tokenizer)
        model = debug_one_forward_pass(model, train_dataset)
        rank0_print("\nDEBUG CHECKS PASSED. Exiting because --debug was supplied.")
        return

    # ================================================================
    # 5. DISTRIBUTED BATCH SIZE
    # ================================================================
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available() and world_size == 1:
        world_size = 1

    micro_global_batch = world_size * args.batch_size_per_gpu
    if args.batch_size % micro_global_batch != 0:
        raise ValueError(
            f"batch_size={args.batch_size} must be divisible by "
            f"WORLD_SIZE({world_size}) * batch_size_per_gpu({args.batch_size_per_gpu}) "
            f"= {micro_global_batch}"
        )
    grad_accum = args.batch_size // micro_global_batch

    rank0_print("\n========== DISTRIBUTED TRAINING ==========")
    rank0_print("WORLD_SIZE:", world_size)
    rank0_print("Visible CUDA devices:", torch.cuda.device_count())
    rank0_print("Per-device batch size:", args.batch_size_per_gpu)
    rank0_print("Gradient accumulation:", grad_accum)
    rank0_print("Effective global batch size:", world_size * args.batch_size_per_gpu * grad_accum)

    # ================================================================
    # 6. HUGGING FACE TRAINER
    # ================================================================
    training_args = make_training_args(args, grad_accum)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator,
    )

    # ================================================================
    # 7. OPTIMIZATION / TRAINING
    # ================================================================
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ================================================================
    # 8. SAVE FULL MT MODEL + TOKENIZERS + DECODER
    # ================================================================
    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    en_tokenizer.save_pretrained(os.path.join(final_model_dir, "en_tokenizer"))
    zh_tokenizer.save_pretrained(os.path.join(final_model_dir, "zh_tokenizer"))
    trainer.model.decoder.save_pretrained(os.path.join(args.output_dir, "trans_decoder"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_tokenizer_dir", required=True)
    parser.add_argument("--zh_tokenizer_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=1_000_000)
    parser.add_argument("--max_eval_samples", type=int, default=2_000)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    train(parser.parse_args())