import os
import json
import argparse


def train(args):

    os.environ.setdefault(
        "WANDB_PROJECT",
        "M4Doc-Translation-Sandbox",
    )

    MAX_LENGTH = args.max_length

    # ============================================================
    # Tokenizers
    # ============================================================
    from transformers import AutoTokenizer

    en_tokenizer = AutoTokenizer.from_pretrained(
        args.en_tokenizer_dir
    )

    zh_tokenizer = AutoTokenizer.from_pretrained(
        args.zh_tokenizer_dir
    )

    # ============================================================
    # Dataset split
    # ============================================================
    with open(args.split_json_file_path, "r") as f:
        json_dict = json.load(f)

    train_name_list = json_dict["train_name_list"]
    valid_name_list = json_dict["valid_name_list"]

    from my_dataset_trans import DoTADatasetTrans

    train_dataset = DoTADatasetTrans(
        en_tokenizer,
        zh_tokenizer,
        args.en_mmd_dir,
        args.zh_mmd_dir,
        train_name_list,
        MAX_LENGTH,
    )

    valid_dataset = DoTADatasetTrans(
        en_tokenizer,
        zh_tokenizer,
        args.en_mmd_dir,
        args.zh_mmd_dir,
        valid_name_list,
        MAX_LENGTH,
    )

    print(
        f"Number of training samples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Number of validation samples: "
        f"{len(valid_dataset):,}"
    )

    # ============================================================
    # Load pretrained Transformer
    # ============================================================
    from transformers import EncoderDecoderModel

    print(
        f"Loading pretrained translation model from: "
        f"{args.pretrained_model_dir}"
    )

    model = EncoderDecoderModel.from_pretrained(
        args.pretrained_model_dir
    )

    print(
        "Original encoder max positions:",
        model.config.encoder.max_position_embeddings,
    )

    print(
        "Original decoder max positions:",
        model.config.decoder.max_position_embeddings,
    )

    # ============================================================
    # Generation config
    # ============================================================
    model.config.decoder_start_token_id = (
        zh_tokenizer.bos_token_id
    )

    model.config.pad_token_id = (
        zh_tokenizer.pad_token_id
    )

    model.config.eos_token_id = (
        zh_tokenizer.eos_token_id
    )

    model.config.max_length = MAX_LENGTH

    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 0
    model.config.length_penalty = 1.0
    model.config.num_beams = 4

    # ============================================================
    # Sanity checks
    # ============================================================

    print(
        "Encoder position embeddings:",
        model.encoder.embeddings.position_embeddings.weight.shape,
    )

    print(
        "Decoder position embeddings:",
        model.decoder.bert.embeddings.position_embeddings.weight.shape,
    )

    print(
        "Encoder vocab:",
        model.config.encoder.vocab_size,
    )

    print(
        "English tokenizer:",
        len(en_tokenizer),
    )

    print(
        "Decoder vocab:",
        model.config.decoder.vocab_size,
    )

    print(
        "Chinese tokenizer:",
        len(zh_tokenizer),
    )

    assert (
        model.config.encoder.vocab_size
        == len(en_tokenizer)
    ), (
        "Encoder vocabulary does not match "
        "English tokenizer."
    )

    assert (
        model.config.decoder.vocab_size
        == len(zh_tokenizer)
    ), (
        "Decoder vocabulary does not match "
        "Chinese tokenizer."
    )

    assert (
        model.encoder.embeddings
        .position_embeddings.num_embeddings
        >= MAX_LENGTH
    )

    assert (
        model.decoder.bert.embeddings
        .position_embeddings.num_embeddings
        >= MAX_LENGTH
    )

    # ============================================================
    # Effective batch size
    # ============================================================

    world_size = int(
        os.environ.get("WORLD_SIZE", 1)
    )

    denominator = (
        world_size
        * args.batch_size_per_gpu
    )

    if args.batch_size % denominator != 0:
        raise ValueError(
            f"Global batch size ({args.batch_size}) "
            f"must be divisible by world_size "
            f"({world_size}) * per-GPU batch size "
            f"({args.batch_size_per_gpu})."
        )

    gradient_accumulation_steps = (
        args.batch_size // denominator
    )

    effective_batch_size = (
        args.batch_size_per_gpu
        * world_size
        * gradient_accumulation_steps
    )

    print(
        f"World size: {world_size}"
    )

    print(
        f"Per-GPU batch size: "
        f"{args.batch_size_per_gpu}"
    )

    print(
        f"Gradient accumulation steps: "
        f"{gradient_accumulation_steps}"
    )

    print(
        f"Effective global batch size: "
        f"{effective_batch_size}"
    )

    # ============================================================
    # Trainer
    # ============================================================
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name="transformer-base-dota-finetune",

        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",

        max_grad_norm=1.0,

        logging_strategy="steps",
        logging_steps=args.logging_steps,

        evaluation_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,

        bf16=args.bf16,

        dataloader_num_workers=args.dataloader_num_workers,

        deepspeed=args.deepspeed,

        report_to=[
            "wandb",
            "tensorboard",
        ],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    # ============================================================
    # Parameters
    # ============================================================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"{total_params:,} total parameters."
    )

    print(
        f"{trainable_params:,} trainable parameters."
    )

    # ============================================================
    # Fine-tuning
    # ============================================================

    # trainer.train()
    trainer.train(
        resume_from_checkpoint="/home/pj24001974/ku60000761/project/test_project/M4Doc/models/dota_trans_jieba_lm_1536_v2/checkpoint-48000"
    )

    print("Training finished.")
    print(
        f"Checkpoints were saved every "
        f"{args.save_steps} steps to:"
    )
    print(args.output_dir)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # ============================================================
    # Pretrained checkpoint
    # ============================================================

    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        required=True,
    )

    # ============================================================
    # Tokenizers
    # ============================================================

    parser.add_argument(
        "--en_tokenizer_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--zh_tokenizer_dir",
        type=str,
        required=True,
    )

    # ============================================================
    # DoTA
    # ============================================================

    parser.add_argument(
        "--en_mmd_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--zh_mmd_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--split_json_file_path",
        type=str,
        required=True,
    )

    # ============================================================
    # Output
    # ============================================================

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    # ============================================================
    # Sequence length
    #
    # Pretrained at 512.
    # Positional embeddings will be expanded to 1536.
    # ============================================================

    parser.add_argument(
        "--max_length",
        type=int,
        default=1536,
    )

    # ============================================================
    # Batch
    # ============================================================

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=4,
    )

    # ============================================================
    # Fine-tuning hyperparameters
    # ============================================================

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
    )

    # ============================================================
    # Evaluate/save candidate every 1000 optimizer steps
    # ============================================================

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )

    # ============================================================
    # Hardware
    # ============================================================

    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    train(args)