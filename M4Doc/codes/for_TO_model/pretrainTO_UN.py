import os
import json
import torch
import jieba
import re
import argparse 
from datasets import load_dataset
from un_dataset import UNTranslationDataset


def train(args):
    os.environ.setdefault(
        "WANDB_PROJECT",
        "M4Doc-T0_UN",
    )


    TRAIN_MAX_LENGTH = args.max_length
    MODEL_MAX_LENGTH = args.model_max_length    

    
    from transformers import AutoTokenizer
    en_tokenizer = AutoTokenizer.from_pretrained(args.en_tokenizer_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(args.zh_tokenizer_dir)

    raw_dataset = load_dataset(
        "Helsinki-NLP/un_pc",
        "en-zh",
        split="train",
    )

    if args.max_train_samples is not None:
        sample_count = min(
            args.max_train_samples + args.max_eval_samples,
            len(raw_dataset),
        )

        raw_dataset = raw_dataset.shuffle(seed=args.seed).select(
            range(sample_count)
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
        max_length=TRAIN_MAX_LENGTH,
    )

    valid_dataset = UNTranslationDataset(
        hf_dataset=valid_data,
        en_tokenizer=en_tokenizer,
        zh_tokenizer=zh_tokenizer,
        max_length=TRAIN_MAX_LENGTH,
    )

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(valid_dataset):,}")


    # ============================================================
    # MODEL CONFIGURATION
    # ============================================================

    from transformers import EncoderDecoderModel, EncoderDecoderConfig, BertConfig

    encoder_config = BertConfig(
        vocab_size=len(en_tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=MODEL_MAX_LENGTH,
        type_vocab_size=1,
        pad_token_id=en_tokenizer.pad_token_id,
    )

    decoder_config = BertConfig(
        vocab_size=len(zh_tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=MODEL_MAX_LENGTH,
        type_vocab_size=1,
        pad_token_id=zh_tokenizer.pad_token_id,

        # Important for translation decoder
        is_decoder=True,
        add_cross_attention=True,
    )

    encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config,
        decoder_config,
    )

    encoder_decoder_config.decoder_start_token_id = zh_tokenizer.bos_token_id
    encoder_decoder_config.eos_token_id = zh_tokenizer.eos_token_id
    encoder_decoder_config.pad_token_id = zh_tokenizer.pad_token_id

    model = EncoderDecoderModel(config=encoder_decoder_config)


    num_gpu = torch.cuda.device_count()

    if num_gpu == 0:
        raise RuntimeError("No CUDA GPUs detected.")

    micro_global_batch = num_gpu * args.batch_size_per_gpu

    if args.batch_size % micro_global_batch != 0:
        raise ValueError(
            f"batch_size={args.batch_size} must be divisible by "
            f"num_gpu × batch_size_per_gpu="
            f"{num_gpu} × {args.batch_size_per_gpu}={micro_global_batch}"
        )

    gradient_accumulation_steps = (
        args.batch_size // micro_global_batch
    )

   
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name="transformer-base-Text-Only-1e-4-15k",
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,

        max_steps=args.max_steps,# added this, 
        lr_scheduler_type="linear",
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,

        logging_strategy='steps',
        logging_steps=1,

        evaluation_strategy='steps',
        eval_steps=args.eval_steps,

        save_strategy='steps',
        save_steps=args.save_steps,
        
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        
        report_to=['wandb', 'tensorboard'],
        deepspeed=args.deepspeed,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(training_args)
    
    total_params = sum(p.numel() for p in model.decoder.parameters())
    print(f'{total_params:,} decoder parameters.')


    if args.resume_from_checkpoint:
        trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint
        )
    else:
        trainer.train()

    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)

    en_tokenizer.save_pretrained(
        os.path.join(final_model_dir, "en_tokenizer")
    )
    zh_tokenizer.save_pretrained(
        os.path.join(final_model_dir, "zh_tokenizer")
    )

    trans_decoder = trainer.model.decoder
    trans_decoder.save_pretrained(
        os.path.join(args.output_dir, "trans_decoder")
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_tokenizer_dir", type=str)
    parser.add_argument("--zh_tokenizer_dir", type=str)
    parser.add_argument("--en_mmd_dir", type=str)
    parser.add_argument("--zh_mmd_dir", type=str)
    parser.add_argument("--split_json_file_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--model_max_length", type=int, default=2048,)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--max_train_samples", type=int, default=1_000_000,)
    parser.add_argument("--max_eval_samples", type=int, default=2000,)
    parser.add_argument("--max_steps", type=int, default=100_000, )
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,)
    
    args = parser.parse_args()
    
    train(args)