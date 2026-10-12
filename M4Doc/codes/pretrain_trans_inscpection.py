import os
import json
import torch
import jieba
import re
import argparse 
def inspect_dataset_sample(
    dataset,
    en_tokenizer,
    zh_tokenizer,
    sample_index=0,
    dataset_name="train",
):
    print(f"\n===== Inspecting {dataset_name} sample {sample_index} =====")

    sample = dataset[sample_index]

    print("Sample type:", type(sample))

    if isinstance(sample, tuple):
        print("Tuple length:", len(sample))
        print("Tuple element types:", [type(x) for x in sample])
        sample = sample[0]

    print("Sample keys:", sample.keys())

    input_ids = sample["input_ids"]
    labels = sample["labels"]

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    print("input_ids shape:", input_ids.shape)
    print("labels shape:", labels.shape)

    valid_labels = labels[labels != -100]

    print("Valid label count:", valid_labels.numel())
    print("Ignored label count:", (labels == -100).sum().item())

    token_64_count = (valid_labels == 64).sum().item()

    print("Token 64 count:", token_64_count)

    if valid_labels.numel() > 0:
        print(
            "Token 64 ratio:",
            token_64_count / valid_labels.numel(),
        )

    print("\nDecoded English input:")
    print(
        en_tokenizer.decode(
            input_ids,
            skip_special_tokens=False,
        )
    )

    print("\nFirst 100 valid label IDs:")
    print(valid_labels[:100].tolist())

    print("\nFirst 100 target tokens:")
    print(
        zh_tokenizer.convert_ids_to_tokens(
            valid_labels[:100].tolist()
        )
    )

    print("\nDecoded Chinese target:")
    print(
        zh_tokenizer.decode(
            valid_labels,
            skip_special_tokens=False,
        )
    )

    print("===== End inspection =====\n")

from collections import Counter


def inspect_target_token_frequencies(
    dataset,
    zh_tokenizer,
    sample_count=100,
):
    counter = Counter()
    total_valid_tokens = 0

    sample_count = min(sample_count, len(dataset))

    for index in range(sample_count):
        sample = dataset[index]
        labels = torch.as_tensor(sample["labels"])

        valid_labels = labels[labels != -100]

        counter.update(valid_labels.tolist())
        total_valid_tokens += valid_labels.numel()

    print(
        f"\n===== Target frequencies over "
        f"{sample_count} samples ====="
    )
    print("Total valid target tokens:", total_valid_tokens)

    for token_id, count in counter.most_common(30):
        token = zh_tokenizer.convert_ids_to_tokens(token_id)
        ratio = count / total_valid_tokens

        print(
            f"id={token_id:<6} "
            f"count={count:<8} "
            f"ratio={ratio:.6f} "
            f"token={repr(token)}"
        )

    print("===== End frequencies =====\n")


def train(args):
    os.environ.setdefault(
        "WANDB_PROJECT",
        "M4Doc-Translation-Sandbox",
    )

    MAX_LENGTH = args.max_length

    
    from transformers import AutoTokenizer
    en_tokenizer = AutoTokenizer.from_pretrained(args.en_tokenizer_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(args.zh_tokenizer_dir)

    with open(args.split_json_file_path, 'r') as f:
        json_dict = json.load(f)
    train_name_list = json_dict["train_name_list"]
    valid_name_list = json_dict["valid_name_list"]
    

    from my_dataset_trans import DoTADatasetTrans
    valid_dataset = DoTADatasetTrans(en_tokenizer, zh_tokenizer, args.en_mmd_dir, args.zh_mmd_dir, valid_name_list, MAX_LENGTH)
    train_dataset = DoTADatasetTrans(en_tokenizer, zh_tokenizer, args.en_mmd_dir, args.zh_mmd_dir, train_name_list, MAX_LENGTH)
    # Print dataset inspection only from GPU/process 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if local_rank == 0:
        inspect_dataset_sample(
            train_dataset,
            en_tokenizer,
            zh_tokenizer,
            sample_index=0,
            dataset_name="train",
        )

        inspect_dataset_sample(
            valid_dataset,
            en_tokenizer,
            zh_tokenizer,
            sample_index=0,
            dataset_name="validation",
        )

        inspect_target_token_frequencies(
            train_dataset,
            zh_tokenizer,
            sample_count=100,
        )

    print(f"Number of training samples: {len(train_dataset):,}")
    print(f"Number of validation samples: {len(valid_dataset):,}")
    
    #for calculation
    print(f"Number of training samples: {len(train_dataset):,}")
    print(f"Number of validation samples: {len(valid_dataset):,}")

    from transformers import EncoderDecoderModel, EncoderDecoderConfig, BertConfig

    encoder_config = BertConfig()
    decoder_config = BertConfig()
    encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    
    # Modify config according to transformer-base
    encoder_decoder_config.encoder.bos_token_id = en_tokenizer.bos_token_id
    encoder_decoder_config.encoder.eos_token_id = en_tokenizer.eos_token_id
    encoder_decoder_config.encoder.hidden_size = 512
    encoder_decoder_config.encoder.intermediate_size = 4096
    encoder_decoder_config.encoder.max_length = MAX_LENGTH
    encoder_decoder_config.encoder.max_position_embeddings = MAX_LENGTH
    encoder_decoder_config.encoder.num_attention_heads = 16
    encoder_decoder_config.encoder.num_hidden_layers = 24
    encoder_decoder_config.encoder.pad_token_id = en_tokenizer.pad_token_id
    encoder_decoder_config.encoder.type_vocab_size = 1
    encoder_decoder_config.encoder.vocab_size = len(en_tokenizer)

    encoder_decoder_config.decoder.bos_token_id = zh_tokenizer.bos_token_id
    encoder_decoder_config.decoder.decoder_start_token_id = zh_tokenizer.bos_token_id
    encoder_decoder_config.decoder.eos_token_id = zh_tokenizer.eos_token_id
    encoder_decoder_config.decoder.hidden_size = 512
    encoder_decoder_config.decoder.intermediate_size = 4096
    encoder_decoder_config.decoder.max_length = MAX_LENGTH
    encoder_decoder_config.decoder.max_position_embeddings = MAX_LENGTH
    encoder_decoder_config.decoder.num_attention_heads = 16
    encoder_decoder_config.decoder.num_hidden_layers = 24
    encoder_decoder_config.decoder.pad_token_id = zh_tokenizer.pad_token_id
    encoder_decoder_config.decoder.type_vocab_size = 1
    encoder_decoder_config.decoder.vocab_size = len(zh_tokenizer)

    encoder_decoder_config.decoder_start_token_id = zh_tokenizer.bos_token_id
    encoder_decoder_config.pad_token_id = zh_tokenizer.pad_token_id
    encoder_decoder_config.eos_token_id = zh_tokenizer.eos_token_id
    encoder_decoder_config.max_length = MAX_LENGTH
    encoder_decoder_config.early_stopping = True
    encoder_decoder_config.no_repeat_ngram_size = 3
    encoder_decoder_config.length_penalty = 1.0
    encoder_decoder_config.num_beams = 4
    encoder_decoder_config.vocab_size = len(zh_tokenizer)
    
    model = EncoderDecoderModel(config=encoder_decoder_config)

    num_gpu = torch.cuda.device_count()
    gradient_accumulation_steps = args.batch_size // (num_gpu * args.batch_size_per_gpu)
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name="transformer-base",
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_strategy='steps',
        logging_steps=1,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        bf16=args.bf16,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
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
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--deepspeed", type=str)
    
    args = parser.parse_args()
    
    train(args)