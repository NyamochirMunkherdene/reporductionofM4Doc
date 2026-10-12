# python learning_rate_un_exp.py \
#   --checkpoint_dir ./models/un_trans_jieba_lm/checkpoint-2000 \
#   --learning_rates 7e-4,1e-4,5e-4,1e-6,3e-6,1e-5,3e-5,5e-5 \
#   --max_steps 200 \
#   --eval_steps 50

import os
import json
import argparse
import torch
from datasets import load_dataset

from un_dataset import UNTranslationDataset


def run_experiment(
    checkpoint_dir,
    en_tokenizer_dir,
    zh_tokenizer_dir,
    learning_rates,
    output_dir,
    max_length=128,
    train_samples=2000,
    eval_samples=500,
    batch_size_per_gpu=4,
    max_steps=200,
    eval_steps=50,
):
    os.makedirs(output_dir, exist_ok=True)

    from transformers import AutoTokenizer, EncoderDecoderModel, TrainingArguments, Trainer

    print("Loading tokenizers...")
    en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer_dir)
    zh_tokenizer = AutoTokenizer.from_pretrained(zh_tokenizer_dir)

    print("Loading dataset (this may download if not present)...")
    raw = load_dataset("Helsinki-NLP/un_pc", "en-zh", split="train")

    # sample and split
    sample_count = min(len(raw), train_samples + eval_samples)
    raw = raw.shuffle(seed=42).select(range(sample_count))
    split = raw.train_test_split(test_size=eval_samples, seed=42)
    train_hf = split["train"]
    eval_hf = split["test"]

    train_dataset = UNTranslationDataset(train_hf, en_tokenizer, zh_tokenizer, max_length)
    eval_dataset = UNTranslationDataset(eval_hf, en_tokenizer, zh_tokenizer, max_length)

    results = []

    device_count = torch.cuda.device_count()
    print(f"Detected GPUs: {device_count}")

    for lr in learning_rates:
        print(f"\n=== LR={lr} experiment starting ===")
        model = EncoderDecoderModel.from_pretrained(checkpoint_dir)

        exp_out = os.path.join(output_dir, f"lr_{lr}")
        os.makedirs(exp_out, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=exp_out,
            per_device_train_batch_size=batch_size_per_gpu,
            per_device_eval_batch_size=batch_size_per_gpu,
            max_steps=max_steps,
            learning_rate=lr,
            warmup_ratio=0.05,
            logging_strategy="steps",
            logging_steps=eval_steps,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="no",
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        print("Evaluating before training...")
        try:
            eval_before = trainer.evaluate()
        except Exception as e:
            eval_before = {"error": str(e)}
        print("Starting short training run...")
        try:
            trainer.train()
            trainer.save_model(exp_out)
            eval_after = trainer.evaluate()
        except Exception as e:
            eval_after = {"error": str(e)}

        print(f"LR={lr} done. eval_before={eval_before}, eval_after={eval_after}")

        results.append({"learning_rate": lr, "eval_before": eval_before, "eval_after": eval_after, "output_dir": exp_out})

        # free memory
        del trainer
        del model
        torch.cuda.empty_cache()

    summary_path = os.path.join(output_dir, "lr_experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"All experiments complete. Summary saved to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./models/un_trans_jieba_lm/checkpoint-4000")
    parser.add_argument("--en_tokenizer_dir", type=str, default="./utils/en_tokenizer")
    parser.add_argument("--zh_tokenizer_dir", type=str, default="./utils/zh_tokenizer")
    parser.add_argument("--learning_rates", type=str, default="7e-4,1e-4,5e-4", help="Comma-separated list of learning rates")
    parser.add_argument("--output_dir", type=str, default="./lr_runs")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=2000)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=50)

    args = parser.parse_args()

    lrs = [float(x) for x in args.learning_rates.split(",") if x.strip()]

    # Resolve paths relative to repository root if not absolute
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(base, p.lstrip("./"))

    run_experiment(
        checkpoint_dir=resolve(args.checkpoint_dir),
        en_tokenizer_dir=resolve(args.en_tokenizer_dir),
        zh_tokenizer_dir=resolve(args.zh_tokenizer_dir),
        learning_rates=lrs,
        output_dir=resolve(args.output_dir),
        max_length=args.max_length,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        batch_size_per_gpu=args.batch_size_per_gpu,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
    )


# Table1: to see how platued checkpoint behave to different lr:
# 		expriment set up: max_steps=200, eval_steps=50
#  	LR		Eval before	Eval after	Change
#  	`1e-6`	7.22121		7.22022		-0.00098
#  	`3e-6`	7.22121		7.21990		-0.00131
#  	`1e-5`	7.22121		7.21376		-0.00745
#  	`3e-5`	7.22121		7.21308		-0.00813
#  	`5e-5`	7.22121		7.20554		-0.01566
#  	`1e-4`	7.22121		7.22297		+0.00177
#  	`5e-4`	7.22121		7.26157		+0.04036
#  	`7e-4`	7.22121		7.27355		+0.05234