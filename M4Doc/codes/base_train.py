import argparse
import json
import os
from typing import List, Tuple

import torch
from transformers import (
    AutoTokenizer,
    DonutProcessor,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    Trainer,
    TrainingArguments,
    VisionEncoderDecoderModel,
)

from base_dataset import DoTADataset, DoTADatasetDataCollator
from base_model_M4Doc import BaseDIMTModelTrain


def load_split_file(dataset_dir: str) -> Tuple[List[str], List[str]]:
    split_path = os.path.join(dataset_dir, "split_dataset.json")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    try:
        return split_data["train_name_list"], split_data["valid_name_list"]
    except KeyError as error:
        raise KeyError(
            f"Missing required key {error} in {split_path}"
        ) from error


def validate_dataset_files(
    names: List[str], image_dir: str, target_dir: str, split_name: str
) -> None:
    missing = []
    for name in names:
        for path in (
            os.path.join(image_dir, name + ".png"),
            os.path.join(target_dir, name + ".mmd"),
        ):
            if not os.path.isfile(path):
                missing.append(path)
                if len(missing) == 10:
                    break
        if len(missing) == 10:
            break
    if missing:
        raise FileNotFoundError(
            f"{split_name} split has missing files (showing up to 10):\n"
            + "\n".join(missing)
        )


def load_component_checkpoint(
    model: BaseDIMTModelTrain, checkpoint_dir: str
) -> None:
    state_dict_dir = os.path.join(checkpoint_dir, "state_dict")
    components = {
        "nougat_encoder": os.path.join(state_dict_dir, "nougat_encoder.bin"),
        "trans_decoder": os.path.join(state_dict_dir, "trans_decoder.bin"),
        "nougat_down_proj": os.path.join(state_dict_dir, "nougat_down_proj.bin"),
    }
    missing = [path for path in components.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "The component checkpoint is incomplete. Missing:\n"
            + "\n".join(missing)
        )

    print(f"Loading Base DIMT component weights from: {checkpoint_dir}")
    for component_name, path in components.items():
        component = getattr(model, component_name)
        component.load_state_dict(torch.load(path, map_location="cpu"))

    


def train(args: argparse.Namespace) -> None:

    required_directories = {
        "base_dir": args.base_dir,
        "trans_model_dir": args.trans_model_dir,
        "nougat_model_dir": args.nougat_model_dir,
        "dataset_dir": args.dataset_dir,
    }

    for path_name, path_value in required_directories.items():
        if not os.path.isdir(path_value):
            raise FileNotFoundError(f"{path_name} does not exist: {path_value}")
    os.makedirs(args.output_dir, exist_ok=True)

    report_to = [] if args.disable_wandb else ["wandb"]
    if not args.disable_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_LOG_MODEL"] = "false"


    #load processor
    donut_processor_dir = os.path.join(args.base_dir, "utils", "donut-base")
    tokenizer_dir = os.path.join(args.base_dir, "utils", "zh_tokenizer")
    nougat_processor = DonutProcessor.from_pretrained(donut_processor_dir)
    nougat_processor.image_processor.size = {"height": 896, "width": 672}
    nougat_processor.image_processor.image_mean = [0.485, 0.456, 0.406]
    nougat_processor.image_processor.image_std = [0.229, 0.224, 0.225]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    trans_model = EncoderDecoderModel.from_pretrained(args.trans_model_dir)
    trans_decoder = trans_model.decoder
    trans_decoder.config.use_cache = False
    del trans_model.encoder

    nougat_model = VisionEncoderDecoderModel.from_pretrained(args.nougat_model_dir)
    nougat_encoder = nougat_model.encoder
    del nougat_model.decoder

    if args.max_length > trans_decoder.config.max_position_embeddings:
        raise ValueError(
            f"--max_length {args.max_length} exceeds the translation decoder's "
            f"max_position_embeddings {trans_decoder.config.max_position_embeddings}."
        )

    #only reads modelstrans_model/config.⇨architecture/settings
    # it is needed to load a trans_model weights
    config = EncoderDecoderConfig.from_pretrained(args.trans_model_dir)
    # only init will run and inside the model we hace, loss function, projection layer, and translation decoder, and nougat endorer
    model = BaseDIMTModelTrain(config, nougat_encoder, trans_decoder)
    if args.resume_model_checkpoint:
        load_component_checkpoint(model, args.resume_model_checkpoint)


    model.eval()

    if args.freeze_nougat:
        for parameter in model.nougat_encoder.parameters():
            parameter.requires_grad = False
    if args.freeze_decoder:
        for parameter in model.trans_decoder.parameters():
            parameter.requires_grad = False

    train_names, valid_names = load_split_file(args.dataset_dir)
    if args.train_limit is not None:
        train_names = train_names[: args.train_limit]
    if args.valid_limit is not None:
        valid_names = valid_names[: args.valid_limit]
    if not train_names:
        raise ValueError("The training split is empty.")
    if not valid_names:
        raise ValueError("The validation split is empty.")

    image_dir = os.path.join(args.dataset_dir, "imgs")
    target_dir = os.path.join(args.dataset_dir, "zh_mmd")

    if not os.path.isdir(image_dir) or not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"Expected dataset directories: {image_dir} and {target_dir}"
        )

    
    validate_dataset_files(train_names, image_dir, target_dir, "training")
    validate_dataset_files(valid_names, image_dir, target_dir, "validation")

    dataset_kwargs = {
        "nougat_processor": nougat_processor,
        "tokenizer": tokenizer,
        "image_dir": image_dir,
        "target_dir": target_dir,
        "max_length": args.max_length,
    }
    train_dataset = DoTADataset(name_list=train_names, **dataset_kwargs)
    valid_dataset = DoTADataset(name_list=valid_names, **dataset_kwargs)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=args.wandb_run_name,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DoTADatasetDataCollator(),
    )

    train_result = trainer.train(
        resume_from_checkpoint="/home/pj24001974/ku60000761/project/test_project/M4Doc/models/Base_model/100k/checkpoint-47750"
        )
        
    evaluation_metrics = trainer.evaluate()


    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", evaluation_metrics)
    trainer.save_state()

    
    print(f"Training completed. Final model saved to: {final_model_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Nougat encoder with the pretrained translation decoder."
    )
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--trans_model_dir", required=True)
    parser.add_argument("--nougat_model_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_model_checkpoint")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)        
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100) #change everytime #for 30000 it is 300
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--freeze_nougat", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_limit", type=int)
    parser.add_argument("--valid_limit", type=int)
    parser.add_argument("--wandb_project", default="M4Doc-Base-DIMT")
    parser.add_argument("--wandb_run_name", default="base-dimt-training")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
