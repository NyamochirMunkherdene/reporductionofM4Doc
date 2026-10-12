import argparse
import json
import os
from types import MethodType
from typing import Dict

import torch
from transformers import (
    AutoTokenizer,
    DonutProcessor,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    GenerationConfig,
    VisionEncoderDecoderModel,
)

from base_dataset import DoTADataset
from base_model_M4Doc import BaseDIMTModelInference
def fixed_prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    **kwargs,
):
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]

    if attention_mask is None:
        attention_mask = torch.ones(
            input_ids.shape,
            dtype=torch.long,
            device=input_ids.device,
        )

    prepared_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache", True),
    }

    if kwargs.get("encoder_hidden_states") is not None:
        prepared_inputs["encoder_hidden_states"] = (
            kwargs["encoder_hidden_states"]
        )

    if kwargs.get("encoder_attention_mask") is not None:
        prepared_inputs["encoder_attention_mask"] = (
            kwargs["encoder_attention_mask"]
        )

    return prepared_inputs

def install_generation_debug_hook(model):
    original_prepare = (
        model.trans_decoder.prepare_inputs_for_generation
    )

    def debug_prepare_inputs_for_generation(
        decoder_self,
        input_ids,
        **kwargs,
    ):
        prepared = original_prepare(
            input_ids=input_ids,
            **kwargs,
        )

        if not hasattr(
            decoder_self,
            "_printed_prepare_debug",
        ):
            decoder_self._printed_prepare_debug = True

            print(
                "\n--- prepare_inputs_for_generation debug ---"
            )
            print(
                "Incoming kwargs:",
                sorted(kwargs.keys()),
            )
            print(
                "Prepared keys:",
                sorted(prepared.keys()),
            )
            print(
                "Incoming has encoder states:",
                "encoder_hidden_states" in kwargs,
            )
            print(
                "Prepared has encoder states:",
                "encoder_hidden_states" in prepared,
            )

            if "encoder_hidden_states" in kwargs:
                print(
                    "Incoming encoder shape:",
                    kwargs[
                        "encoder_hidden_states"
                    ].shape,
                )

            if "encoder_hidden_states" in prepared:
                print(
                    "Prepared encoder shape:",
                    prepared[
                        "encoder_hidden_states"
                    ].shape,
                )

        return prepared

    model.trans_decoder.prepare_inputs_for_generation = (
        MethodType(
            debug_prepare_inputs_for_generation,
            model.trans_decoder,
        )
    )




def get_first_step_logits(
    model,
    pixel_values,
    tokenizer,
):
    with torch.no_grad():
        encoder_outputs = model.nougat_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )

        encoder_hidden_states = model.nougat_down_proj(
            encoder_outputs.last_hidden_state
        )

        start_ids = torch.tensor(
            [[tokenizer.bos_token_id]],
            device=pixel_values.device,
            dtype=torch.long,
        )

        decoder_outputs = model.trans_decoder(
            input_ids=start_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=True,
        )

    return (
        decoder_outputs.logits[:, -1, :].float(),
        encoder_hidden_states.float(),
    )



def load_checkpoint(
    model: BaseDIMTModelInference,
    checkpoint_dir: str,
) -> None:
    state_dict_dir = os.path.join(
        checkpoint_dir,
        "state_dict",
    )

    checkpoint_paths = {
        "nougat_encoder": os.path.join(
            state_dict_dir,
            "nougat_encoder.bin",
        ),
        "trans_decoder": os.path.join(
            state_dict_dir,
            "trans_decoder.bin",
        ),
        "nougat_down_proj": os.path.join(
            state_dict_dir,
            "nougat_down_proj.bin",
        ),
    }
    
    missing_paths = [
        path
        for path in checkpoint_paths.values()
        if not os.path.isfile(path)
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing Base DIMT checkpoint files:\n"
            + "\n".join(missing_paths)
        )

    print("Loading trained Base DIMT checkpoint:")
    print(checkpoint_dir)

    model.nougat_encoder.load_state_dict(
        torch.load(
            checkpoint_paths["nougat_encoder"],
            map_location="cpu",
        ),
        strict=True,
    )

    model.trans_decoder.load_state_dict(
        torch.load(
            checkpoint_paths["trans_decoder"],
            map_location="cpu",
        ),
        strict=True,
    )

    model.nougat_down_proj.load_state_dict(
        torch.load(
            checkpoint_paths["nougat_down_proj"],
            map_location="cpu",
        ),
        strict=True,
    )

def prepare_teacher_forced_batch(
    sample: Dict[str, torch.Tensor],
    device: torch.device,
    data_type: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """
    Add a batch dimension and move tensors to the correct device and dtype.
    """

    return {
        "nougat_pixel_values": (
            sample["nougat_pixel_values"]
            .unsqueeze(0)
            .to(
                device=device,
                dtype=data_type,
            )
        ),
        "decoder_input_ids": (
            sample["decoder_input_ids"]
            .unsqueeze(0)
            .to(
                device=device,
                dtype=torch.long,
            )
        ),
        "decoder_attention_mask": (
            sample["decoder_attention_mask"]
            .unsqueeze(0)
            .to(
                device=device,
                dtype=torch.long,
            )
        ),
        "labels": (
            sample["labels"]
            .unsqueeze(0)
            .to(
                device=device,
                dtype=torch.long,
            )
        ),
    }


def run_teacher_forced_test(
    model: BaseDIMTModelInference,
    sample: Dict[str, torch.Tensor],
    tokenizer,
    device: torch.device,
    data_type: torch.dtype,
) -> None:
    """
    Evaluate the checkpoint using the real preceding target tokens.

    This bypasses generate() and tests whether the trained checkpoint
    learned correct next-token prediction under teacher forcing.
    """

    batch = prepare_teacher_forced_batch(
        sample=sample,
        device=device,
        data_type=data_type,
    )

    with torch.no_grad():
        encoder_outputs = model.nougat_encoder(
            pixel_values=batch["nougat_pixel_values"],
            return_dict=True,
        )

        encoder_hidden_states = model.nougat_down_proj(
            encoder_outputs.last_hidden_state
        )

        decoder_outputs = model.trans_decoder(
            input_ids=batch["decoder_input_ids"],
            attention_mask=batch["decoder_attention_mask"],
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=True,
        )

    logits = decoder_outputs.logits

    predicted_ids = (
        logits.argmax(dim=-1)[0]
        .detach()
        .cpu()
    )

    decoder_ids = (
        batch["decoder_input_ids"][0]
        .detach()
        .cpu()
    )

    labels = (
        batch["labels"][0]
        .detach()
        .cpu()
    )

    valid_mask = labels.ne(-100)

    attention_mask = (
        batch["decoder_attention_mask"][0]
        .detach()
        .cpu()
        .bool()
    )

    same_position_accuracy = (
        predicted_ids[attention_mask]
        .eq(decoder_ids[attention_mask])
        .float()
        .mean()
        .item()
    )

    if valid_mask.any():
        accuracy = (
            predicted_ids[valid_mask]
            .eq(labels[valid_mask])
            .float()
            .mean()
            .item()
        )

        loss = torch.nn.functional.cross_entropy(
            logits[0][valid_mask.to(logits.device)],
            labels[valid_mask].to(logits.device),
        ).item()
    else:
        accuracy = float("nan")
        loss = float("nan")

    print(
        "Same-position reconstruction accuracy:",
        f"{same_position_accuracy:.6f}",
    )

    valid_labels = labels[valid_mask]

    print("\n--- Teacher-forced evaluation ---")
    print(f"Teacher-forced loss: {loss:.6f}")
    print(
        f"Teacher-forced token accuracy: "
        f"{accuracy:.6f}"
    )

    first_valid_index = (
        int(valid_mask.nonzero(as_tuple=False)[0].item())
        if valid_mask.any()
        else None
    )

    if first_valid_index is not None:
        first_input_id = decoder_ids[first_valid_index].item()
        first_prediction_id = predicted_ids[
            first_valid_index
        ].item()
        first_label_id = labels[first_valid_index].item()

        print("\nFirst teacher-forced position:")
        print(
            "Input:",
            first_input_id,
            repr(
                tokenizer.convert_ids_to_tokens(
                    first_input_id
                )
            ),
        )
        print(
            "Prediction:",
            first_prediction_id,
            repr(
                tokenizer.convert_ids_to_tokens(
                    first_prediction_id
                )
            ),
        )
        print(
            "Label:",
            first_label_id,
            repr(
                tokenizer.convert_ids_to_tokens(
                    first_label_id
                )
            ),
        )

    print(
        "\nDecoder inputs:",
        tokenizer.decode(
            decoder_ids[:150],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    )

    print(
        "\nPredictions:",
        tokenizer.decode(
            predicted_ids[:150],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    )

    print(
        "\nLabels:",
        tokenizer.decode(
            valid_labels[:150],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    )

    print(
        "\nFirst 50 decoder IDs:",
        decoder_ids[:50].tolist(),
    )

    print(
        "First 50 predicted IDs:",
        predicted_ids[:50].tolist(),
    )

    print(
        "First 50 label IDs:",
        valid_labels[:50].tolist(),
    )


def print_first_step_predictions(
    generation_outputs,
    tokenizer,
    top_k: int = 10,
) -> None:
    """
    Print the first generated-token distribution and BOS logit margin.
    """

    scores = getattr(
        generation_outputs,
        "scores",
        None,
    )

    if not scores:
        print("No generation scores were returned.")
        return

    first_logits = scores[0][0].float()

    bos_id = tokenizer.bos_token_id

    if bos_id is None:
        raise ValueError(
            "The tokenizer does not define bos_token_id."
        )

    bos_logit = first_logits[bos_id]

    other_logits = first_logits.clone()
    other_logits[bos_id] = float("-inf")

    best_other_logit, best_other_id = (
        other_logits.max(dim=0)
    )

    logit_margin = bos_logit - best_other_logit

    probabilities = torch.softmax(
        first_logits,
        dim=-1,
    )

    values, indices = torch.topk(
        probabilities,
        k=top_k,
    )

    print("\n--- First-token logit margin ---")
    print("BOS logit:", bos_logit.item())
    print(
        "Best other logit:",
        best_other_logit.item(),
    )
    print(
        "BOS logit margin:",
        logit_margin.item(),
    )
    print(
        "Best other token:",
        best_other_id.item(),
        repr(
            tokenizer.convert_ids_to_tokens(
                best_other_id.item()
            )
        ),
    )

    print("\nFirst generated-token probabilities:")

    for probability, token_id in zip(
        values.tolist(),
        indices.tolist(),
    ):
        token = tokenizer.convert_ids_to_tokens(
            token_id
        )

        print(
            f"id={token_id:<7} "
            f"probability={probability:.10f} "
            f"token={repr(token)}"
        )


def main(args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Use FP32 for this diagnostic.
    data_type = torch.float32

    print("Device:", device)
    print("Model dtype:", data_type)

    donut_processor_dir = os.path.join(
        args.base_dir,
        "utils",
        "donut-base",
    )

    tokenizer_dir = os.path.join(
        args.base_dir,
        "utils",
        "zh_tokenizer",
    )

    nougat_processor = DonutProcessor.from_pretrained(
        donut_processor_dir
    )

    nougat_processor.image_processor.size = {
        "height": 896,
        "width": 672,
    }

    nougat_processor.image_processor.image_mean = [
        0.485,
        0.456,
        0.406,
    ]

    nougat_processor.image_processor.image_std = [
        0.229,
        0.224,
        0.225,
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir
    )

    trans_model = EncoderDecoderModel.from_pretrained(
        args.trans_model_dir
    )

    trans_decoder = trans_model.decoder

    nougat_model = (
        VisionEncoderDecoderModel.from_pretrained(
            args.nougat_model_dir,
            device_map="cpu",
        )
    )

    nougat_encoder = nougat_model.encoder

    config = EncoderDecoderConfig.from_pretrained(
        args.trans_model_dir
    )

    model = BaseDIMTModelInference(
        config,
        nougat_encoder,
        trans_decoder,
    )

    load_checkpoint(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
    )

    decoder_start_token_id = (
        trans_model.config.decoder_start_token_id
    )

    if decoder_start_token_id is None:
        decoder_start_token_id = (
            tokenizer.bos_token_id
        )

    if decoder_start_token_id is None:
        raise ValueError(
            "No decoder start token is configured."
        )

    model.config.decoder_start_token_id = (
        decoder_start_token_id
    )
    model.config.bos_token_id = (
        tokenizer.bos_token_id
    )
    model.config.eos_token_id = (
        tokenizer.eos_token_id
    )
    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    model.trans_decoder.config.decoder_start_token_id = (
        decoder_start_token_id
    )
    model.trans_decoder.config.bos_token_id = (
        tokenizer.bos_token_id
    )
    model.trans_decoder.config.eos_token_id = (
        tokenizer.eos_token_id
    )
    model.trans_decoder.config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.trans_decoder.config.use_cache = (
        args.use_cache
    )

    
    model.to(
        device=device,
        dtype=data_type,
    )
    model.eval()

    if args.patch_generation:
        model.trans_decoder.prepare_inputs_for_generation = MethodType(
            fixed_prepare_inputs_for_generation,
            model.trans_decoder,
        )
        print("Installed fixed prepare_inputs_for_generation.")
    else:
        print("Using original prepare_inputs_for_generation.")

    if args.debug_generation:
        install_generation_debug_hook(model)
        print("Installed generation debug hook.")

    print("Tokenizer vocabulary:", len(tokenizer))
    print(
        "Decoder vocabulary:",
        model.trans_decoder.config.vocab_size,
    )

    print(
        "Decoder start token:",
        decoder_start_token_id,
        tokenizer.convert_ids_to_tokens(
            decoder_start_token_id
        ),
    )

    print(
        "BOS:",
        tokenizer.bos_token_id,
        tokenizer.bos_token,
    )

    print(
        "EOS:",
        tokenizer.eos_token_id,
        tokenizer.eos_token,
    )

    print(
        "PAD:",
        tokenizer.pad_token_id,
        tokenizer.pad_token,
    )

    split_path = os.path.join(
        args.dataset_dir,
        "split_dataset.json",
    )

    with open(
        split_path,
        "r",
        encoding="utf-8",
    ) as file:
        split_data = json.load(file)

    if args.split_name == "train":
        sample_names = split_data[
            "train_name_list"
        ][: args.num_samples]
    elif args.split_name == "valid":
        sample_names = split_data[
            "valid_name_list"
        ][: args.num_samples]
    elif args.split_name == "test":
        sample_names = split_data[
            "test_name_list"
        ][: args.num_samples]
    else:
        raise ValueError(
            f"Unsupported split: {args.split_name}"
        )

    if not sample_names:
        raise ValueError(
            f"No samples found in split: "
            f"{args.split_name}"
        )

    print(
        f"Evaluating split: {args.split_name}"
    )
    print(
        f"Number of samples: {len(sample_names)}"
    )

    image_dir = os.path.join(
        args.dataset_dir,
        "imgs",
    )

    target_dir = os.path.join(
        args.dataset_dir,
        "zh_mmd",
    )

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    if not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"Target directory not found: {target_dir}"
        )

    dataset = DoTADataset(
        nougat_processor=nougat_processor,
        tokenizer=tokenizer,
        image_dir=image_dir,
        target_dir=target_dir,
        name_list=sample_names,
        max_length=args.max_length,
    )

    if len(dataset) >= 2:
        sample_a = dataset[0]
        sample_b = dataset[1]

        pixel_values_a = (
            sample_a["nougat_pixel_values"]
            .unsqueeze(0)
            .to(
                device=device,
                dtype=data_type,
            )
        )

        pixel_values_b = (
            sample_b["nougat_pixel_values"]
            .unsqueeze(0)
            .to(
                device=device,
                dtype=data_type,
            )
        )

        logits_a, hidden_a = get_first_step_logits(
            model,
            pixel_values_a,
            tokenizer,
        )

        logits_b, hidden_b = get_first_step_logits(
            model,
            pixel_values_b,
            tokenizer,
        )

        print("\n--- Two-image conditioning comparison ---")

        print(
            "Encoder feature difference:",
            (hidden_a - hidden_b)
            .abs()
            .mean()
            .item(),
        )

        print(
            "First-token logit difference:",
            (logits_a - logits_b)
            .abs()
            .mean()
            .item(),
        )

        print(
            "BOS logit A:",
            logits_a[
                0,
                tokenizer.bos_token_id,
            ].item(),
        )

        print(
            "BOS logit B:",
            logits_b[
                0,
                tokenizer.bos_token_id,
            ].item(),
        )

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=False,
        use_cache=args.use_cache,
        early_stopping=(
            args.num_beams > 1
        ),
        decoder_start_token_id=(
            decoder_start_token_id
        ),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    for index, name in enumerate(sample_names):
        print("\n" + "=" * 80)
        print(
            f"Sample {index + 1}/"
            f"{len(sample_names)}: {name}"
        )
        print("=" * 80)

        sample = dataset[index]

        pixel_values = (
            sample["nougat_pixel_values"]
            .unsqueeze(0)
            .to(
                device=device,
                dtype=data_type,
            )
        )

        with torch.no_grad():
            generation_outputs = model.generate(
                nougat_pixel_values=pixel_values,
                generation_config=generation_config,
            )

        sequence = (
            generation_outputs.sequences[0]
            .detach()
            .cpu()
        )

        print("\n--- Autoregressive generation ---")
        print(
            "Generated IDs:",
            sequence.tolist(),
        )

        raw_text = tokenizer.decode(
            sequence,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        clean_text = tokenizer.decode(
            sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        print(
            "Decoded with special tokens:",
            raw_text,
        )

        print(
            "Decoded without special tokens:",
            clean_text,
        )

        print_first_step_predictions(
            generation_outputs=generation_outputs,
            tokenizer=tokenizer,
            top_k=10,
        )

        run_teacher_forced_test(
            model=model,
            sample=sample,
            tokenizer=tokenizer,
            device=device,
            data_type=data_type,
        )

        result_path = os.path.join(
            args.output_dir,
            name + ".mmd",
        )

        with open(
            result_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(clean_text)

        print(
            "\nSaved generation to:",
            result_path,
        )


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_dir",
        required=True,
    )

    parser.add_argument(
        "--base_dir",
        required=True,
    )

    parser.add_argument(
        "--trans_model_dir",
        required=True,
    )

    parser.add_argument(
        "--nougat_model_dir",
        required=True,
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=1536,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--split_name",
        choices=[
            "train",
            "valid",
            "test",
        ],
        default="train",
    )

    parser.add_argument(
        "--use_cache",
        action="store_true",
    )

    parser.add_argument(
        "--patch_generation",
        action="store_true",
    )
    parser.add_argument(
        "--debug_generation",
        action="store_true",
    )
    



    main(parser.parse_args())


##seems difference model.trans_decoder(
#     input_ids=start_ids,
#     encoder_hidden_states=encoder_hidden_states,
# )

#model.trans_decoder.generate(...)


#The most likely difference is that the original prepare_inputs_for_generation() is not preserving encoder_hidden_states.

#next step to check
#Check what generate() actually passes

#Wrap the original method without replacing its behavior:
