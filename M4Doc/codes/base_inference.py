#load Nougat encoder
#load translation decoder from your trans_model
#combine them into one trainable model

import os
import json
from PIL import Image
import argparse

import torch
from transformers import AutoTokenizer, DonutProcessor, VisionEncoderDecoderModel, EncoderDecoderModel, EncoderDecoderConfig, GenerationConfig

from base_model_M4Doc import BaseDIMTModelInference
def decode_preserve_structure(tokenizer, ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)

    special_tokens = {
        tokenizer.bos_token,
        tokenizer.eos_token,
        tokenizer.pad_token,
        tokenizer.cls_token,
        tokenizer.sep_token,
    }
    special_tokens.discard(None)

    pieces = []

    for tok in tokens:
        if tok in special_tokens:
            continue

        # Preserve structural newline token exactly
        if tok == "\n\n":
            pieces.append("\n\n")
            continue

        # Markdown heading markers
        if tok in ("#", "##", "###", "####", "#####", "######"):
            pieces.append(tok)
            continue

        # WordPiece continuation
        if tok.startswith("##"):
            pieces.append(tok[2:])
            continue

        pieces.append(tok)

    # Join intelligently
    text = ""
    for piece in pieces:
        if piece == "\n\n":
            text = text.rstrip() + "\n\n"
        elif piece.startswith("#") and (
            piece == "#" or set(piece) == {"#"}
        ):
            if text and not text.endswith("\n"):
                text += " "
            text += piece
        else:
            if text and not text.endswith((" ", "\n")):
                text += " "
            text += piece

    return text.strip()

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        data_type = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float32
        )
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        data_type = torch.float32
    else:
        device = torch.device("cpu")
        data_type = torch.float32

    MAX_LENGTH = 1536 #1536

    # load processor and tokenizer
    #process image to meet model standard
    #without processor< I would have to write all preprocessing myself.
    #resize image
    # convert image to tensor
    # normalize RGB values
    # add batch dimension

    nougat_processor = DonutProcessor.from_pretrained(os.path.join(args.base_dir, 'utils/donut-base'))
    nougat_processor.image_processor.size = {'height': 896, 'width': 672}
    nougat_processor.image_processor.image_mean = [0.485, 0.456, 0.406]
    nougat_processor.image_processor.image_std = [0.229, 0.224, 0.225]
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.base_dir, 'utils/zh_tokenizer'))

    # load model


    trans_model = EncoderDecoderModel.from_pretrained(args.trans_model_dir)
    trans_decoder = trans_model.decoder

    nougat_model = VisionEncoderDecoderModel.from_pretrained(args.nougat_model_dir, device_map='cpu')
    nougat_encoder = nougat_model.encoder



    config = EncoderDecoderConfig.from_pretrained(args.trans_model_dir)
    model = BaseDIMTModelInference(config, nougat_encoder, trans_decoder)


    state_dict_dir = os.path.join(args.checkpoint_dir, "state_dict")
    
    nougat_path = os.path.join(state_dict_dir, "nougat_encoder.bin")
    decoder_path = os.path.join(state_dict_dir, "trans_decoder.bin")
    projection_path = os.path.join(state_dict_dir, "nougat_down_proj.bin")

    required_paths = [
        nougat_path,
        decoder_path,
        projection_path,
    ]

    missing_paths = [
        path for path in required_paths
        if not os.path.isfile(path)
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing Base DIMT checkpoint files:\n"
            + "\n".join(missing_paths)
        )


        
    print("Loading trained Base DIMT checkpoint")

    model.nougat_encoder.load_state_dict(torch.load(nougat_path, map_location="cpu"))
    model.trans_decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    model.nougat_down_proj.load_state_dict(torch.load(projection_path, map_location="cpu"))
        
    model.to(device=device, dtype=data_type)
    model.eval()

    decoder_start_token_id = (
        trans_model.config.decoder_start_token_id
    )

    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.bos_token_id

    if decoder_start_token_id is None:
        raise ValueError("No decoder_start_token_id is configured.")

    model.config.decoder_start_token_id = decoder_start_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.trans_decoder.config.decoder_start_token_id = (
        decoder_start_token_id
    )
    model.trans_decoder.config.bos_token_id = tokenizer.bos_token_id
    model.trans_decoder.config.eos_token_id = tokenizer.eos_token_id
    model.trans_decoder.config.pad_token_id = tokenizer.pad_token_id
    
    generation_config = GenerationConfig(
        max_new_tokens=1516,
        num_beams=1,
        do_sample=False,
        use_cache=True,
        decoder_start_token_id=decoder_start_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.trans_decoder.config.use_cache = True


    image_dir = os.path.join(args.dataset_dir, 'imgs')

    json_file_path = os.path.join(
        args.dataset_dir,
        "split_dataset.json",
    )
    print(json_file_path)
    with open(json_file_path, 'r') as f:
        json_dict = json.load(f)

    test_name_list = json_dict['test_name_list']

    os.makedirs(args.output_dir, exist_ok=True)

    from tqdm import tqdm
    import time
    total_inference_time = 0.0

    # import transformers
    # print(transformers.__version__)
    #trnasformer must be 4.32.1

    for index, name in enumerate(tqdm(test_name_list)):
        image_file_path = os.path.join(image_dir, name+'.png')
        with Image.open(image_file_path) as image:
            image = image.convert("RGB")

            nougat_pixel_values = nougat_processor(
                image,
                return_tensors="pt",
            )["pixel_values"]
        nougat_pixel_values = nougat_pixel_values.to(
            device=device,
            dtype=data_type,
        )

        start_time = time.time()
        # Use a short fixed decoder prompt.


        with torch.no_grad():
            generation_outputs = model.generate(
                nougat_pixel_values=nougat_pixel_values,
                generation_config=generation_config,
            )

       
        inference_time =time.time() - start_time
        total_inference_time += inference_time

        if hasattr(generation_outputs, "sequences"):
            generated_ids = generation_outputs.sequences
        else:
            generated_ids = generation_outputs

        sequence = generation_outputs.sequences[0].detach().cpu()

        zh_text = decode_preserve_structure(
            tokenizer,
            sequence,
        )

        result_file_path = os.path.join(args.output_dir, name+'.mmd')
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write(zh_text)

        

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--trans_model_dir", type=str)
    parser.add_argument("--nougat_model_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    
    args = parser.parse_args()
    
    main(args)

