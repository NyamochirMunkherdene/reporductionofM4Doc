import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, EncoderDecoderModel
import re


def read_text_file(path:Path)->str:
    """read an English MMD/text file."""
    return path.read_text(encoding="utf-8").strip()

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
    
    print("started")
    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device:{device}")
    print(f"loading model from:{args.model_dir}")

    en_tokenizer=AutoTokenizer.from_pretrained(args.en_tokenizer_dir)
    zh_tokenizer=AutoTokenizer.from_pretrained(args.zh_tokenizer_dir)
    #zh_tokenizer.backend_tokenizer.decoder = (decoders.WordPiece(prefix="##",cleanup=True,))

    model = EncoderDecoderModel.from_pretrained(args.model_dir)

    model.to(device)
    model.eval()

    with open(args.split_json_file_path, "r", encoding="utf-8") as file:
        split_data=json.load(file)
    #print(split_data.keys())
    #dict_keys(['num_train_samples', 'ßnum_valid_samples', 'num_test_samples', 'train_name_list', 'valid_name_list', 'test_name_list'])

    test_names=split_data[args.split_key]
    input_dir=Path(args.en_mmd_dir)
    output_dir=Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    print(f"Number of samples:{len(test_names)}")
    for index, sample_name in enumerate(test_names, start=1):
        input_path=input_dir/f"{sample_name}.mmd"

        
        if not input_path.exists():
            print(f"Missing {input_path}")
            continue

        source_text = read_text_file(input_path)
        
        if not source_text:
            print(f"[Empty] {input_path}")
            continue
        

        encoded =en_tokenizer(source_text, return_tensors="pt", truncation=True, max_length=args.max_length)
        #en_tokenizer returns {"input_ids":, "attention_mask":}

        input_ids=encoded["input_ids"].to(device)
        attention_mask=encoded["attention_mask"].to(device)

        with torch.no_grad():
            #**kwargs accpet any conditions which is not listed
            decoder_start_token_id = (model.config.decoder_start_token_id)
            if decoder_start_token_id is None:
                decoder_start_token_id = (zh_tokenizer.bos_token_id)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, #It is especially important when several samples of different lengths are put into one batch
                max_length=args.max_length,
                num_beams=args.num_beams,
                decoder_start_token_id = (decoder_start_token_id), 
                eos_token_id=(zh_tokenizer.eos_token_id),
                pad_token_id=(zh_tokenizer.pad_token_id), 
                no_repeat_ngram_size=3,
                )

        translated_text = decode_preserve_structure(
            zh_tokenizer,
            generated_ids[0]
        )
   
        output_path=output_dir/f"{sample_name}.mmd"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(translated_text, encoding="utf-8")

        if index % 10 == 0 or index ==len(test_names):
            print(f"Processed{index}/{len(test_names)}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Run text-only inference using model_trans")
    
    parser.add_argument("--model_dir", required=True, help="path to model_trans checkpoint.",)
    parser.add_argument("--en_tokenizer_dir", required=True, help="English tokenizer directory",)
    parser.add_argument("--zh_tokenizer_dir", required=True, help="Chinese tokenizer directory",)
    parser.add_argument("--en_mmd_dir", required=True, help="Directory containing English reference files.",)
    parser.add_argument("--split_json_file_path", required=True, help="Path to split_dataset.json.",)
    parser.add_argument("--output_dir", required=True, help="Directory for generated Chinese files.",)
    parser.add_argument("--split_key", default="test_name_list", help="Key in split JSON, such as test_name_list or valid_name_list.",)
    parser.add_argument("--max_length", type=int, default=1536,)
    parser.add_argument("--num_beams", type=int, default= 4)
    args=parser.parse_args()
    main(args)
    
    
    
    
    