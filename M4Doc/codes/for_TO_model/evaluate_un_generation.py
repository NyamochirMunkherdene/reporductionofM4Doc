import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_pairs(pred_dir: Path, ref_dir: Path) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []

    pred_files = sorted(pred_dir.glob("*.txt"))
    for pred_path in pred_files:
        stem = pred_path.stem
        ref_path = ref_dir / f"{stem}.txt"
        if not ref_path.exists():
            continue

        pred_text = normalize_text(pred_path.read_text(encoding="utf-8"))
        ref_text = normalize_text(ref_path.read_text(encoding="utf-8"))
        pairs.append((stem, pred_text, ref_text))

    if not pairs:
        raise FileNotFoundError(f"No matching prediction/reference pairs found in {pred_dir} and {ref_dir}")

    return pairs


def compute_bleu(pred_text: str, ref_text: str) -> float:
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except Exception as exc:
        raise RuntimeError("nltk is required for BLEU scoring") from exc

    hypothesis = [pred_text.split()]
    reference = [[ref_text.split()]]
    smoothie = SmoothingFunction().method1
    return corpus_bleu(reference, hypothesis, smoothing_function=smoothie)


def compute_bleu_st(pred_text: str, ref_text: str) -> float:
    try:
        from sacrebleu import corpus_bleu as sacre_corpus_bleu
    except Exception as exc:
        raise RuntimeError("sacrebleu is required for BLEU-st scoring") from exc

    return sacre_corpus_bleu([pred_text], [[ref_text]]).score / 100.0


def compute_steds(pred_text: str, ref_text: str) -> float:
    def levenshtein_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if len(a) == 0:
            return len(b)
        if len(b) == 0:
            return len(a)

        prev_row = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            row = [i]
            for j, cb in enumerate(b, start=1):
                insert_cost = row[j - 1] + 1
                delete_cost = prev_row[j] + 1
                replace_cost = prev_row[j - 1] + (ca != cb)
                row.append(min(insert_cost, delete_cost, replace_cost))
            prev_row = row

        return prev_row[-1]

    pred_tokens = pred_text.split()
    ref_tokens = ref_text.split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not ref_tokens:
        return 0.0
    distance = levenshtein_distance(" ".join(pred_tokens), " ".join(ref_tokens))
    max_len = max(len(pred_tokens), len(ref_tokens))
    return 1.0 - (distance / max_len) if max_len else 1.0


def main(args):
    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)
    output_path = Path(args.output_json)

    pairs = load_pairs(pred_dir, ref_dir)

    bleu_scores = []
    bleu_st_scores = []
    steds_scores = []

    for stem, pred_text, ref_text in pairs:
        bleu_scores.append(compute_bleu(pred_text, ref_text))
        bleu_st_scores.append(compute_bleu_st(pred_text, ref_text))
        steds_scores.append(compute_steds(pred_text, ref_text))

    result = {
        "num_pairs": len(pairs),
        "bleu": sum(bleu_scores) / len(bleu_scores),
        "bleu_st": sum(bleu_st_scores) / len(bleu_st_scores),
        "steds": sum(steds_scores) / len(steds_scores),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated translations against reference files")
    parser.add_argument("--pred_dir", required=True, help="Directory with generated .txt files")
    parser.add_argument("--ref_dir", required=True, help="Directory with reference .txt files")
    parser.add_argument("--output_json", default=None, help="Where to save the aggregate metrics")
    args = parser.parse_args()

    if args.output_json is None:
        args.output_json = os.path.join(args.pred_dir, "metrics.json")

    main(args)
