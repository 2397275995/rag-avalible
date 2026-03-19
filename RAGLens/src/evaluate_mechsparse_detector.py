import argparse
import json
from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from RAGLens import RAGLens


def load_jsonl(jsonl_path: str) -> List[dict]:
    rows: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate MechSparse detector (AUC/Acc/macro-F1).")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--detector_pkl", type=str, required=True)
    parser.add_argument("--dataset_jsonl", type=str, required=True, help="mechsparse_redeep_chunk_dataset.jsonl")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    sae = torch.load(args.sae_path, map_location=model.device)
    sae.eval()

    detector = RAGLens.load(args.detector_pkl, tokenizer=tokenizer, model=model, sae=sae)

    rows = load_jsonl(args.dataset_jsonl)
    rows = [r for r in rows if str(r.get("split", "unknown")) == args.split]
    if not rows:
        raise RuntimeError(f"No rows found for split={args.split} in {args.dataset_jsonl}")

    inputs: List[str] = [r["input"] for r in rows]
    outputs: List[str] = [r["output"] for r in rows]
    labels: np.ndarray = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    H: List[float] = [float(r["H"]) for r in rows]

    y_scores: List[float] = []
    for i in range(0, len(inputs), args.batch_size):
        j = min(i + args.batch_size, len(inputs))
        batch_inputs = inputs[i:j]
        batch_outputs = outputs[i:j]
        batch_H = H[i:j]
        with torch.inference_mode():
            probs = detector.predict_proba(batch_inputs, batch_outputs, H_values=batch_H)
        y_scores.extend(probs.tolist())

    y_scores_np = np.array(y_scores, dtype=np.float32)
    y_pred = (y_scores_np > args.threshold).astype(np.int64)

    if len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, y_scores_np))
    else:
        auc = float("nan")

    acc = float(accuracy_score(labels, y_pred))
    macro_f1 = float(f1_score(labels, y_pred, average="macro"))

    print(
        json.dumps(
            {
                "split": args.split,
                "n": int(len(labels)),
                "AUC": auc,
                "Acc": acc,
                "macroF1": macro_f1,
                "threshold": args.threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

