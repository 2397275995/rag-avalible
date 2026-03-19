import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from RAGLens import RAGLens


def load_mechsparse_dataset(jsonl_path: str, limit: int | None = None):
    inputs, outputs, labels, H = [], [], [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_i, line in enumerate(f):
            if limit is not None and line_i >= limit:
                break
            obj = json.loads(line)
            inputs.append(obj["input"])
            outputs.append(obj["output"])
            labels.append(int(obj["label"]))
            H.append(float(obj["H"]))
    return inputs, outputs, labels, H


def main():
    parser = argparse.ArgumentParser(description="Train MechSparse detector (SAE + conditional MI + GAM/EBM)")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--sae_path", type=str, required=True, help="torch.load-able SAE checkpoint")
    parser.add_argument("--dataset_jsonl", type=str, required=True, help="Output of ReDeEP/mechsparse_extract.py")
    parser.add_argument("--hookpoint", type=str, default="model.layers.20", help="Kept for compatibility; not used in MechSparse encoding")
    parser.add_argument("--circuits_json", type=str, required=True, help="mechsparse_circuits.json from extractor")
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--out_pkl", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.circuits_json, "r", encoding="utf-8") as f:
        circuits = json.load(f)
    copy_heads = [tuple(x) for x in circuits["copy_heads"]]
    knowledge_layers = circuits["knowledge_layers"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    sae = torch.load(args.sae_path, map_location="cpu")

    inputs, outputs, labels, H = load_mechsparse_dataset(args.dataset_jsonl, limit=args.limit)

    detector = RAGLens(
        tokenizer=tokenizer,
        model=model,
        sae=sae,
        hookpoint=args.hookpoint,
        copy_heads=copy_heads,
        knowledge_layers=knowledge_layers,
        use_mechsparse=True,
        top_k=args.top_k,
    )

    detector.fit(
        inputs=inputs,
        outputs=outputs,
        labels=labels,
        H_values=H,
        conditional_mi=True,
        n_cond_bins=8,
    )

    detector.save(args.out_pkl)
    print(f"[MechSparse] Saved detector: {args.out_pkl}")


if __name__ == "__main__":
    main()

