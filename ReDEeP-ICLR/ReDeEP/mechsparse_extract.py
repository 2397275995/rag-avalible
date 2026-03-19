import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class MechSparseCircuits:
    copy_heads: List[Tuple[int, int]]
    knowledge_layers: List[int]
    ext_map: Dict[str, Tuple[int, int]]
    para_map: Dict[str, int]


def _load_source_info(source_info_path: str) -> Dict[str, dict]:
    source_info_dict = {}
    with open(source_info_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            source_info_dict[data["source_id"]] = data
    return source_info_dict


def _construct_chunk_df(
    response_chunk_json: str,
    source_info_dict: Dict[str, dict],
    top_n: int,
    splits: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]], Dict[str, int]]:
    """
    Read ReDeEP chunk-level JSON produced by `chunk_level_detect.py` (scores per span),
    and return a flat dataframe with exactly `top_n` external features and `top_n` parameter features.

    It also returns:
      - ext_map: column_name -> (layer, head)
      - para_map: column_name -> layer_id  (parsed from 'layer_{i}')
    """
    with open(response_chunk_json, "r", encoding="utf-8") as f:
        response = json.load(f)

    data_dict = {
        "response_id": [],
        "span_id": [],
        "task_type": [],
        "input": [],
        "output": [],
        **{f"ext_{k}": [] for k in range(top_n)},
        **{f"par_{k}": [] for k in range(top_n)},
        "label": [],
    }

    ext_map: Dict[str, Tuple[int, int]] = {}
    para_map: Dict[str, int] = {}

    splits_set = set(splits) if splits is not None else None
    for i, resp in enumerate(response):
        if splits_set is not None and resp.get("split") not in splits_set:
            continue
        source_id = resp["source_id"]
        task_type = source_info_dict.get(source_id, {}).get("task_type", "unknown")
        prompt_text = source_info_dict.get(source_id, {}).get("prompt", "")
        response_text = resp.get("response", "")
        response_spans = resp.get("response_spans", None)
        scores = resp.get("scores", [])
        for j, span_obj in enumerate(scores):
            data_dict["response_id"].append(i)
            data_dict["span_id"].append(j)
            data_dict["task_type"].append(task_type)
            data_dict["input"].append(prompt_text)
            if response_spans is not None and j < len(response_spans):
                a, b = response_spans[j]
                data_dict["output"].append(response_text[a:b])
            else:
                # fallback: use full response text
                data_dict["output"].append(response_text)

            prompt_attention_score: Dict[str, float] = span_obj.get("prompt_attention_score", {})
            parameter_knowledge_scores: Dict[str, float] = span_obj.get("parameter_knowledge_scores", {})

            # Make deterministic order (as in original code: list(values())[k])
            ext_items = list(prompt_attention_score.items())
            par_items = list(parameter_knowledge_scores.items())

            if len(ext_items) < top_n or len(par_items) < top_n:
                raise ValueError(
                    f"Not enough features in span scores. Got ext={len(ext_items)}, par={len(par_items)}, need={top_n}. "
                    f"Check your ReDeEP chunk json: {response_chunk_json}"
                )

            for k in range(top_n):
                ext_key, ext_val = ext_items[k]
                par_key, par_val = par_items[k]
                data_dict[f"ext_{k}"].append(float(ext_val))
                data_dict[f"par_{k}"].append(float(par_val))

                # Save mapping (only need once)
                if f"ext_{k}" not in ext_map:
                    # ext_key is like "(layer, head)"
                    # It is stored as string by chunk_level_detect.py; safe parse:
                    # expected: "(12, 3)" or "(12,3)"
                    cleaned = ext_key.strip().lstrip("(").rstrip(")")
                    parts = [p.strip() for p in cleaned.split(",")]
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        ext_map[f"ext_{k}"] = (int(parts[0]), int(parts[1]))
                    else:
                        ext_map[f"ext_{k}"] = (-1, -1)

                if f"par_{k}" not in para_map:
                    # par_key is like "layer_0", "layer_1", ...
                    if par_key.startswith("layer_"):
                        para_map[f"par_{k}"] = int(par_key.split("_", 1)[1])
                    else:
                        para_map[f"par_{k}"] = -1

            data_dict["label"].append(int(span_obj.get("hallucination_label", 0)))

    df = pd.DataFrame(data_dict)
    if df.empty:
        raise ValueError(f"No test split rows found in: {response_chunk_json}")
    return df, ext_map, para_map


def _select_circuits_via_pcc(
    df: pd.DataFrame,
    ext_map: Dict[str, Tuple[int, int]],
    para_map: Dict[str, int],
    pcc_ext_threshold: float,
    pcc_par_threshold: float,
    late_layer_min: int,
) -> MechSparseCircuits:
    """
    Implements the paper's selection rule in a pragmatic way:
      - copy heads 𝒜: PCC(ext_feature, label) < pcc_ext_threshold  (ext correlates with NON-hallucination)
      - knowledge layers 𝓕: PCC(par_feature, label) > pcc_par_threshold AND layer >= late_layer_min
    """
    label = df["label"].astype(float).values

    copy_heads = []
    for col, (l, h) in ext_map.items():
        if l < 0:
            continue
        pcc, _ = pearsonr(df[col].astype(float).values, label)
        if np.isfinite(pcc) and pcc < pcc_ext_threshold:
            copy_heads.append((l, h))

    knowledge_layers = set()
    for col, layer_id in para_map.items():
        if layer_id < 0 or layer_id < late_layer_min:
            continue
        pcc, _ = pearsonr(df[col].astype(float).values, label)
        if np.isfinite(pcc) and pcc > pcc_par_threshold:
            knowledge_layers.add(layer_id)

    copy_heads = sorted(list(set(copy_heads)))
    knowledge_layers = sorted(list(knowledge_layers))
    return MechSparseCircuits(copy_heads=copy_heads, knowledge_layers=knowledge_layers, ext_map=ext_map, para_map=para_map)


def _compute_H_proxy(
    df: pd.DataFrame,
    alpha: float,
    beta: float,
    m: float,
) -> np.ndarray:
    """
    ReDeEP-style proxy:
      H = m * norm(sum(par)) - alpha * norm(sum(ext))
    (beta is reserved if you later want per-head weighting; currently folded into alpha.)
    """
    ext_cols = [c for c in df.columns if c.startswith("ext_")]
    par_cols = [c for c in df.columns if c.startswith("par_")]
    ext_sum = df[ext_cols].sum(axis=1).astype(float).values
    par_sum = df[par_cols].sum(axis=1).astype(float).values

    scaler = MinMaxScaler()
    ext_norm = scaler.fit_transform(ext_sum.reshape(-1, 1)).reshape(-1)
    par_norm = scaler.fit_transform(par_sum.reshape(-1, 1)).reshape(-1)

    H = m * par_norm - alpha * ext_norm
    return H.astype(np.float32)


def _write_mechsparse_jsonl(
    out_jsonl: str,
    df: pd.DataFrame,
    H: np.ndarray,
):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(len(df)):
            obj = {
                "response_id": int(df.iloc[i]["response_id"]),
                "span_id": int(df.iloc[i]["span_id"]),
                "task_type": str(df.iloc[i]["task_type"]),
                "input": str(df.iloc[i]["input"]),
                "output": str(df.iloc[i]["output"]),
                "label": int(df.iloc[i]["label"]),
                "H": float(H[i]),
                # Keep raw vectors for optional analysis / ablation
                "ext": [float(df.iloc[i][c]) for c in df.columns if c.startswith("ext_")],
                "par": [float(df.iloc[i][c]) for c in df.columns if c.startswith("par_")],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extract MechSparse circuits (A, F) and H proxy from ReDeEP chunk outputs.")
    parser.add_argument("--response_chunk_json", type=str, required=True, help="Path to llama*_response_chunk.json")
    parser.add_argument("--source_info_jsonl", type=str, required=True, help="Path to source_info_spans.jsonl / source_info.jsonl")
    parser.add_argument("--top_n", type=int, default=32, help="How many ext/par scores per span are present (default 32)")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated splits to include from response json (e.g., 'train,test' or 'train').",
    )
    parser.add_argument("--pcc_ext_threshold", type=float, default=-0.4, help="PCC(ext, label) < threshold => copy head")
    parser.add_argument("--pcc_par_threshold", type=float, default=0.3, help="PCC(par, label) > threshold => knowledge layer")
    parser.add_argument("--late_layer_min", type=int, default=18, help="Only keep knowledge layers >= this")
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for external score in H proxy (matches chunk_level_reg defaults)")
    parser.add_argument("--beta", type=float, default=1.0, help="Reserved weight for ext (kept for doc completeness)")
    parser.add_argument("--m", type=float, default=1.0, help="Weight for parameter score in H proxy (matches chunk_level_reg defaults)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for circuits.json and dataset.jsonl")
    args = parser.parse_args()

    source_info_dict = _load_source_info(args.source_info_jsonl)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    df, ext_map, para_map = _construct_chunk_df(args.response_chunk_json, source_info_dict, args.top_n, splits=splits)

    circuits = _select_circuits_via_pcc(
        df=df,
        ext_map=ext_map,
        para_map=para_map,
        pcc_ext_threshold=args.pcc_ext_threshold,
        pcc_par_threshold=args.pcc_par_threshold,
        late_layer_min=args.late_layer_min,
    )
    H = _compute_H_proxy(df, alpha=args.alpha, beta=args.beta, m=args.m)

    os.makedirs(args.out_dir, exist_ok=True)
    circuits_path = os.path.join(args.out_dir, "mechsparse_circuits.json")
    with open(circuits_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "copy_heads": [[l, h] for (l, h) in circuits.copy_heads],
                "knowledge_layers": circuits.knowledge_layers,
                "ext_map": {k: [v[0], v[1]] for k, v in circuits.ext_map.items()},
                "para_map": circuits.para_map,
                "params": {
                    "pcc_ext_threshold": args.pcc_ext_threshold,
                    "pcc_par_threshold": args.pcc_par_threshold,
                    "late_layer_min": args.late_layer_min,
                    "top_n": args.top_n,
                    "alpha": args.alpha,
                    "m": args.m,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    dataset_jsonl = os.path.join(args.out_dir, "mechsparse_redeep_chunk_dataset.jsonl")
    _write_mechsparse_jsonl(dataset_jsonl, df, H)

    print(f"[MechSparse] Wrote circuits: {circuits_path}")
    print(f"[MechSparse] Wrote dataset : {dataset_jsonl}")
    print(f"[MechSparse] |A|={len(circuits.copy_heads)}  |F|={len(circuits.knowledge_layers)}  N={len(df)}")


if __name__ == "__main__":
    main()

