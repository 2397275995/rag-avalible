"""
MechSparse AARF+ (feature-aware reweighting) — research prototype.

This script provides a minimal, reproducible implementation of AARF+ consistent with the paper text:
  - Use a MechSparse detector (SAE + GAM/EBM) to predict hallucination risk at each generation step
  - If risk > tau, amplify "copy" circuits and dampen "param knowledge" circuits

Implementation notes (pragmatic):
  - We apply reweighting at LAYER level (not per-head), because HF modules expose the post-attention o_proj output.
  - "Copy circuit" layers are inferred from copy_heads' layer ids.
  - "Knowledge FFN" layers are given by knowledge_layers.
  - Detector must be trained and saved via RAGLens.save().

Usage (example):
  python mechsparse_aarf_plus.py ^
    --model_name_or_path <local_hf_model_path> ^
    --detector_pkl <path_to_mechsparse_detector.pkl> ^
    --prompt "..." --max_new_tokens 128
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os

# Allow importing from RAGLens/src
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
RAGLENS_SRC = os.path.join(ROOT, "RAGLens", "src")
sys.path.insert(0, RAGLENS_SRC)

from RAGLens import RAGLens  # noqa: E402
from mechsparse_residuals import get_mechsparse_residuals_llama_like  # noqa: E402


@dataclass
class ReweightConfig:
    tau: float = 0.5
    alpha2: float = 1.3
    beta2: float = 0.7
    max_active_features: int = 8
    max_copy_layers_per_step: int = 6
    max_knowledge_layers_per_step: int = 6


class LayerReweighter:
    def __init__(
        self,
        model,
        *,
        copy_layers: Set[int],
        knowledge_layers: Set[int],
    ):
        self.model = model
        self.copy_layers = copy_layers
        self.knowledge_layers = knowledge_layers
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._enabled = False
        self._active_copy_layers: Set[int] = set()
        self._active_knowledge_layers: Set[int] = set()
        self._alpha2 = 1.0
        self._beta2 = 1.0

    def set_active(
        self,
        *,
        active_copy_layers: Set[int],
        active_knowledge_layers: Set[int],
        alpha2: float,
        beta2: float,
    ):
        self._enabled = True
        self._active_copy_layers = set(int(x) for x in active_copy_layers)
        self._active_knowledge_layers = set(int(x) for x in active_knowledge_layers)
        self._alpha2 = float(alpha2)
        self._beta2 = float(beta2)

    def disable(self):
        self._enabled = False
        self._active_copy_layers = set()
        self._active_knowledge_layers = set()
        self._alpha2 = 1.0
        self._beta2 = 1.0

    def _get_layers(self):
        base = getattr(self.model, "model", None) or getattr(self.model, "base_model", None)
        return base.layers

    def install(self):
        layers = self._get_layers()

        def make_attn_hook(layer_idx: int):
            def hook(module, inputs, outputs):
                if not self._enabled or layer_idx not in self._active_copy_layers:
                    return outputs
                return outputs * self._alpha2  # scale post-attention o_proj output
            return hook

        def make_mlp_hook(layer_idx: int):
            def hook(module, inputs, outputs):
                if not self._enabled or layer_idx not in self._active_knowledge_layers:
                    return outputs
                return outputs * self._beta2  # scale FFN down_proj output
            return hook

        for l in sorted(self.copy_layers):
            if l < 0 or l >= len(layers):
                continue
            # scale post-attention output (o_proj output)
            self._handles.append(layers[l].self_attn.o_proj.register_forward_hook(make_attn_hook(l)))

        for l in sorted(self.knowledge_layers):
            if l < 0 or l >= len(layers):
                continue
            # scale FFN output (down_proj output)
            self._handles.append(layers[l].mlp.down_proj.register_forward_hook(make_mlp_hook(l)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

def _get_input_text_with_template(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_special_tokens=True,
            add_generation_prompt=True,
        )
    return prompt


def _get_sae_device_dtype(sae) -> tuple[torch.device, torch.dtype]:
    device = getattr(sae, "device", None)
    dtype = getattr(sae, "dtype", None)
    if device is None or dtype is None:
        try:
            p = next(sae.parameters())
            if device is None:
                device = p.device
            if dtype is None:
                dtype = p.dtype
        except StopIteration:
            device = device or torch.device("cpu")
            dtype = dtype or torch.float32
    return device, dtype


@torch.no_grad()
def select_feature_guided_layers(
    *,
    model,
    tokenizer,
    detector: RAGLens,
    sae,
    prompt: str,
    out_suffix: str,
    max_active_features: int,
    max_copy_layers_per_step: int,
    max_knowledge_layers_per_step: int,
) -> tuple[Set[int], Set[int]]:
    """
    SAE-feature-guided approximation of AARF+:
      - compute SAE activations for the last token's circuit residual r=[r_ext;r_par]
      - keep top positive features among detector.top_k_indices
      - attribute each feature back to copy-head layers / knowledge layers via SAE encoder weights
      - choose top layers to amplify/dampen

    Note: This is an efficient attribution proxy (not integrated gradients).
    """
    copy_heads = detector.copy_heads or []
    knowledge_layers = detector.knowledge_layers or []
    if len(copy_heads) == 0 and len(knowledge_layers) == 0:
        return set(), set()

    copy_layers_all = {int(l) for (l, _h) in copy_heads}
    knowledge_layers_all = {int(l) for l in knowledge_layers}

    # Require SAE to be tensor-returning and have an encoder weight matrix for attribution.
    if not hasattr(sae, "encode"):
        return copy_layers_all, knowledge_layers_all
    if not hasattr(sae, "encoder") or not hasattr(sae.encoder, "weight"):
        return copy_layers_all, knowledge_layers_all

    sae_device, sae_dtype = _get_sae_device_dtype(sae)

    input_text = _get_input_text_with_template(tokenizer, prompt)
    full_text = input_text + out_suffix

    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False if getattr(tokenizer, "chat_template", None) else True,
    )

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    seq_len = input_ids.shape[-1]
    token_idx = max(0, seq_len - 1)

    res = get_mechsparse_residuals_llama_like(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        copy_heads=copy_heads,
        knowledge_layers=knowledge_layers,
        device=model.device,
        token_idx=token_idx,
        return_parts=True,
    )

    # Circuit residual for last token
    r_ext_tok = res.r_ext[0, token_idx, :].to(sae_device).to(sae_dtype)
    r_par_tok = res.r_par[0, token_idx, :].to(sae_device).to(sae_dtype)
    r_cat_tok = torch.cat([r_ext_tok, r_par_tok], dim=-1)  # (2*hidden,)

    z = sae.encode(r_cat_tok.unsqueeze(0))  # (1, d_sae) or EncoderOutput-like
    if isinstance(z, torch.Tensor):
        z_vec = z[0]
    elif hasattr(z, "pre_acts"):
        # SparseCoder returns an EncoderOutput-like NamedTuple
        z_vec = z.pre_acts[0]
    else:
        return copy_layers_all, knowledge_layers_all

    # Candidate feature indices from detector
    if detector.top_k_indices is not None:
        cand = detector.top_k_indices
        if isinstance(cand, np.ndarray):
            candidate = [int(x) for x in cand.tolist()]
        else:
            candidate = [int(x) for x in cand]
    else:
        candidate = list(range(z_vec.shape[0]))

    active: list[tuple[int, float]] = []
    for idx in candidate:
        if idx < 0 or idx >= z_vec.numel():
            continue
        val = float(z_vec[idx].item())
        if val > 0:
            active.append((idx, val))

    if not active:
        return copy_layers_all, knowledge_layers_all

    active.sort(key=lambda x: x[1], reverse=True)
    active = active[: max_active_features]

    hidden = int(model.config.hidden_size)
    W = sae.encoder.weight.to(device=z_vec.device, dtype=z_vec.dtype)  # (d_sae, 2*hidden)
    W_ext = W[:, :hidden]  # (d_sae, hidden)
    W_par = W[:, hidden:]  # (d_sae, hidden)

    copy_scores: Dict[int, float] = {}
    knowledge_scores: Dict[int, float] = {}

    if res.r_ext_parts is None or res.r_par_parts is None:
        return copy_layers_all, knowledge_layers_all

    for feat_idx, _val in active:
        wext_j = W_ext[feat_idx]  # (hidden,)
        wpar_j = W_par[feat_idx]  # (hidden,)

        # Copy layers: for each (layer, head) component, score by dot product.
        for (l, _h), head_vec in res.r_ext_parts.items():
            head_vec_tok = head_vec[0].to(device=wext_j.device, dtype=wext_j.dtype)
            score = float(torch.dot(wext_j, head_vec_tok).item())
            prev = copy_scores.get(int(l), -1e18)
            copy_scores[int(l)] = max(prev, score)

        # Knowledge layers: per-layer FFN down_proj contribution.
        for l, layer_vec in res.r_par_parts.items():
            layer_vec_tok = layer_vec[0].to(device=wpar_j.device, dtype=wpar_j.dtype)
            score = float(torch.dot(wpar_j, layer_vec_tok).item())
            prev = knowledge_scores.get(int(l), -1e18)
            knowledge_scores[int(l)] = max(prev, score)

    active_copy_layers = set(
        sorted(copy_scores.keys(), key=lambda l: copy_scores[l], reverse=True)[: max_copy_layers_per_step]
    )
    active_knowledge_layers = set(
        sorted(knowledge_scores.keys(), key=lambda l: knowledge_scores[l], reverse=True)[: max_knowledge_layers_per_step]
    )

    # Fallback to safe full circuit sets if selection fails.
    if not active_copy_layers:
        active_copy_layers = set(copy_layers_all)
    if not active_knowledge_layers:
        active_knowledge_layers = set(knowledge_layers_all)

    return active_copy_layers, active_knowledge_layers


def generate_with_aarf_plus(
    *,
    model,
    tokenizer,
    detector: RAGLens,
    sae,
    prompt: str,
    cfg: ReweightConfig,
    max_new_tokens: int,
    temperature: float,
):
    model.eval()
    device = model.device

    # Map copy_heads -> copy_layers (layer-level intervention)
    copy_layers = set(int(l) for (l, _h) in (detector.copy_heads or []))
    knowledge_layers = set(int(l) for l in (detector.knowledge_layers or []))

    reweighter = LayerReweighter(model, copy_layers=copy_layers, knowledge_layers=knowledge_layers)
    reweighter.install()

    try:
        # Tokenize prompt
        if getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_special_tokens=True,
                add_generation_prompt=True,
            )
        else:
            prompt_text = prompt

        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        generated = input_ids

        for _step in range(max_new_tokens):
            # Build current text for detector: input(prompt) + output(generated suffix)
            # Here we treat prompt as "input" and generated suffix as "output"
            current_text = tokenizer.decode(generated[0], skip_special_tokens=False)
            # Heuristic split: we keep original prompt_text as input, suffix as output
            if current_text.startswith(prompt_text):
                out_suffix = current_text[len(prompt_text) :]
            else:
                out_suffix = current_text

            # Predict hallucination risk for current suffix (use H=None online)
            # Note: This is a lightweight trigger; for the paper you should evaluate on labeled sets offline.
            if len(out_suffix.strip()) > 0:
                p_hall = float(detector.predict_proba([prompt], [out_suffix], H_values=None)[0])
            else:
                p_hall = 0.0

            if p_hall > cfg.tau:
                active_copy_layers, active_knowledge_layers = select_feature_guided_layers(
                    model=model,
                    tokenizer=tokenizer,
                    detector=detector,
                    sae=sae,
                    prompt=prompt,
                    out_suffix=out_suffix,
                    max_active_features=cfg.max_active_features,
                    max_copy_layers_per_step=cfg.max_copy_layers_per_step,
                    max_knowledge_layers_per_step=cfg.max_knowledge_layers_per_step,
                )
                reweighter.set_active(
                    active_copy_layers=active_copy_layers,
                    active_knowledge_layers=active_knowledge_layers,
                    alpha2=cfg.alpha2,
                    beta2=cfg.beta2,
                )
            else:
                reweighter.disable()

            with torch.inference_mode():
                out = model(input_ids=generated, use_cache=False, return_dict=True)
                logits = out.logits[:, -1, :]
                if temperature <= 0:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_id], dim=-1)

        return tokenizer.decode(generated[0], skip_special_tokens=True)
    finally:
        reweighter.remove()


def main():
    parser = argparse.ArgumentParser(description="MechSparse AARF+ generation prototype")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--detector_pkl", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--alpha2", type=float, default=1.3)
    parser.add_argument("--beta2", type=float, default=0.7)
    parser.add_argument("--max_active_features", type=int, default=8)
    parser.add_argument("--max_copy_layers_per_step", type=int, default=6)
    parser.add_argument("--max_knowledge_layers_per_step", type=int, default=6)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # SAE must be loaded by the user; here we assume you have a torch.load-able SAE checkpoint.
    # For a complete pipeline, train SAE with RAGLens/sparsify and then load here.
    raise_if_no_sae = True
    sae_path = os.environ.get("MECHSPARSE_SAE_PATH")
    if sae_path is None:
        if raise_if_no_sae:
            raise RuntimeError(
                "Please set env MECHSPARSE_SAE_PATH to a trained SAE checkpoint path (torch.load)."
            )
    sae = torch.load(sae_path, map_location=model.device)
    sae.eval()

    detector = RAGLens.load(args.detector_pkl, tokenizer=tokenizer, model=model, sae=sae)

    cfg = ReweightConfig(
        tau=args.tau,
        alpha2=args.alpha2,
        beta2=args.beta2,
        max_active_features=args.max_active_features,
        max_copy_layers_per_step=args.max_copy_layers_per_step,
        max_knowledge_layers_per_step=args.max_knowledge_layers_per_step,
    )
    text = generate_with_aarf_plus(
        model=model,
        tokenizer=tokenizer,
        detector=detector,
        sae=sae,
        prompt=args.prompt,
        cfg=cfg,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(text)


if __name__ == "__main__":
    main()

