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
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os

# Allow importing from RAGLens/src
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
RAGLENS_SRC = os.path.join(ROOT, "RAGLens", "src")
sys.path.insert(0, RAGLENS_SRC)

from RAGLens import RAGLens  # noqa: E402


@dataclass
class ReweightConfig:
    tau: float = 0.5
    alpha2: float = 1.3
    beta2: float = 0.7


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
        self._alpha2 = 1.0
        self._beta2 = 1.0

    def enable(self, *, alpha2: float, beta2: float):
        self._enabled = True
        self._alpha2 = float(alpha2)
        self._beta2 = float(beta2)

    def disable(self):
        self._enabled = False
        self._alpha2 = 1.0
        self._beta2 = 1.0

    def _get_layers(self):
        base = getattr(self.model, "model", None) or getattr(self.model, "base_model", None)
        return base.layers

    def install(self):
        layers = self._get_layers()

        def make_attn_hook(layer_idx: int):
            def hook(module, inputs, outputs):
                if not self._enabled:
                    return outputs
                # outputs: (bs, seq, hidden)
                return outputs * self._alpha2
            return hook

        def make_mlp_hook(layer_idx: int):
            def hook(module, inputs, outputs):
                if not self._enabled:
                    return outputs
                return outputs * self._beta2
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


def generate_with_aarf_plus(
    *,
    model,
    tokenizer,
    detector: RAGLens,
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
                reweighter.enable(alpha2=cfg.alpha2, beta2=cfg.beta2)
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
    sae = torch.load(sae_path, map_location="cpu")

    detector = RAGLens.load(args.detector_pkl, tokenizer=tokenizer, model=model, sae=sae)

    cfg = ReweightConfig(tau=args.tau, alpha2=args.alpha2, beta2=args.beta2)
    text = generate_with_aarf_plus(
        model=model,
        tokenizer=tokenizer,
        detector=detector,
        prompt=args.prompt,
        cfg=cfg,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(text)


if __name__ == "__main__":
    main()

