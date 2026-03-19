from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass
class MechSparseBatchResiduals:
    """
    Residual streams per token (batch-first).
    Shapes:
      - r_ext: (bs, seq, hidden)
      - r_par: (bs, seq, hidden)
    """
    r_ext: torch.Tensor
    r_par: torch.Tensor
    # Optional per-component contributions used for feature-aware attribution
    # (only populated when token_idx is provided and return_parts=True).
    r_ext_parts: Optional[Dict[Tuple[int, int], torch.Tensor]] = None
    r_par_parts: Optional[Dict[int, torch.Tensor]] = None


def _is_llama_like(model) -> bool:
    # Supports LLaMA / Qwen-Llama style HF modules with model.model.layers[i]
    base = getattr(model, "model", None) or getattr(model, "base_model", None)
    if base is None:
        return False
    layers = getattr(base, "layers", None)
    return layers is not None


def _get_layers(model):
    base = getattr(model, "model", None) or getattr(model, "base_model", None)
    return base.layers


def get_mechsparse_residuals_llama_like(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    copy_heads: Sequence[Tuple[int, int]],
    knowledge_layers: Sequence[int],
    device: Optional[torch.device] = None,
    token_idx: Optional[int] = None,
    return_parts: bool = False,
) -> MechSparseBatchResiduals:
    """
    Extract approximate circuit-specific residual contributions for LLaMA-like HF models.

    External-copy residual (r_ext):
      - For each selected (layer, head), we reconstruct the head output using:
          head_ctx = AttnWeights @ V
        then project through the corresponding slice of o_proj, and sum over heads.
      - Requires `output_attentions=True` and a hook on v_proj output.

    Param-knowledge residual (r_par):
      - Sum mlp.down_proj outputs from selected `knowledge_layers`.

    Notes:
      - This is a pragmatic implementation aimed at research prototyping.
      - Some model variants may name submodules differently; this function assumes:
          layers[l].self_attn.v_proj, layers[l].self_attn.o_proj, layers[l].mlp.down_proj
    """
    if not _is_llama_like(model):
        raise ValueError("Model is not LLaMA-like; cannot extract MechSparse residuals with this helper.")

    if device is None:
        device = model.device

    layers = _get_layers(model)
    bs, seq = input_ids.shape
    hidden = model.config.hidden_size
    if token_idx is not None:
        token_idx = int(token_idx)
        if token_idx < 0 or token_idx >= seq:
            raise ValueError(f"token_idx out of range: token_idx={token_idx}, seq={seq}")

    # Group copy_heads by layer for efficiency
    copy_heads_by_layer: Dict[int, List[int]] = {}
    for l, h in copy_heads:
        copy_heads_by_layer.setdefault(int(l), []).append(int(h))
    for l in copy_heads_by_layer:
        copy_heads_by_layer[l] = sorted(list(set(copy_heads_by_layer[l])))

    knowledge_layers = sorted(list(set(int(l) for l in knowledge_layers)))

    v_cache: Dict[int, torch.Tensor] = {}
    mlp_cache: Dict[int, torch.Tensor] = {}
    handles = []

    def _make_v_hook(layer_idx: int):
        def hook(module, inputs, outputs):
            # outputs: (bs, seq, hidden) or (bs, seq, n_heads*head_dim)
            v_cache[layer_idx] = outputs
        return hook

    def _make_mlp_hook(layer_idx: int):
        def hook(module, inputs, outputs):
            # outputs: (bs, seq, hidden) from down_proj
            mlp_cache[layer_idx] = outputs
        return hook

    # Register hooks only on needed layers
    for l in copy_heads_by_layer.keys():
        handles.append(layers[l].self_attn.v_proj.register_forward_hook(_make_v_hook(l)))
    for l in knowledge_layers:
        handles.append(layers[l].mlp.down_proj.register_forward_hook(_make_mlp_hook(l)))

    try:
        with torch.inference_mode():
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device) if attention_mask is not None else None,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for h in handles:
            h.remove()

    attentions = out.attentions  # tuple(num_layers) each: (bs, n_heads, tgt, src)
    r_ext = torch.zeros((bs, seq, hidden), device=device, dtype=torch.float32)
    r_par = torch.zeros((bs, seq, hidden), device=device, dtype=torch.float32)
    r_ext_parts: Optional[Dict[Tuple[int, int], torch.Tensor]] = {} if (return_parts and token_idx is not None) else None
    r_par_parts: Optional[Dict[int, torch.Tensor]] = {} if (return_parts and token_idx is not None) else None

    # Build r_ext by reconstructing each selected head output
    for layer_idx, heads in copy_heads_by_layer.items():
        if layer_idx >= len(attentions):
            continue
        attn_w = attentions[layer_idx].to(torch.float32)  # (bs, n_heads, tgt, src)
        v = v_cache.get(layer_idx, None)
        if v is None:
            continue
        v = v.to(torch.float32)

        # Reshape V to (bs, n_heads, src, head_dim)
        n_heads = model.config.num_attention_heads
        head_dim = hidden // n_heads
        if v.shape[-1] == hidden:
            v = v.view(bs, seq, n_heads, head_dim).transpose(1, 2)  # (bs, n_heads, seq, head_dim)
        else:
            v = v.view(bs, seq, n_heads, head_dim).transpose(1, 2)

        # If token_idx is provided, we only compute ctx at that position.
        # Otherwise compute full ctx for all tgt positions.
        if token_idx is None:
            # ctx: (bs, n_heads, tgt, head_dim)
            ctx = torch.matmul(attn_w, v)
        else:
            # attn_w_token: (bs, n_heads, src)
            attn_w_token = attn_w[:, :, token_idx, :]
            # ctx_tok: (bs, n_heads, head_dim)
            ctx_tok = torch.matmul(attn_w_token, v)

        # Project each selected head through its o_proj slice
        o_proj = layers[layer_idx].self_attn.o_proj
        W_o = o_proj.weight.to(device=device, dtype=torch.float32)  # (hidden, hidden_in)

        for head in heads:
            if head < 0 or head >= n_heads:
                continue
            in_start = head * head_dim
            in_end = (head + 1) * head_dim
            W_slice = W_o[:, in_start:in_end]  # (hidden, head_dim)

            if token_idx is None:
                head_ctx = ctx[:, head, :, :]  # (bs, tgt, head_dim)
                # out: (bs, tgt, hidden)
                head_out = torch.einsum("bth,dh->btd", head_ctx, W_slice)
                r_ext += head_out
            else:
                head_ctx_tok = ctx_tok[:, head, :]  # (bs, head_dim)
                # out: (bs, hidden)
                head_out_tok = torch.einsum("bh,dh->bd", head_ctx_tok, W_slice)
                r_ext[:, token_idx, :] += head_out_tok
                if r_ext_parts is not None:
                    r_ext_parts[(layer_idx, head)] = head_out_tok

    # Build r_par by summing MLP down_proj outputs from knowledge layers
    for layer_idx in knowledge_layers:
        mlp_out = mlp_cache.get(layer_idx, None)
        if mlp_out is None:
            continue
        mlp_out = mlp_out.to(torch.float32)
        if token_idx is None:
            r_par += mlp_out
        else:
            r_par[:, token_idx, :] += mlp_out[:, token_idx, :]
            if r_par_parts is not None:
                r_par_parts[layer_idx] = mlp_out[:, token_idx, :]

    return MechSparseBatchResiduals(
        r_ext=r_ext,
        r_par=r_par,
        r_ext_parts=r_ext_parts,
        r_par_parts=r_par_parts,
    )

