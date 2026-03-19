import tqdm
import torch
from typing import List, Optional, Sequence, Tuple

from mechsparse_residuals import get_mechsparse_residuals_llama_like


def _get_sae_device_dtype(sae) -> tuple[torch.device, torch.dtype]:
    """
    Some SAE implementations (e.g. sparsify.SparseCoder) expose .device/.dtype,
    while the repo's SimpleSAE only has parameters.
    """
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

def sae_encoding(input_ids, attention_mask, hookpoint, model, sae, activation=False, topk=False):
    hook_results = []
    def get_hidden_state(module, inputs, outputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = outputs.flatten(0, 1)
        hook_results.append(inputs.flatten(0, 1) if sae.cfg.transcode else outputs)
    with torch.inference_mode():
        handle = model.base_model.get_submodule(hookpoint).register_forward_hook(get_hidden_state)
        _ = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
        )
        hidden_state = hook_results[0]
        handle.remove()
        sae_device, sae_dtype = _get_sae_device_dtype(sae)
        features = sae.encode(hidden_state.to(sae_device).to(sae_dtype))
    if not topk:        
        # Tensor-returning SAE (e.g. SimpleSAE): already equals pre-acts / activations.
        if isinstance(features, torch.Tensor):
            return features

        # SparseCoder/SAE-style output: expect EncoderOutput-like object with pre_acts.
        if hasattr(features, "pre_acts"):
            if activation:
                # Keep only top-k activations (requires top_indices/top_acts).
                if hasattr(features, "top_indices") and hasattr(features, "top_acts"):
                    pre_acts = torch.zeros_like(features.pre_acts)
                    pre_acts = pre_acts.scatter(1, features.top_indices, features.top_acts)
                    return pre_acts
            return features.pre_acts

        raise ValueError(f"Unsupported SAE encode output type: {type(features)}")
    else:
        if isinstance(features, torch.Tensor):
            raise ValueError("topk not supported for Tensor-returning SAE")

        if hasattr(features, "top_acts") and hasattr(features, "top_indices"):
            return features.top_acts, features.top_indices
        raise ValueError("SAE encode output missing top_acts/top_indices required for topk=True")

def encode_outputs(inputs, outputs, hookpoint, tokenizer, model, sae, agg="max", activation=False, show_progress=True, fixed_indices=None):
    assert len(inputs) == len(outputs)
    if show_progress:
        iterator = tqdm.tqdm(range(len(inputs)))
    else:
        iterator = range(len(inputs))
    if agg.startswith("acti_"):
        activation = True
        agg = agg[5:]
    features = []
    for idx in iterator:
        input = inputs[idx]
        output = outputs[idx]
        if tokenizer.chat_template:
            input = tokenizer.apply_chat_template([{'role': 'user', 'content': input}], tokenize=False, add_special_tokens=True, add_generation_prompt=True)
        text = input + output
        encoded_text = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False if tokenizer.chat_template else True)
        pre_acts = sae_encoding(encoded_text['input_ids'], encoded_text['attention_mask'], hookpoint, model, sae, activation=activation)
        offsets = encoded_text['offset_mapping'][0]
        output_start_idx = None
        output_end_idx = len(encoded_text['input_ids'][0])
        for i, span in enumerate(offsets):
            if span[0] <= len(input) < span[1]:
                output_start_idx = i
            if output_start_idx is not None and output_end_idx is not None:
                break
        if agg == "max":
            agg_pre_acts = pre_acts[output_start_idx:output_end_idx].max(dim=0)[0]
        elif agg == "mean":
            agg_pre_acts = pre_acts[output_start_idx:output_end_idx].mean(dim=0)
        elif agg == "sum":
            agg_pre_acts = pre_acts[output_start_idx:output_end_idx].sum(dim=0)
        else:
            raise ValueError(f"Unknown agg method: {agg}")
        features.append(agg_pre_acts.cpu())
    features = torch.stack(features, dim=0)
    return features.half()


def encode_mechsparse_outputs(
    inputs: List[str],
    outputs: List[str],
    tokenizer,
    model,
    sae,
    *,
    copy_heads: Sequence[Tuple[int, int]],
    knowledge_layers: Sequence[int],
    agg: str = "max",
    activation: bool = False,
    show_progress: bool = True,
):
    """
    MechSparse variant of RAGLens encoding:
      - Build text = input + output (with chat template if needed)
      - Run a single forward pass to obtain r_ext/r_par per token using ReDeEP-selected circuits
      - Concatenate residuals: r = [r_ext; r_par]  -> shape (seq, 2*hidden)
      - SAE encode(r) and aggregate over output-token span (max/mean/sum)

    This matches the MechSparse paper's "circuit-guided SAE" idea, while staying compatible
    with the rest of RAGLens (MI selection + GAM).
    """
    assert len(inputs) == len(outputs)
    iterator = tqdm.tqdm(range(len(inputs))) if show_progress else range(len(inputs))

    if agg.startswith("acti_"):
        activation = True
        agg = agg[5:]

    features = []
    for idx in iterator:
        input_text = inputs[idx]
        output_text = outputs[idx]
        if getattr(tokenizer, "chat_template", None):
            input_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                tokenize=False,
                add_special_tokens=True,
                add_generation_prompt=True,
            )
        full_text = input_text + output_text
        encoded = tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False if getattr(tokenizer, "chat_template", None) else True,
        )
        offsets = encoded["offset_mapping"][0]

        # Find output token span using the same heuristic as encode_outputs
        output_start_idx = None
        output_end_idx = encoded["input_ids"].shape[-1]
        for i, span in enumerate(offsets):
            if span[0] <= len(input_text) < span[1]:
                output_start_idx = i
            if output_start_idx is not None and output_end_idx is not None:
                break
        if output_start_idx is None:
            # Fallback: treat last tokens as output
            output_start_idx = max(0, output_end_idx - 1)

        batch_res = get_mechsparse_residuals_llama_like(
            model=model,
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask", None),
            copy_heads=copy_heads,
            knowledge_layers=knowledge_layers,
            device=model.device,
        )

        # r: (seq, 2*hidden)
        r = torch.cat([batch_res.r_ext[0], batch_res.r_par[0]], dim=-1)

        # SAE encode expects (tokens, d_in)
        sae_device, sae_dtype = _get_sae_device_dtype(sae)
        z = sae.encode(r.to(sae_device).to(sae_dtype))
        # Tensor-returning SAE (SimpleSAE): z is already a (tokens, d_sae) activation matrix.
        if isinstance(z, torch.Tensor):
            pre_acts = z
        else:
            if type(sae).__name__ == "SparseAutoEncoder":
                pre_acts = z
            else:
                if activation:
                    pre_acts = torch.zeros_like(z.pre_acts)
                    pre_acts = pre_acts.scatter(1, z.top_indices, z.top_acts)
                else:
                    pre_acts = z.pre_acts

        token_feats = pre_acts[output_start_idx:output_end_idx]
        if agg == "max":
            agg_feats = token_feats.max(dim=0)[0]
        elif agg == "mean":
            agg_feats = token_feats.mean(dim=0)
        elif agg == "sum":
            agg_feats = token_feats.sum(dim=0)
        else:
            raise ValueError(f"Unknown agg method: {agg}")
        features.append(agg_feats.detach().cpu())

    return torch.stack(features, dim=0).half()