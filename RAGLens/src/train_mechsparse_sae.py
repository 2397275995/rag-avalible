import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mechsparse_residuals import get_mechsparse_residuals_llama_like


@dataclass
class SAEConfig:
    d_in: int
    d_hidden: int
    l1_lambda: float = 1e-3


class SimpleSAE(torch.nn.Module):
    """
    Minimal SAE: Linear -> ReLU -> Linear, trained with MSE recon + L1 on activations.
    This is sufficient for "MechSparse SAE" input dim = 2*hidden_size.
    """

    def __init__(self, cfg: SAEConfig, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.cfg = type("Cfg", (), {})()
        self.cfg.transcode = False
        self.d_in = cfg.d_in
        self.d_hidden = cfg.d_hidden
        self.l1_lambda = cfg.l1_lambda
        self.encoder = torch.nn.Linear(cfg.d_in, cfg.d_hidden, bias=True)
        self.decoder = torch.nn.Linear(cfg.d_hidden, cfg.d_in, bias=True)
        self.to(device=device, dtype=dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def load_jsonl_text_pairs(path: str, limit: int | None = None):
    inputs, outputs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            inputs.append(obj["input"])
            outputs.append(obj["output"])
    return inputs, outputs


def sample_r_vectors_for_batch(
    *,
    model,
    tokenizer,
    inputs: List[str],
    outputs: List[str],
    copy_heads: Sequence[Tuple[int, int]],
    knowledge_layers: Sequence[int],
    max_tokens_per_sample: int,
) -> torch.Tensor:
    """
    Build training examples by sampling up to `max_tokens_per_sample` tokens from the output span.
    Returns a tensor (n_examples, 2*hidden).
    """
    device = model.device
    examples = []

    for inp, out in zip(inputs, outputs):
        if getattr(tokenizer, "chat_template", None):
            inp_eff = tokenizer.apply_chat_template(
                [{"role": "user", "content": inp}],
                tokenize=False,
                add_special_tokens=True,
                add_generation_prompt=True,
            )
        else:
            inp_eff = inp

        full_text = inp_eff + out
        enc = tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False if getattr(tokenizer, "chat_template", None) else True,
        )
        offsets = enc["offset_mapping"][0]

        # locate output span start token (same heuristic as RAGLens)
        out_start = None
        out_end = enc["input_ids"].shape[-1]
        for i, span in enumerate(offsets):
            if span[0] <= len(inp_eff) < span[1]:
                out_start = i
            if out_start is not None:
                break
        if out_start is None:
            continue

        res = get_mechsparse_residuals_llama_like(
            model=model,
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc.get("attention_mask", None).to(device) if enc.get("attention_mask", None) is not None else None,
            copy_heads=copy_heads,
            knowledge_layers=knowledge_layers,
            device=device,
        )
        r = torch.cat([res.r_ext[0], res.r_par[0]], dim=-1)  # (seq, 2*hidden)
        out_r = r[out_start:out_end]
        if out_r.shape[0] == 0:
            continue

        # sample tokens uniformly to cap compute
        if out_r.shape[0] > max_tokens_per_sample:
            idx = torch.linspace(0, out_r.shape[0] - 1, steps=max_tokens_per_sample).long()
            out_r = out_r[idx]
        examples.append(out_r.detach().to(torch.float32).cpu())

    if len(examples) == 0:
        raise RuntimeError("No training examples collected. Check tokenizer span alignment and dataset.")
    return torch.cat(examples, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Train a minimal MechSparse SAE on circuit residual inputs (2*hidden).")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_jsonl", type=str, required=True, help="mechsparse_redeep_chunk_dataset.jsonl")
    parser.add_argument("--circuits_json", type=str, required=True, help="mechsparse_circuits.json")
    parser.add_argument("--out_sae", type=str, required=True)
    parser.add_argument("--limit_samples", type=int, default=512, help="How many jsonl lines to use")
    parser.add_argument("--batch_collect", type=int, default=16, help="How many samples per collect step")
    parser.add_argument("--max_tokens_per_sample", type=int, default=32)
    parser.add_argument("--d_hidden", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l1_lambda", type=float, default=1e-3)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    with open(args.circuits_json, "r", encoding="utf-8") as f:
        circuits = json.load(f)
    copy_heads = [tuple(x) for x in circuits["copy_heads"]]
    knowledge_layers = circuits["knowledge_layers"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.dtype == "bf16" else (torch.float16 if args.dtype == "fp16" else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    inputs, outputs = load_jsonl_text_pairs(args.dataset_jsonl, limit=args.limit_samples)

    hidden = model.config.hidden_size
    d_in = 2 * hidden
    device = model.device
    dtype = torch.bfloat16 if args.dtype == "bf16" else (torch.float16 if args.dtype == "fp16" else torch.float32)
    sae = SimpleSAE(SAEConfig(d_in=d_in, d_hidden=args.d_hidden, l1_lambda=args.l1_lambda), device=device, dtype=dtype)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr)

    # Training loop: repeatedly collect small sets of r-vectors and train on them
    step = 0
    while step < args.steps:
        # mini-collect
        for i in range(0, len(inputs), args.batch_collect):
            batch_inp = inputs[i : i + args.batch_collect]
            batch_out = outputs[i : i + args.batch_collect]
            if len(batch_inp) == 0:
                continue
            r_cpu = sample_r_vectors_for_batch(
                model=model,
                tokenizer=tokenizer,
                inputs=batch_inp,
                outputs=batch_out,
                copy_heads=copy_heads,
                knowledge_layers=knowledge_layers,
                max_tokens_per_sample=args.max_tokens_per_sample,
            )
            x = r_cpu.to(device=device, dtype=dtype)

            sae.train()
            x_hat, z = sae(x)
            recon = torch.mean((x_hat - x) ** 2)
            sparsity = torch.mean(torch.abs(z))
            loss = recon + sae.l1_lambda * sparsity

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 50 == 0:
                print(f"[SAE] step={step} loss={loss.item():.6f} recon={recon.item():.6f} l1={sparsity.item():.6f}")
            step += 1
            if step >= args.steps:
                break

    os.makedirs(os.path.dirname(args.out_sae), exist_ok=True)
    torch.save(sae, args.out_sae)
    print(f"[SAE] saved: {args.out_sae}")


if __name__ == "__main__":
    main()

