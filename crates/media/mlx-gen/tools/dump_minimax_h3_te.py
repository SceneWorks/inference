"""MiniMax-H3 text-encoder (Qwen3-VL-32B) context-extraction parity fixture (sc-17143).

Runs the **transformers** ``Qwen3VLTextModel`` — an independent reference graph — at TINY dims and
dumps the inputs, weights and the select-layer hidden states the Rust/MLX port must reproduce.

Why tiny-but-real: the shipped encoder is 64 layers x 5120 hidden (66.7 GB), far too large to
commit. Every *structural* knob is preserved though — bias-less GQA with distinct query/kv head
counts, a ``head_dim`` that is deliberately NOT ``hidden_size / num_heads`` (mirroring the real
128 != 5120/64), per-head q/k RMSNorm before RoPE, SwiGLU, the causal mask, theta 5e6 — so the tiny
model exercises the same block math the real one does.

The layer arithmetic is preserved as a RATIO rather than a value: the real model selects HF
``hidden_states[50]`` out of 64 layers, i.e. it runs 50 and leaves 14 unused. The fixture selects
``hidden_states[4]`` out of 6, running 4 and leaving 2. Critically it also dumps
``hidden_states[3]`` and ``hidden_states[5]`` so the Rust test can assert the port is *unequal* to
both neighbours — a parity fixture that only carries the correct answer cannot catch an off-by-one,
because a shifted implementation still produces a plausible-looking tensor.

Random norm weights: Qwen3 RMSNorm initializes to ones, which hides the weight-multiply entirely.
Every norm is re-randomized and non-degeneracy is asserted before writing.

Run from a torch venv:  ~/Repos/mflux/.venv/bin/python tools/dump_minimax_h3_te.py
"""

from __future__ import annotations

import json
import os

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from _paths import fixture, hf_hub_cache

torch.manual_seed(0)

# Tiny dims. head_dim (32) != hidden/heads (16), mirroring the real 128 != 5120/64.
VOCAB, HIDDEN, INTER, LAYERS = 256, 64, 128, 6
HEADS, KVHEADS, HEAD_DIM = 4, 2, 32
EPS, THETA = 1e-6, 5_000_000.0

# The real model takes HF hidden_states[50] of 64 (50 run, 14 unused). Here: [4] of 6 (4 run, 2
# unused) — the same "select < depth" shape, so the unused-tail trim is genuinely exercised.
SELECT = 4
# Neighbours dumped so the Rust test can prove a one-off in EITHER direction fails.
NEIGHBOURS = (SELECT - 1, SELECT + 1)
PREFIX = 3  # `<|im_start|>user\n` — derived from the shipped chat_template.json (see below)
SEQ = 12

MODEL_REPO = "models--MiniMaxAI--MiniMax-H3"


def snapshot_dir() -> str | None:
    """The MiniMax-H3 snapshot root, if this machine has one (for the template/token half)."""
    root = hf_hub_cache() / MODEL_REPO / "snapshots"
    if not root.is_dir():
        return None
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    return str(snaps[-1]) if snaps else None


def template_facts() -> dict:
    """Derive the chat-template prefix and the special-token ids FROM THE SHIPPED FILES.

    This is the half of the fixture that pins configuration no tensor can witness:

    * ``PREFIX``/``PREFIX_TOKENS`` come from rendering the shipped ``chat_template.json``, not from
      the template string transcribed into Rust;
    * the seven MiniMax specials come from ``tokenizer_config.json``'s
      ``additional_special_tokens``, whose ORDER is the id assignment.
    """
    snap = snapshot_dir()
    if snap is None:
        raise SystemExit(
            "no MiniMax-H3 snapshot in the HF cache; the template/token half of the fixture needs "
            "tokenizer/ and text_encoder/chat_template.json"
        )
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
    with open(os.path.join(snap, "text_encoder", "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]

    probe = "PROMPT"
    rendered = tk.apply_chat_template(
        [{"role": "user", "content": probe}],
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_str, suffix_str = rendered.split(probe)
    prefix_ids = tk(prefix_str, add_special_tokens=False).input_ids

    with open(os.path.join(snap, "tokenizer", "tokenizer_config.json")) as f:
        tcfg = json.load(f)
    additional = tcfg["additional_special_tokens"]

    return {
        "prefix_str": prefix_str,
        "suffix_str": suffix_str,
        "prefix_ids": prefix_ids,
        "additional_special_tokens": additional,
        "special_ids": {t: tk.convert_tokens_to_ids(t) for t in additional},
        "len_tokenizer": len(tk),
    }


@torch.no_grad()
def main():
    facts = template_facts()
    assert (
        len(facts["prefix_ids"]) == PREFIX
    ), f"shipped template prefix is {len(facts['prefix_ids'])} tokens, fixture assumes {PREFIX}"

    cfg = Qwen3VLTextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KVHEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        max_position_embeddings=512,
    )
    model = Qwen3VLTextModel(cfg).eval()

    # Qwen3 RMSNorm init = ones, which would hide the weight-multiply from the parity test.
    for name, p in model.named_parameters():
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            p.data = 1.0 + 0.1 * torch.randn_like(p.data)

    input_ids = torch.randint(0, VOCAB, (1, SEQ))
    attention_mask = torch.ones_like(input_ids)
    out = model(
        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
    )
    assert len(out.hidden_states) == LAYERS + 1, "hidden_states must be depth+1 (index 0 = embeds)"

    def ctx(idx: int) -> torch.Tensor:
        # `.clone()`: the slice is a VIEW of the same storage as the untrimmed state, and
        # `safetensors` refuses to serialize tensors that share memory.
        return out.hidden_states[idx][:, PREFIX:].clone()

    selected = ctx(SELECT)
    lo, hi = (ctx(i) for i in NEIGHBOURS)

    # A golden whose neighbours are numerically indistinguishable cannot gate an off-by-one.
    for label, other in (("lo", lo), ("hi", hi)):
        d = (selected - other).abs().max().item()
        assert d > 1e-2, f"hidden_states[{SELECT}] vs {label} neighbour differ by only {d:.3e}"
    assert selected.std().item() > 1e-3, "selected context is ~constant"

    tensors = {f"language_model.{k}": v for k, v in model.state_dict().items()}
    tensors["in.input_ids"] = input_ids.to(torch.int32)
    tensors["in.attention_mask"] = attention_mask.to(torch.int32)
    tensors["out.context"] = selected
    tensors[f"out.context_at_{NEIGHBOURS[0]}"] = lo
    tensors[f"out.context_at_{NEIGHBOURS[1]}"] = hi
    # The untrimmed state too, so the prefix-drop itself is separately checkable.
    tensors["out.hidden_untrimmed"] = out.hidden_states[SELECT].clone()
    tensors = {
        k: (v if v.dtype == torch.int32 else v.to(torch.float32)).contiguous()
        for k, v in tensors.items()
    }

    from safetensors.torch import save_file

    meta = {
        "select_hidden": str(SELECT),
        "num_layers": str(LAYERS),
        "prefix_tokens": str(PREFIX),
        "neighbours": ",".join(str(n) for n in NEIGHBOURS),
        "prefix_str": facts["prefix_str"],
        "suffix_str": facts["suffix_str"],
        "prefix_ids": ",".join(str(i) for i in facts["prefix_ids"]),
        "additional_special_tokens": ",".join(facts["additional_special_tokens"]),
        "special_ids": json.dumps(facts["special_ids"]),
        "len_tokenizer": str(facts["len_tokenizer"]),
    }
    path = fixture("mlx-gen-minimax-h3/tests/fixtures/te_context.safetensors")
    save_file(tensors, path, metadata=meta)
    print(f"wrote {path}  ({len(tensors)} tensors, context {tuple(selected.shape)})")
    print(f"  template prefix {facts['prefix_ids']} from the shipped chat_template.json")
    print(f"  <d> -> {facts['special_ids']['<d>']}")


if __name__ == "__main__":
    main()
