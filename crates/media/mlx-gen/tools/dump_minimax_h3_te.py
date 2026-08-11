"""MiniMax-H3 text-encoder (Qwen3-VL-32B) context-extraction parity fixture (sc-17143, corrected by
sc-18741).

Two halves, from two references, and the fixture records which produced which:

* the **tensor** half runs the **transformers** ``Qwen3VLTextModel`` — an independent reference
  graph, and exactly the stack the official conditioner reads ``hidden_states[50]`` off — at TINY
  dims, and dumps the inputs, weights and select-layer hidden states the Rust/MLX port must
  reproduce;
* the **presentation** half is taken from the **official diffusers conditioner**
  (``diffusers.modular_pipelines.minimax_h3.encoders.MiniMaxH3TextEncoderStep``) applied to the real
  shipped tokenizer.

sc-18741: THERE IS NO TEMPLATE PREFIX TO SLICE
-----------------------------------------------

The first version of this script rendered the shipped ``chat_template.json``, measured its 3-token
prefix, and dumped ``hidden_states[SELECT][:, 3:]`` as the golden. The port matched, so parity was
green — but the official conditioner is one line::

    token_ids = components.tokenizer(prompt, add_special_tokens=False)["input_ids"]

with the block's own description reading "the prompt verbatim, with no chat template and no special
tokens". So there is no prefix to slice.

Measured against the real tokenizer, the 3-token slice does land exactly on the
``<|im_start|>user\\n`` boundary for ordinary prompts, so the damage is *not* lost prompt tokens. It
is the 5-token generation cue ``<|im_end|>\\n<|im_start|>assistant\\n`` that nothing ever removed:
the DiT was conditioned on ``prompt + 5 rows of chat-turn control tokens`` (16 rows instead of 11 for
a 9-word prompt). A prompt beginning with whitespace additionally loses a real token, because the
tokenizer merges the template's trailing newline into it.

``chat_template.json`` is present in ``text_encoder/`` only because the component is a byte-identical
copy of ``Qwen/Qwen3-VL-32B-Instruct``, where that file drives chat. Its presence was never evidence
that H3 conditions through it — and sc-17143 could not check, because ``MiniMaxH3`` had not yet
landed on diffusers ``main`` (PR #14355, merged 2026-08-05, in no tagged release).

The golden is therefore the **untrimmed** select-layer state, and the metadata carries the reference
presentation ids for a probe prompt **plus** the ids the old chat-template render would have
produced, so ``tests/te_parity.rs`` can assert the port produces the first and not the second.

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

Run from a venv with torch + transformers:  python3 tools/dump_minimax_h3_te.py
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
SEQ = 12

# The probe the presentation facts are derived from. Deliberately >3 tokens and starting with a
# word the old 3-token prefix slice would have eaten, so the two id vectors differ visibly.
PROBE_PROMPT = "a red fox leaps over a mossy log at dawn"

MODEL_REPO = "models--MiniMaxAI--MiniMax-H3"


def snapshot_dir() -> str | None:
    """The MiniMax-H3 snapshot root, if this machine has one (for the template/token half)."""
    root = hf_hub_cache() / MODEL_REPO / "snapshots"
    if not root.is_dir():
        return None
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    return str(snaps[-1]) if snaps else None


def presentation_facts() -> dict:
    """Derive the PRESENTATION and the special-token ids from the shipped files + the reference.

    This is the half of the fixture that pins configuration no tensor can witness:

    * ``presentation_ids`` is the official conditioner's own call —
      ``tokenizer(prompt, add_special_tokens=False)`` — against the real shipped tokenizer;
    * ``templated_ids`` is what sc-17143's chat-template render produced for the same prompt, kept
      as an explicit NEGATIVE control so the Rust side can assert the port emits the first and never
      the second. Pinning only the current token count would not catch a template coming back with a
      compensating slice;
    * the seven MiniMax specials come from ``tokenizer_config.json``'s
      ``additional_special_tokens``, whose ORDER is the id assignment.
    """
    snap = snapshot_dir()
    if snap is None:
        raise SystemExit(
            "no MiniMax-H3 snapshot in the HF cache; the presentation/token half of the fixture "
            "needs tokenizer/ and text_encoder/chat_template.json"
        )
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))

    # THE reference call. `diffusers.modular_pipelines.minimax_h3.encoders.MiniMaxH3TextEncoderStep`
    # spells it exactly this way, and `MiniMaxH3FL2VATextEncoderStep` /
    # `MiniMaxH3Ref2VATextEncoderStep` spell it the same way for the prompt half of their
    # presentations.
    presentation_ids = tk(PROBE_PROMPT, add_special_tokens=False)["input_ids"]

    # The negative control: sc-17143's rendering, kept so its absence is testable.
    with open(os.path.join(snap, "text_encoder", "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    templated = tk.apply_chat_template(
        [{"role": "user", "content": PROBE_PROMPT}],
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=True,
    )
    templated_ids = tk(templated, add_special_tokens=False).input_ids
    # ...and what the shipped port actually fed the DiT: that render with 3 leading ids dropped.
    sc17143_ids = templated_ids[3:]
    assert presentation_ids != sc17143_ids, "the probe must distinguish the two presentations"

    with open(os.path.join(snap, "tokenizer", "tokenizer_config.json")) as f:
        tcfg = json.load(f)
    additional = tcfg["additional_special_tokens"]

    return {
        "presentation_ids": presentation_ids,
        "templated_ids": templated_ids,
        "sc17143_ids": sc17143_ids,
        "additional_special_tokens": additional,
        "special_ids": {t: tk.convert_tokens_to_ids(t) for t in additional},
        "len_tokenizer": len(tk),
    }


@torch.no_grad()
def main():
    facts = presentation_facts()

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
        # UNTRIMMED — the reference conditioner returns `hidden_states[layer]` whole
        # (`encoders.py`: `return outputs.hidden_states[text_encoder_layer]`). sc-18741.
        return out.hidden_states[idx].clone()

    selected = ctx(SELECT)
    lo, hi = (ctx(i) for i in NEIGHBOURS)
    assert selected.shape[1] == SEQ, "the context keeps one row per presentation token"

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
    tensors = {
        k: (v if v.dtype == torch.int32 else v.to(torch.float32)).contiguous()
        for k, v in tensors.items()
    }

    from safetensors.torch import save_file

    import transformers

    meta = {
        # Which reference produced which half — the sc-18740/18741 methodology requirement.
        "provenance": "official-conditioner",
        "tensor_reference": "transformers.Qwen3VLTextModel",
        "presentation_reference": "diffusers.MiniMaxH3TextEncoderStep",
        "reference_version": f"transformers {transformers.__version__}",
        "select_hidden": str(SELECT),
        "num_layers": str(LAYERS),
        "neighbours": ",".join(str(n) for n in NEIGHBOURS),
        # The presentation contract, and the two things it must NOT be.
        "applies_chat_template": "false",
        "add_special_tokens": "false",
        "probe_prompt": PROBE_PROMPT,
        "presentation_ids": ",".join(str(i) for i in facts["presentation_ids"]),
        "templated_ids": ",".join(str(i) for i in facts["templated_ids"]),
        "sc17143_ids": ",".join(str(i) for i in facts["sc17143_ids"]),
        "additional_special_tokens": ",".join(facts["additional_special_tokens"]),
        "special_ids": json.dumps(facts["special_ids"]),
        "len_tokenizer": str(facts["len_tokenizer"]),
        "story": "sc-17143, corrected by sc-18741",
    }
    path = fixture("mlx-gen-minimax-h3/tests/fixtures/te_context.safetensors")
    save_file(tensors, path, metadata=meta)
    print(f"wrote {path}  ({len(tensors)} tensors, context {tuple(selected.shape)})")
    print(f"  probe {PROBE_PROMPT!r}")
    print(f"    reference presentation ({len(facts['presentation_ids'])} ids): {facts['presentation_ids']}")
    print(f"    sc-17143's templated+sliced ({len(facts['sc17143_ids'])} ids): {facts['sc17143_ids']}")
    print(f"  <d> -> {facts['special_ids']['<d>']}")


if __name__ == "__main__":
    main()
