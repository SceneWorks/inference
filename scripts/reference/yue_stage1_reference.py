#!/usr/bin/env python3
"""Regenerate the YuE stage-1 decode parity fixture (sc-19380, epic sc-19373).

`crates/audio/candle-audio-yue/src/stage1/parity.rs` checks the candle-llm driven stage 1
(`Stage1Lm`) against this script's output, token for token. The fixture is **committed**; the
gate needs no weights and no environment variable.

What the reference is
---------------------
YuE-v1 `inference/infer.py` (github.com/multimodal-art-projection/YuE @ YuE-v1
6d4f0b1f8ce6a55fb2392e959394c46e07ee334d) runs stage 1 as one Hugging Face `generate` call per
lyric segment. This script makes the same call, with the same arguments, on a tiny
`LlamaForCausalLM`:

* ``guidance_scale`` 1.5 for the first segment and 1.2 after — Hugging Face runs it through
  ``UnbatchedClassifierFreeGuidanceLogitsProcessor`` (the unconditional stream is the prompt's last
  token plus what is generated after it);
* ``top_p=0.93``, ``temperature=1.0``, ``repetition_penalty=1.1``, ``min_new_tokens=100``,
  ``eos_token_id=pad_token_id=<EOA>``, and no ``top_k`` (so the library default of 50 applies);
* a forced ``<EOA>`` when the budget, not the model, ends a segment.

Two pieces come from the epic's design reference, YuE-exllamav2
(github.com/sgsdxzy/YuE-exllamav2 @ a644036251c96613e0c4bb192a9309bfc046dafd,
``src/yue/infer_stage1.py``), because the epic specifies them:

* the sampling allow-list ``[32002] + [45334, 56721]`` (``gen_settings.allow_tokens``) in place of
  infer.py's ``BlockTokenRangeProcessor(0, 32002)``;
* the "smart context" ``shorten_input`` that drops the oldest ``[start_of_segment]`` block when the
  sequence outgrows ``cache_size - max_new_tokens - 1`` (infer.py keeps the last tokens instead).

The random draw
---------------
Torch's RNG cannot be reproduced in Rust, so the parity is made exact **by construction**:
``torch.multinomial`` is replaced, for the duration of each ``generate`` call, by an inverse-CDF draw
from the SplitMix64 stream candle-llm's host sampler uses (same constants, same 24-bit ``next_f32``,
same candidate order: descending probability, ties to the lower id). Every Hugging Face logits
processor still runs unmodified; only the final categorical draw is substituted. No claim of
parity with torch's own RNG is made.

The weights
-----------
The tiny model's weights are a pure function of the tensor name (SplitMix64 seeded with the name's
FNV-1a-64 hash, ``(2u - 1) * scale`` in float32), so the Rust test rebuilds them bit-identically
instead of reading a committed checkpoint; ``tensorSums`` in the fixture pins the generator. Hidden
channel 0 is a constant "bias channel" (the embedding writes a constant into it and no layer writes
to it), which lets the LM head lean towards codebook-0 tokens and ``<EOA>`` the way the trained
model does, so the fixture exercises both segment endings and a clean track split.

Also recorded: the smart-context rule on crafted sequences and the reference ``save`` split
(`infer.py` lines 227-246, run through the real ``CodecManipulator``) on crafted raw outputs.

Regenerating (the shared CPU reference environment, see its README)::

    /Users/michael/.cache/sceneworks-yue-ref/venv/bin/python \
        scripts/reference/yue_stage1_reference.py dump

The reference environment is ``~/.cache/sceneworks-yue-ref`` (torch 2.14.0 CPU, transformers
5.17.0, the YuE-v1 clone). Everything runs in float32 on the CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "crates/audio/candle-audio-yue/tests/fixtures/yue_stage1_parity.json"
YUE_INFERENCE = Path.home() / ".cache/sceneworks-yue-ref/yue/inference"

SOA, EOA, XCODEC_SEP = 32001, 32002, 32016
CODEC_OFFSET, ALLOW_MAX = 45334, 56721
START_OF_SEGMENT = [518, 2962, 29918, 974, 29918, 28192, 29962]
END_OF_SEGMENT = [518, 355, 29918, 974, 29918, 28192, 29962]

CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 56832,
    "max_position_embeddings": 480,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_act": "silu",
    "bos_token_id": 1,
    "eos_token_id": 2,
    "torch_dtype": "float32",
}
# `(2u - 1) * scale` per weight kind; norms are ones.
SCALES = {"embed": 1.0, "attn": 0.5, "mlp": 0.5, "lm_head": 1.0}
# The bias channel: embedding column 0 is this constant; the LM head's column 0 is
# `CB0_BIAS` on codebook-0 rows, `EOA_BIAS` on `<EOA>`, 0 elsewhere.
BIAS = {"embedChannel": 3.0, "cb0": 4.5, "eoa": 5.9}

DECODE = {
    "maxNewTokens": 130,
    "minNewTokens": 100,
    "topP": 0.93,
    "topK": 50,
    "temperature": 1.0,
    "repetitionPenalty": 1.1,
    "guidance": [1.5, 1.2],
    "contextLimit": CONFIG["max_position_embeddings"],
    "seed": 42,
}

INC = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1


class SplitMix64:
    """candle-llm `primitives::sampler::SplitMix64`, verbatim."""

    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + INC) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def next_f32(self) -> np.float32:
        return np.float32(self.next_u64() >> 40) / np.float32(1 << 24)


def splitmix_uniform(seed: int, n: int) -> np.ndarray:
    """The first `n` `next_f32` draws of `SplitMix64(seed)`, vectorised (counter form)."""
    i = np.arange(1, n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + i * np.uint64(INC)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(40)).astype(np.float32) / np.float32(1 << 24)


def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for b in name.encode():
        h = ((h ^ b) * 0x100000001B3) & MASK64
    return h


def tensor_specs(cfg: dict) -> list[tuple[str, tuple[int, ...], str]]:
    h, i, v = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    kv = cfg["num_key_value_heads"] * (h // cfg["num_attention_heads"])
    specs = [("model.embed_tokens.weight", (v, h), "embed")]
    for layer in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{layer}"
        specs += [
            (f"{p}.input_layernorm.weight", (h,), "norm"),
            (f"{p}.self_attn.q_proj.weight", (h, h), "attn"),
            (f"{p}.self_attn.k_proj.weight", (kv, h), "attn"),
            (f"{p}.self_attn.v_proj.weight", (kv, h), "attn"),
            (f"{p}.self_attn.o_proj.weight", (h, h), "attn"),
            (f"{p}.post_attention_layernorm.weight", (h,), "norm"),
            (f"{p}.mlp.gate_proj.weight", (i, h), "mlp"),
            (f"{p}.mlp.up_proj.weight", (i, h), "mlp"),
            (f"{p}.mlp.down_proj.weight", (h, i), "mlp"),
        ]
    specs += [("model.norm.weight", (h,), "norm"), ("lm_head.weight", (v, h), "lm_head")]
    return specs


def build_weights(cfg: dict) -> dict[str, np.ndarray]:
    out = {}
    for name, shape, kind in tensor_specs(cfg):
        n = int(np.prod(shape))
        if kind == "norm":
            w = np.ones(n, dtype=np.float32)
        else:
            u = splitmix_uniform(fnv1a64(name), n)
            w = (u * np.float32(2.0) - np.float32(1.0)) * np.float32(SCALES[kind])
        out[name] = w.reshape(shape)
    # The bias channel (see the module docstring).
    out["model.embed_tokens.weight"][:, 0] = np.float32(BIAS["embedChannel"])
    for name in out:
        if name.endswith(("o_proj.weight", "down_proj.weight")):
            out[name][0, :] = np.float32(0.0)
    head = out["lm_head.weight"]
    head[:, 0] = np.float32(0.0)
    head[CODEC_OFFSET : CODEC_OFFSET + 1024, 0] = np.float32(BIAS["cb0"])
    head[EOA, 0] = np.float32(BIAS["eoa"])
    return out


def prompt_blocks() -> list[list[int]]:
    """Four synthetic segment prompt blocks in the reference layout: segment 0 carries a text head
    before `[start_of_segment]`; later blocks open with `[end_of_segment][start_of_segment]`; every
    block ends `<SOA><xcodec>`. Text ids are drawn below the special range."""
    rng = SplitMix64(0x5E6)
    text = lambda n: [100 + rng.next_u64() % 30000 for _ in range(n)]  # noqa: E731
    blocks = [text(24) + START_OF_SEGMENT + text(9) + [SOA, XCODEC_SEP]]
    for n in (7, 11, 6):
        blocks.append(END_OF_SEGMENT + START_OF_SEGMENT + text(n) + [SOA, XCODEC_SEP])
    return blocks


def shorten_input(seq: list[int], max_context: int) -> list[int]:
    """YuE-exllamav2 `Stage1Pipeline.shorten_input`, verbatim but for the tensor plumbing."""
    import torch

    seq_t = torch.tensor([seq])
    pattern = torch.tensor(START_OF_SEGMENT)
    pattern_length = pattern.numel()
    while seq_t.shape[-1] > max_context:
        windows = seq_t[0].unfold(0, pattern_length, 1)
        matches = (windows == pattern).all(dim=1)
        match_indices = torch.nonzero(matches).flatten()
        if match_indices.numel() < 3:
            return seq_t[:, -max_context:][0].tolist()
        first_segment_start = match_indices[0].item()
        second_segment_start = match_indices[1].item()
        seq_t = torch.cat((seq_t[:, :first_segment_start], seq_t[:, second_segment_start:]), dim=-1)
    return seq_t[0].tolist()


class AllowRange:
    """The stage-1 allow-list `[EOA] + [CODEC_OFFSET, ALLOW_MAX]` as a logits processor."""

    def __call__(self, input_ids, scores):
        import torch

        keep = torch.zeros(scores.shape[-1], dtype=torch.bool)
        keep[EOA] = True
        keep[CODEC_OFFSET : ALLOW_MAX + 1] = True
        return scores.masked_fill(~keep, -float("inf"))


def splitmix_multinomial(rng: SplitMix64):
    """A `torch.multinomial(probs, 1)` replacement: candle-llm's inverse-CDF draw."""
    import torch

    def draw(probs, num_samples=1, *args, **kwargs):
        assert probs.shape[0] == 1 and num_samples == 1
        p = probs[0].detach().to(torch.float32).numpy()
        order = sorted(np.nonzero(p > 0)[0].tolist(), key=lambda i: (-p[i], i))
        total = np.float32(0.0)
        for i in order:
            total = np.float32(total + p[i])
        target = np.float32(rng.next_f32() * total)
        for i in order:
            target = np.float32(target - p[i])
            if target <= 0:
                return torch.tensor([[i]])
        return torch.tensor([[order[-1]]])

    return draw


def build_model(weights: dict[str, np.ndarray]):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = {k: v for k, v in CONFIG.items() if k not in ("architectures", "model_type", "torch_dtype")}
    model = LlamaForCausalLM(LlamaConfig(**cfg, attn_implementation="eager")).to(torch.float32)
    state = {k: torch.from_numpy(v.copy()) for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if "rotary" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    return model.eval()


def render(model, prompts: list[list[int]], guidance: list[float] | None) -> dict:
    """One stage-1 render, segment by segment, the way infer.py drives `generate`."""
    import torch
    import transformers.generation.logits_process as lp
    from transformers import LogitsProcessorList

    rng = SplitMix64(DECODE["seed"])
    first_scores: list[np.ndarray] = []
    capture = {"pending": False}
    penalty_call = lp.RepetitionPenaltyLogitsProcessor.__call__

    def recording(self, input_ids, scores):
        # The penalty is the first processor after CFG: its input is the CFG-mixed scores (or the
        # raw logits without guidance) — what `Stage1Lm` records as a segment's first scores.
        if capture["pending"]:
            first_scores.append(scores[0].detach().clone().numpy())
            capture["pending"] = False
        return penalty_call(self, input_ids, scores)

    max_new = DECODE["maxNewTokens"]
    max_context = DECODE["contextLimit"] - max_new - 1
    seq: list[int] = []
    segments, ended_by, windows = [], [], []
    real_multinomial = torch.multinomial
    lp.RepetitionPenaltyLogitsProcessor.__call__ = recording
    torch.multinomial = splitmix_multinomial(rng)
    try:
        for i, block in enumerate(prompts):
            seq = seq + block
            full = seq if len(seq) <= max_context else shorten_input(seq, max_context)
            windows.append(len(full))
            kwargs = dict(
                max_new_tokens=max_new,
                min_new_tokens=DECODE["minNewTokens"],
                do_sample=True,
                top_p=DECODE["topP"],
                temperature=DECODE["temperature"],
                repetition_penalty=DECODE["repetitionPenalty"],
                eos_token_id=EOA,
                pad_token_id=EOA,
                logits_processor=LogitsProcessorList([AllowRange()]),
            )
            if guidance is not None:
                kwargs["guidance_scale"] = guidance[0] if i == 0 else guidance[1]
            capture["pending"] = True
            with torch.no_grad():
                out = model.generate(input_ids=torch.tensor([full]), **kwargs)
            gen = out[0, len(full) :].tolist()
            if gen[-1] != EOA:
                gen.append(EOA)
                ended_by.append("budget")
            else:
                ended_by.append("eoa")
            segments.append(gen)
            seq = seq + gen
    finally:
        torch.multinomial = real_multinomial
        lp.RepetitionPenaltyLogitsProcessor.__call__ = penalty_call

    split = reference_split(seq, 0)
    tops = []
    for row in first_scores:
        ids = np.argsort(-row, kind="stable")[:64]
        tops.append([[int(t), float(row[t])] for t in ids])
    return {
        "segments": segments,
        "endedBy": ended_by,
        "windowLengths": windows,
        "sequence": seq,
        "split": split,
        "firstScoresTop64": tops,
    }


def reference_split(ids: list[int], range_begin: int) -> dict:
    """infer.py lines 227-246 (the `save` split), run through the real `CodecManipulator`."""
    sys.path.insert(0, str(YUE_INFERENCE))
    from codecmanipulator import CodecManipulator
    from einops import rearrange

    codectool = CodecManipulator("xcodec", 0, 1)
    ids = np.array(ids)
    try:
        soa_idx = np.where(ids == SOA)[0].tolist()
        eoa_idx = np.where(ids == EOA)[0].tolist()
        if len(soa_idx) != len(eoa_idx):
            raise ValueError(f"invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}")
        vocals, instrumentals = [], []
        for i in range(range_begin, len(soa_idx)):
            codec_ids = ids[soa_idx[i] + 1 : eoa_idx[i]]
            if codec_ids[0] == 32016:
                codec_ids = codec_ids[1:]
            codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
            vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
            vocals.append(vocals_ids)
            instrumentals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
            instrumentals.append(instrumentals_ids)
        vocals = np.concatenate(vocals, axis=1)
        instrumentals = np.concatenate(instrumentals, axis=1)
    except (ValueError, AssertionError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"vocals": vocals[0].tolist(), "instrumental": instrumentals[0].tolist()}


def split_cases() -> list[dict]:
    c = lambda code: CODEC_OFFSET + code  # noqa: E731
    head = [7, 8, 9]
    cases = {
        "cot_two_segments_odd_lengths": (
            head + [SOA, XCODEC_SEP, c(1), c(2), c(3), c(4), c(5), EOA]
            + [11, SOA, XCODEC_SEP, c(6), c(7), c(8), EOA],
            0,
        ),
        "icl_reference_pair_is_skipped": (
            head + [SOA, XCODEC_SEP, c(900), c(901), c(902), c(903), EOA, 12]
            + [SOA, XCODEC_SEP, c(10), c(20), c(30), c(40), EOA],
            1,
        ),
        "no_separator_is_kept": (head + [SOA, c(5), c(6), c(7), c(8), EOA], 0),
        "mid_track_token_beyond_codebook0": (
            head + [SOA, XCODEC_SEP, c(1), c(2), c(3 * 1024 + 5), c(4), EOA],
            0,
        ),
        "unpaired_soa": (head + [SOA, XCODEC_SEP, c(1), c(2), EOA, SOA, c(3)], 0),
        "track_starts_beyond_codebook0": (head + [SOA, XCODEC_SEP, c(1024), c(1), EOA], 0),
    }
    return [
        {"name": name, "raw": raw, "skipPairs": skip, **reference_split(raw, skip)}
        for name, (raw, skip) in cases.items()
    ]


def shorten_cases() -> list[dict]:
    s, e = START_OF_SEGMENT, END_OF_SEGMENT
    three = [1, 2] + s + [10, 11, 12] + e + s + [20, 21] + e + s + [30]
    four = three + e + s + [40, 41]
    cases = [
        ("drops_the_oldest_block", three, len(three) - 3),
        ("drops_blocks_until_it_fits", four, 20),
        ("falls_back_to_the_last_tokens", [1] + s + [2, 3] + e + s + [4, 5, 6], 9),
        ("fits_untouched", three, len(three)),
    ]
    return [
        {"name": n, "seq": seq, "maxContext": m, "out": shorten_input(seq, m)} for n, seq, m in cases
    ]


def dump() -> None:
    import torch
    import transformers

    weights = build_weights(CONFIG)
    model = build_model(weights)
    prompts = prompt_blocks()
    fixture = {
        "schema": 1,
        "producer": "scripts/reference/yue_stage1_reference.py",
        "reference": {
            "yue": "multimodal-art-projection/YuE@6d4f0b1f8ce6a55fb2392e959394c46e07ee334d (YuE-v1 inference/infer.py)",
            "exllamav2": "sgsdxzy/YuE-exllamav2@a644036251c96613e0c4bb192a9309bfc046dafd (src/yue/infer_stage1.py: allow-list, shorten_input)",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "dtype": "float32",
        },
        "config": CONFIG,
        "scales": SCALES,
        "bias": BIAS,
        "tensorSums": {k: float(np.sum(v, dtype=np.float64)) for k, v in weights.items()},
        "decode": DECODE,
        "prompts": prompts,
        "runs": {
            "guidance": render(model, prompts, DECODE["guidance"]),
            "noGuidance": render(model, prompts, None),
        },
        "shortenCases": shorten_cases(),
        "splitCases": split_cases(),
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    for name, run in fixture["runs"].items():
        lens = [len(s) for s in run["segments"]]
        print(f"{name}: segment lengths {lens} ended by {run['endedBy']} windows {run['windowLengths']}")
        print(f"  split: {'error' if 'error' in run['split'] else len(run['split']['vocals'])} frames")
    print(f"wrote {FIXTURE.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["dump"])
    parser.parse_args()
    dump()


if __name__ == "__main__":
    main()
