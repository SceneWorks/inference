"""LTX-2.5 text-encoder → connector-input golden vs the v1.2.0 reference — sc-18770.

sc-18770 wired LTX-2.5's text encoder to the shared Gemma 4 decoder. Everything that file's
tests could assert without an oracle — geometry, dtype, finiteness, non-degeneracy, pad-row
invariance — is a *self-consistency* claim. This golden is the missing external one: the
reference `LTXGemmaTextEncoder` + `EmbeddingsProcessor` from Lightricks/LTX-2 v1.2.0, on the real
**unquantized** `gemma4-12b-with-proj` encoder and the real 2.5 DiT's connectors, over one fixed
prompt, compared absolutely.

The fixture is deliberately **backend-neutral**: it carries the reference's own `input_ids` and
`mask01` so neither backend re-tokenizes, and every tensor is f32. `mlx-gen-ltx` and
`candle-gen-ltx` assert against the same file.

Note on layout. Upstream there is no `connector.safetensors` — that is *our* tier converter's
layout (sc-18775). Upstream, `text_embedding_projection.{video,audio}_aggregate_embed` live in the
packed TE (the "with-proj" in its name) and the two `*_embeddings_connector` stacks live in the DiT
under `model.diffusion_model.`. This script reads both from the upstream files directly, so no
tooling of ours is in the oracle's path.

What is dumped, and in which order:

* `video_features` / `audio_features` — the `FeatureExtractorV2` output, in the tokenizer's own
  **left-padded** order, so `mask01` indexes them directly.
* `video_embeddings` / `audio_embeddings` — the RAW connector output. The connector reorders its
  input to right-padded and replaces the pad tail with learnable registers, so rows
  `0..num_valid` are the valid tokens and the rest are register rows. The reference's
  `create_embeddings` additionally multiplies the video encoding by a binary mask; that product is
  NOT dumped, because the port's `encode_av_with_features` returns the pre-mask tensor and a golden
  must compare like with like (the same choice `ltx_connector_golden.safetensors` made for 2.3).

`normed` (the `[1, 256, 188160]` per-token-RMS concat) is deliberately not dumped — it is ~192 MB
and adds nothing the features do not already pin.

Run:
    LTX2_SRC=<checkout>/packages/ltx-core/src \\
    LTX25_TE_DIR=<HF snapshot root of Lightricks/LTX-2.5> \\
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx25_te_connector_golden.py

`LTX25_TE_DIR` is the snapshot root holding `text_encoders/` and `diffusion_models/`; override the
two files directly with `LTX25_TE_FILE` / `LTX25_DIT_FILE`.

Output (committed): mlx-gen-ltx/tests/fixtures/ltx25_te_connector_golden.safetensors
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from _ltx25_diffvae_ref import REFERENCE_COMMIT, ltx_core_on_path, report, require_finite
from _paths import fixture, require_env

torch.manual_seed(0)

#: The one prompt both backends inherit. Short enough to leave a long pad run at MAX_LEN, which is
#: what makes the padding-mask component of the attention mask observable at all.
PROMPT = "A slow dolly shot across a rain-slicked street at night, neon reflections."

#: The connector prepends 128 learnable registers and refuses a sequence shorter than that count,
#: and requires the length to be a multiple of it. 256 is the length the 2.3 sibling golden was
#: recorded at, so both gates run at the same geometry.
MAX_LEN = 256

TE_GLOB = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
DIT_GLOB = "ltx-2.5-22b-dev-transformer-bf16.safetensors"


def _resolve() -> tuple[Path, Path]:
    """`(unquantized packed TE, bf16 DiT)`, from explicit overrides or the snapshot root."""
    te = os.environ.get("LTX25_TE_FILE")
    dit = os.environ.get("LTX25_DIT_FILE")
    if te and dit:
        return Path(te).expanduser(), Path(dit).expanduser()
    root = Path(
        require_env(
            "LTX25_TE_DIR",
            "the Lightricks/LTX-2.5 snapshot root holding text_encoders/ and diffusion_models/ "
            f"(or set LTX25_TE_FILE + LTX25_DIT_FILE). Needs the UNQUANTIZED {TE_GLOB}.",
        )
    ).expanduser()
    return root / "text_encoders" / TE_GLOB, root / "diffusion_models" / DIT_GLOB


te_path, dit_path = _resolve()
for label, path in (("TE", te_path), ("DiT", dit_path)):
    if not path.is_file():
        raise SystemExit(f"{label} checkpoint not found: {path}")
print(f"[ref] TE  {te_path}")
print(f"[ref] DiT {dit_path}")

ltx_core_on_path()

from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader  # noqa: E402
from ltx_core.text_encoders.gemma import (  # noqa: E402
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaAssets,
    GemmaTextEncoderConfigurator,
    build_gemma_tokenizer,
    convert_to_additive_mask,
    get_gemma_ops,
)

loader = SafetensorsModelStateDictLoader()
dit_metadata = loader.metadata(str(dit_path))
model_version = dit_metadata.get("model_version")
gemma_version = (dit_metadata.get("gemma_source_checkpoint") or {}).get("gemma_version")
print(f"[ref] model_version={model_version!r} gemma_version={gemma_version!r}")
assert gemma_version, "the 2.5 DiT must declare gemma_source_checkpoint.gemma_version"


def _assign_f32(module: torch.nn.Module, sd: dict[str, torch.Tensor], what: str) -> None:
    """`load_state_dict(assign=True)` in f32, popping as it goes.

    The encoder is 26.3 GB of bf16 built on the `meta` device, so the weights must be *assigned*
    rather than copied, and casting the whole dict up front would hold both copies at once.
    """
    keys = list(sd)
    for key in keys:
        sd[key] = sd.pop(key).to(torch.float32)
    missing, unexpected = module.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if "inv_freq" not in k and not k.endswith("embed_scale")]
    assert not missing, f"{what}: weights missing: {missing[:8]}"
    assert not unexpected, f"{what}: checkpoint keys with no home: {unexpected[:8]}"


# --- (1) the reference Gemma 4 text encoder, on the real unquantized checkpoint ----------------
sd_ops, module_ops = get_gemma_ops(str(te_path))
encoder = GemmaTextEncoderConfigurator.with_gemma_model_path(str(te_path)).from_metadata({})
_assign_f32(encoder, loader.load(str(te_path), sd_ops).sd, "gemma text encoder")
for ops in module_ops:
    # `ProcessorLoad` builds the multimodal image/video processor, which upstream needs only for
    # the I2V *enhance* path and which drags in torchvision. `encode` — the only path this golden
    # exercises — never touches `self.processor`, so it is skipped rather than installed.
    if ops.name == "ProcessorLoad":
        continue
    if ops.matcher(encoder):
        encoder = ops.mutator(encoder)
encoder = encoder.to(dtype=torch.float32).eval()

# --- (2) tokenize with the reference tokenizer (LEFT padding, the whole point of the mask) -----
tokenizer = build_gemma_tokenizer(GemmaAssets.load(str(te_path)))
tokenizer.max_length = MAX_LEN
pairs = tokenizer.tokenize_with_weights(PROMPT)["gemma"]
input_ids = torch.tensor([[tok for tok, _ in pairs]], dtype=torch.long)
mask01 = torch.tensor([[int(w) for _, w in pairs]], dtype=torch.long)
assert input_ids.shape == (1, MAX_LEN), f"tokenizer produced {tuple(input_ids.shape)}, want (1, {MAX_LEN})"
num_valid = int(mask01.sum().item())
pads = MAX_LEN - num_valid
assert pads >= 2, f"the prompt must leave at least two padded positions, got {pads}"
assert mask01[0, :pads].sum().item() == 0, "the reference tokenizer must LEFT-pad"
print(f"[tok] {num_valid} valid tokens, {pads} left-pad positions")

# --- (3) hidden states, exactly as `LTXGemmaTextEncoder.encode` produces them ------------------
with torch.no_grad():
    outputs = encoder.model.model(input_ids=input_ids, attention_mask=mask01, output_hidden_states=True)
hidden_states = tuple(outputs.hidden_states)
del outputs
print(f"[gemma] {len(hidden_states)} hidden states, each {tuple(hidden_states[0].shape)}")

# --- (4) the reference feature extractor + both connectors -------------------------------------
processor = EmbeddingsProcessorConfigurator.with_gemma_model_path(str(te_path)).from_metadata(dit_metadata)
proj_sd = loader.load(str(te_path), EMBEDDINGS_PROCESSOR_KEY_OPS).sd
conn_sd = loader.load(str(dit_path), EMBEDDINGS_PROCESSOR_KEY_OPS).sd
_assign_f32(processor, {**proj_sd, **conn_sd}, "embeddings processor")
processor = processor.to(dtype=torch.float32).eval()

with torch.no_grad():
    video_features, audio_features = processor.feature_extractor(hidden_states, mask01, "left")
    additive = convert_to_additive_mask(mask01, video_features.dtype)
    video_embeddings, audio_embeddings, _ = processor.create_embeddings(video_features, audio_features, additive)

tensors = {
    "input_ids": input_ids.to(torch.int32).contiguous(),
    "mask01": mask01.to(torch.int32).contiguous(),
    "video_features": video_features.to(torch.float32).contiguous(),
    "audio_features": audio_features.to(torch.float32).contiguous(),
    "video_embeddings": video_embeddings.to(torch.float32).contiguous(),
    "audio_embeddings": audio_embeddings.to(torch.float32).contiguous(),
}
for name, tensor in tensors.items():
    require_finite(name, tensor)
    report(name, tensor)

meta = {
    "story": "sc-18770",
    "reference": (
        f"Lightricks/LTX-2 @ {REFERENCE_COMMIT} (v1.2.0) "
        "Gemma4 TE + text_embedding_projection + Embeddings1DConnector"
    ),
    "prompt": PROMPT,
    "seq": str(MAX_LEN),
    "num_valid": str(num_valid),
    "dim": str(video_features.shape[-1]),
    "audio_dim": str(audio_features.shape[-1]),
    "gemma_version": str(gemma_version),
    "te_checkpoint": te_path.name,
    "dit_checkpoint": dit_path.name,
    "dtype": "f32",
    "embedding_order": (
        "video_embeddings/audio_embeddings are the RAW connector output: rows 0..num_valid are the "
        "valid tokens (the connector reorders left-padding to the front), the rest are learnable "
        "registers. video_features/audio_features keep the tokenizer's left-padded order."
    ),
    "backend_neutral": "consumed by mlx-gen-ltx and candle-gen-ltx (sc-18770)",
}

out = fixture("mlx-gen-ltx/tests/fixtures/ltx25_te_connector_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out, metadata=meta)
print(f"wrote {out}")
