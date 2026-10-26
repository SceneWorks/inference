"""Dump the FLUX 3 Image Klein controlled-fixture goldens for the native MLX port (sc-26234).

Everything runs the FROZEN upstream modules (see `_flux3_common.py`) on CPU, with seeded miniature
geometry that keeps every architectural switch of the release — the per-mode early blocks, the
shared single-stream blocks over the per-sample `[image, image_cond, text]` concatenation, the
4-axis RoPE, the q/k RMSNorm, the parallel attention + SwiGLU block, the target-time (flow-map)
conditioning, the `LastLayer` head, the padded Qwen3-VL window with its layer stacking, and the
flow-map Euler sampler — only the widths shrink.

The text encoder is driven through upstream's REAL `TextEncoder.encode` against a miniature
`Qwen3VLForConditionalGeneration` + WordLevel tokenizer snapshot carrying the PINNED chat template,
so the template rendering, the single-string tokenization, the pad-to-multiple window, the pad
position ids, the padded attention and the selected-layer stacking are all upstream's own code.
The tower runs in bf16, as released; the DiT and sampler goldens run in f32 (the oracle's CPU path,
where upstream's `torch.autocast("cuda")` is a no-op) plus one bf16 DiT forward under
`torch.autocast("cpu", bfloat16)`, the CPU twin of the release's policy.

Writes `mlx-gen-flux3/tests/fixtures/`:

* `tiny-snapshot/config.json`, `tiny-snapshot/model.safetensors` — the package, in the released
  layout (`config.json` schema of the pinned package, bf16 DiT with upstream key names).
* `tiny-snapshot/text_encoder/{config.json, model.safetensors, tokenizer.json, tokenizer_config.json,
  chat_template.json}` — the encoder resource, in the Qwen3-VL snapshot layout (`save_pretrained`).
* `flux3_text_golden.safetensors` — `TextEncoder.encode` on four prompts: window ids / mask /
  positions and the stacked `[L, layers·hidden]` conditioning.
* `flux3_dit_golden.safetensors` — one single-sample and one two-sample flow-map forward in f32,
  and the single-sample forward under bf16 autocast.
* `flux3_e2e_golden.safetensors` — `denoise_flow_map` end to end from injected noise (4 steps),
  the decoded image and its RGB8 conversion, through the shared miniature autoencoder; plus the
  same loop under bf16 autocast (`tokens_bf16`, the release's sampling policy) with the oracle's
  own bf16-vs-f32 distance in the metadata (`e2e_bf16_distance`, relative to peak).

Inputs: `FLUX3_SRC` (the frozen source tree) and `FLUX3_PACKAGE` (the pinned HF package snapshot
directory — only its `text_encoder/chat_template.json` and `config.json` are read, sha256-verified).
Run: `FLUX3_SRC=… FLUX3_PACKAGE=… python -I tools/dump_flux3_golden.py` (see `_flux3_common.py`).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

# `python -I` keeps the script directory off `sys.path`; the shared helpers live beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _flux3_common import (  # noqa: E402
    AE_LATENT_CHANNELS,
    DIT_AXES_DIM,
    DIT_DEPTH,
    DIT_DEPTH_SINGLE,
    DIT_HEADS,
    DIT_HIDDEN,
    DIT_IN_CHANNELS,
    DIT_MLP_RATIO,
    DIT_THETA,
    ENCODER_DIR,
    FIXTURE_DIR,
    FLUX3_COMMIT,
    FLUX3_PACKAGE_REVISION,
    OUTPUT_LAYERS,
    PAD_MULTIPLE,
    SNAPSHOT,
    TEXT_DIM,
    TEXT_LAYERS,
    TOWER_SCALE,
    import_upstream,
    patch_dit_constants,
    save_safetensors,
    tiny_autoencoder,
    write_autoencoder_fixture,
    write_json,
)

import_upstream()

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen3VLForConditionalGeneration  # noqa: E402
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig  # noqa: E402

torch.manual_seed(0)

# sha256 of the pinned package files this oracle reads (revision FLUX3_PACKAGE_REVISION).
PINNED_PACKAGE_SHA256 = {
    "config.json": "c4fc638a9618e8ddc25f013c009be4d2aa9bd631d89dcf45149a377dd472c67d",
}

# ---- miniature text window (the release: pad to 80, clamp 4096) -----------------------------------
MAX_LENGTH = 24

# A WordLevel vocabulary small enough for the tiny tower; every template word that is not listed
# maps to [UNK], which is fine — the ids only have to be the ones upstream's own tokenizer produces.
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
WORDS = ["[UNK]", "system", "user", "assistant", "a", "red", "fox", "in", "the", "snow", "golden",
         "hour", ",", ":", "of", "and", "cat", "blue", "sky", "with", "hat", "on", "roof", "sun"]
PROMPTS = {
    # 7 real tokens (prefix 2 + 6 words + suffix 3 = 11) -> padded to 16
    "short": "a red fox in the snow",
    # 2 + 12 + 3 = 17 real tokens -> padded to 24 (the clamp, with no truncation)
    "exact": "a red fox in the snow , a cat on a roof",
    # over the clamp: 2 + 25 + 3 > 24 -> truncated to 24, no pad
    "long": "a red fox in the snow , a cat on a roof with a hat , golden hour , blue sky and sun of the",
    # the template alone: 5 real tokens -> padded to 8
    "empty": "",
}


def package_root() -> Path:
    root = Path(os.environ["FLUX3_PACKAGE"]) if os.environ.get("FLUX3_PACKAGE") else None
    if root is None:
        raise SystemExit("FLUX3_PACKAGE is required: the pinned black-forest-labs/flux-3-image-klein snapshot")
    for rel, want in PINNED_PACKAGE_SHA256.items():
        got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"FLUX3_PACKAGE/{rel}: sha256 {got} != pinned {want}")
    return root


def bf16_round_(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(p.to(torch.bfloat16).to(torch.float32))


def write_encoder_snapshot(package: Path) -> str:
    """The miniature Qwen3-VL snapshot: a WordLevel tokenizer carrying the PINNED chat template
    (so `apply_chat_template` is the release's own rendering), and a full-depth 36-layer tower."""
    if ENCODER_DIR.exists():
        shutil.rmtree(ENCODER_DIR)
    ENCODER_DIR.mkdir(parents=True)
    vocab = {tok: i for i, tok in enumerate(SPECIALS + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    chat_template = json.loads((package / "text_encoder" / "chat_template.json").read_text(encoding="utf-8"))[
        "chat_template"
    ]
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
        chat_template=chat_template,
    )
    fast.padding_side = "right"
    fast.save_pretrained(str(ENCODER_DIR))
    # The pinned template renders a single user turn + generation prompt to exactly the two
    # literals the native window uses (gen_core::flux3::CHAT_PREFIX / CHAT_SUFFIX).
    rendered = fast.apply_chat_template(
        [{"role": "user", "content": "PROMPT"}], tokenize=False, add_generation_prompt=True
    )
    assert rendered == "<|im_start|>user\nPROMPT<|im_end|>\n<|im_start|>assistant\n", rendered

    cfg = Qwen3VLConfig(
        text_config={
            "vocab_size": len(vocab),
            "hidden_size": TEXT_DIM,
            "intermediate_size": 48,
            "num_hidden_layers": TEXT_LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5_000_000.0,
            "rope_scaling": {"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
            "attention_bias": False,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": True,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": TEXT_DIM,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        tie_word_embeddings=True,
    )
    model = Qwen3VLForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(TOWER_SCALE * torch.randn_like(p))
    bf16_round_(model)
    model.save_pretrained(str(ENCODER_DIR), safe_serialization=True)
    return hashlib.sha256(chat_template.encode("utf-8")).hexdigest()


class _StubProcessor:
    """What upstream's `TextEncoder.__init__` reads off `AutoProcessor`: `.tokenizer` and
    `.apply_chat_template`. The miniature snapshot has no image processor, so the stub hands the
    tokenizer's own rendering through — the same `apply_chat_template` the real processor
    delegates to for a text-only message list."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path):
        return cls(PreTrainedTokenizerFast.from_pretrained(path))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )


def tiny_dit_config() -> dict:
    """The miniature package `config.json` in the pinned package's schema."""
    return {
        "format_version": 1,
        "model": "flux-3-image-klein-tiny-fixture",
        "quantization": None,
        "torch_dtype": "bfloat16",
        "dit": {
            "hidden": DIT_HIDDEN,
            "heads": DIT_HEADS,
            "depth": DIT_DEPTH,
            "depth_single": DIT_DEPTH_SINGLE,
            "mlp_ratio": DIT_MLP_RATIO,
            "axes_dim": list(DIT_AXES_DIM),
            "theta": DIT_THETA,
            "in_channels": dict(DIT_IN_CHANNELS),
            "out_modes": ["image"],
            "vec_in_dim": None,
            "image_rope_scale": 0.5,
            "flow_map": True,
        },
        "text_encoder": {
            "path": "text_encoder",
            "tokenizer": "text_encoder",
            "layers": list(OUTPUT_LAYERS),
            "pad_multiple": PAD_MULTIPLE,
            "max_tokens": MAX_LENGTH,
            "dtype": "bfloat16",
        },
        "vae": {"path": "ae.safetensors", "dtype": "float32"},
        "sampler": {"steps": 4, "guidance": None, "alpha": 6.93, "size": [64, 64]},
    }


def tiny_dit(dit_module):
    """The seeded miniature flow-map DiT through upstream's `DiT(flow_map=True)` (module constants
    patched), every parameter perturbed away from its init."""
    patch_dit_constants(dit_module)
    g = torch.Generator().manual_seed(2634)
    model = dit_module.DiT(flow_map=True).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".scale"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
    return model


def write_dit_snapshot(model) -> None:
    """`tiny-snapshot/model.safetensors` (bf16, as released) + `config.json`."""
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous().to(torch.bfloat16) for k, v in model.state_dict().items()}
    save_file(state, str(SNAPSHOT / "model.safetensors"), metadata={"format": "pt"})
    print(f"wrote {SNAPSHOT / 'model.safetensors'} ({len(state)} tensors)")
    write_json(SNAPSHOT / "config.json", tiny_dit_config())


def load_dit(dit_module, checkpoint_module, dtype: torch.dtype):
    """`checkpoint.load_checkpoint(DiT(flow_map), tiny package, cpu, dtype)` — upstream's loader
    over the written snapshot, so the goldens see exactly the bf16-rounded parameters the port
    loads."""
    patch_dit_constants(dit_module)
    with torch.device("meta"):
        dit = dit_module.DiT(flow_map=True)
    return checkpoint_module.load_checkpoint(dit, str(SNAPSHOT), "cpu", dtype)


def dump_text(text_encoder_module) -> tuple[dict, dict]:
    text_encoder_module.AutoProcessor = _StubProcessor
    text_encoder_module.OUTPUT_LAYERS = tuple(OUTPUT_LAYERS)
    text_encoder_module.MAX_LENGTH = MAX_LENGTH
    text_encoder_module.PAD_MULTIPLE = PAD_MULTIPLE
    enc = text_encoder_module.TextEncoder(str(ENCODER_DIR), device="cpu")
    tok = enc.processor.tokenizer
    out = {"pad_id": torch.tensor([tok.pad_token_id], dtype=torch.int32)}
    encodings = {}
    for name, prompt in PROMPTS.items():
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        full_ids = tok(text, return_tensors="pt")["input_ids"][0]
        real = min(int(full_ids.shape[0]), MAX_LENGTH)
        with torch.no_grad():
            tokens, ids = enc.encode(prompt)
        length = int(tokens.shape[0])
        inputs = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        mask = inputs["attention_mask"][0]
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        assert int(mask.sum()) == real, (name, int(mask.sum()), real)
        out[f"{name}/input_ids"] = inputs["input_ids"][0].to(torch.int32)
        out[f"{name}/mask"] = mask.to(torch.int32)
        out[f"{name}/positions"] = positions.to(torch.int32)
        out[f"{name}/tokens"] = tokens.float()
        out[f"{name}/ids"] = ids.float()
        encodings[name] = (tokens, ids)
        print(f"text {name}: real={real} length={length} ids={inputs['input_ids'][0].tolist()}")
    return out, encodings


def pack_t2i(pipeline_module, text, image_tokens, h, w):
    ids = pipeline_module.scaled_ids(pipeline_module.latent_ids(h, w, "cpu"))
    return pipeline_module.pack([[text, None, (image_tokens, ids)]]), ids


def as_f32(packed):
    """The f32 oracle: upstream's `pack` builds bf16 data/timesteps for the CUDA autocast path; the
    f32 run (no autocast on CPU) feeds the same bf16-rounded VALUES as f32, which is exactly what
    the native `Precision::Fp32` path computes with (its Linears cast to f32, its timestep is
    f32)."""
    packed.data = {m: v.float() for m, v in packed.data.items()}
    packed.timesteps = {m: v.float() for m, v in packed.timesteps.items()}
    return packed


def dump_dit(dit_module, checkpoint_module, pipeline_module, encodings) -> dict:
    g = torch.Generator().manual_seed(3)
    out = {}
    # --- single sample, f32 ---
    dit = load_dit(dit_module, checkpoint_module, torch.float32)
    h, w = 2, 3
    x = torch.randn((h * w, AE_LATENT_CHANNELS), generator=g).to(torch.bfloat16).float()
    t, t_ref = 0.9163, 0.78
    text = encodings["short"]
    packed, ids = pack_t2i(pipeline_module, text, x, h, w)
    as_f32(packed)
    packed.timesteps["image"].fill_(t)
    with torch.no_grad():
        velocity = dit(packed, timesteps_ref=torch.tensor([t_ref]))
    out["single/x"] = x
    out["single/image_ids"] = ids
    out["single/t"] = torch.tensor([t])
    out["single/t_ref"] = torch.tensor([t_ref])
    out["single/velocity"] = velocity.float()
    print(f"dit single: velocity {tuple(velocity.shape)} mean|v|={velocity.abs().mean():.4f}")
    # --- two samples (different prompts, grids and times), f32: the packed bounds law ---
    x2 = torch.randn((3 * 2, AE_LATENT_CHANNELS), generator=g).to(torch.bfloat16).float()
    ids2 = pipeline_module.scaled_ids(pipeline_module.latent_ids(3, 2, "cpu"))
    t2, t2_ref = 0.5, 0.31
    text2 = encodings["empty"]
    packed2 = as_f32(pipeline_module.pack([[text, None, (x, ids)], [text2, None, (x2, ids2)]]))
    # Upstream fills one timestep for the whole image mode; a per-sample time is written per
    # span so the two-sample law under test is the one the CFG pack (sc-26239) will use too.
    (a0, b0), (a1, b1) = packed2.mode_indices["image"]
    packed2.timesteps["image"][a0:b0] = t
    packed2.timesteps["image"][a1:b1] = t2
    with torch.no_grad():
        velocity2 = dit(packed2, timesteps_ref=torch.tensor([t_ref, t2_ref]))
    out["pair/x2"] = x2
    out["pair/image_ids2"] = ids2
    out["pair/t2"] = torch.tensor([t2])
    out["pair/t2_ref"] = torch.tensor([t2_ref])
    out["pair/velocity"] = velocity2.float()
    assert torch.allclose(velocity2[:h * w], velocity, atol=1e-5), "sample 0 must not see sample 1"
    # --- single sample under bf16 autocast (the release's policy), bf16 weights ---
    dit_bf16 = load_dit(dit_module, checkpoint_module, torch.bfloat16)
    packed_bf16, _ = pack_t2i(pipeline_module, text, x, h, w)
    packed_bf16.timesteps["image"].fill_(t)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        velocity_bf16 = dit_bf16(packed_bf16, timesteps_ref=torch.tensor([t_ref]))
    assert velocity_bf16.dtype == torch.bfloat16, velocity_bf16.dtype
    out["single/velocity_bf16"] = velocity_bf16.float()
    diff = (velocity_bf16.float() - velocity).abs().max().item()
    print(f"dit bf16 autocast vs f32: max|delta|={diff:.3e} peak={velocity.abs().max().item():.3e}")
    return out


def dump_e2e(dit_module, checkpoint_module, pipeline_module, sampling_module, autoencoder_module, encodings) -> dict:
    g = torch.Generator().manual_seed(4)
    dit = load_dit(dit_module, checkpoint_module, torch.float32)
    ae = tiny_autoencoder(autoencoder_module)
    h, w = 2, 3  # 32 x 48 pixels
    noise = torch.randn((h * w, AE_LATENT_CHANNELS), generator=g).to(torch.bfloat16).float()
    text = encodings["short"]
    packed, scaled = pack_t2i(pipeline_module, text, noise.clone(), h, w)
    as_f32(packed)
    schedule = sampling_module.timesteps(4, 6.93)
    with torch.no_grad():
        tokens = sampling_module.denoise_flow_map(dit, packed, schedule)
        image = ae.decode(tokens, autoencoder_module.latent_ids(h, w, "cpu"))
    rgb8 = torch.from_numpy(__import__("numpy").asarray(pipeline_module.to_pil(image)))
    print(f"e2e: tokens {tuple(tokens.shape)} image {tuple(image.shape)} rgb8 {tuple(rgb8.shape)} "
          f"|image| max {image.abs().max().item():.4f}")
    # --- the same loop under bf16 autocast over bf16 weights: the release's sampling policy
    # (`packed.data["image"]` stays f32 from the `.float()` noise; the DiT runs bf16 and
    # `dt * velocity` rounds its product to bf16, as on CUDA).
    dit_bf16 = load_dit(dit_module, checkpoint_module, torch.bfloat16)
    packed_bf16, _ = pack_t2i(pipeline_module, text, noise.clone(), h, w)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        tokens_bf16 = sampling_module.denoise_flow_map(dit_bf16, packed_bf16, schedule)
    assert tokens_bf16.dtype == torch.float32, tokens_bf16.dtype
    peak = tokens.abs().max().item()
    distance = (tokens_bf16 - tokens).abs().max().item() / max(peak, 1.0)
    print(f"e2e bf16 autocast vs f32: distance {distance:.6e} of peak {peak:.3e}")
    return {
        "noise": noise,
        "schedule": schedule,
        "tokens": tokens.float(),
        "tokens_bf16": tokens_bf16.float(),
        "image": image.float(),
        "rgb8": rgb8.to(torch.uint8),
    }, {"e2e_bf16_distance": f"{distance:.6e}"}


def main() -> None:
    package = package_root()
    template_sha = write_encoder_snapshot(package)
    dit_module = importlib.import_module("flux3.dit")
    checkpoint_module = importlib.import_module("flux3.checkpoint")
    pipeline_module = importlib.import_module("flux3.pipeline")
    sampling_module = importlib.import_module("flux3.sampling")
    autoencoder_module = importlib.import_module("flux3.autoencoder")
    text_encoder_module = importlib.import_module("flux3.text_encoder")

    write_dit_snapshot(tiny_dit(dit_module))
    write_autoencoder_fixture(tiny_autoencoder(autoencoder_module))

    meta = {
        "upstream_commit": FLUX3_COMMIT,
        "package_revision": FLUX3_PACKAGE_REVISION,
        "chat_template_sha256": template_sha,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    text, encodings = dump_text(text_encoder_module)
    save_safetensors(FIXTURE_DIR / "flux3_text_golden.safetensors", text, {**meta, "max_length": MAX_LENGTH, "pad_multiple": PAD_MULTIPLE})
    save_safetensors(FIXTURE_DIR / "flux3_dit_golden.safetensors", dump_dit(dit_module, checkpoint_module, pipeline_module, encodings), meta)
    e2e, e2e_meta = dump_e2e(dit_module, checkpoint_module, pipeline_module, sampling_module, autoencoder_module, encodings)
    save_safetensors(FIXTURE_DIR / "flux3_e2e_golden.safetensors", e2e, {**meta, **e2e_meta})


if __name__ == "__main__":
    main()
