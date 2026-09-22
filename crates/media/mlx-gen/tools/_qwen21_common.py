"""Shared setup for the Qwen-Image 2.1 fixture generators (`dump_qwen21_*.py`, sc-24108).

Every generator builds the SAME tiny, seeded, frozen-upstream components — a `QwenImage21Pipeline`
whose transformer / RGBA VAE / Qwen3-VL text encoder are the diffusers + transformers classes at the
pinned revisions with miniature configs — so a Rust component test and the Rust end-to-end test read
one consistent snapshot. The tiny snapshot is written in the exact layout of `Qwen/Qwen-Image-2.1`
(`processor/`, `scheduler/`, `text_encoder/`, `transformer/`, `vae/`, `model_index.json`), so the
production loader in `mlx-gen-qwen-image-2-1/src/loader.rs` is exercised unchanged.

Frozen upstream (recorded in `mlx-gen-qwen-image-2-1/src/lib.rs` as constants):
  * HF `Qwen/Qwen-Image-2.1` @ 790c92633540aa0cb11d9abf19eb46d861714758 (scheduler config below)
  * diffusers `QwenImage21Pipeline` @ 8b3c707ebd3ec4881f4190cf42931da07eaf3b65
  * `QwenLM/Qwen-Image-2.1` @ fb7ae1d1f9611cd91524d03c53c5246b36ac8577 (presets, defaults)

Run with the pinned venv (torch CPU is enough; nothing here touches a GPU or real weights):
    python3.12 tools/dump_qwen21_pipeline.py      # writes the tiny snapshot + e2e goldens (run first)
    python3.12 tools/dump_qwen21_scheduler.py
    python3.12 tools/dump_qwen21_text_encoder.py
    python3.12 tools/dump_qwen21_transformer.py
    python3.12 tools/dump_qwen21_vae.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from _paths import fixture

FIXTURE_DIR = Path(fixture("mlx-gen-qwen-image-2-1/tests/fixtures"))
SNAPSHOT_DIR = FIXTURE_DIR / "tiny-snapshot"

# `Qwen/Qwen-Image-2.1/scheduler/scheduler_config.json` @ 790c9263, verbatim.
SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# The seven upstream presets (README `aspect_ratios`), (width, height).
PRESETS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

SYS_PROMPT = "Comprehend and analyze the provided prompt."

# Tiny geometry shared by every component. The transformer consumes the VAE latent directly, so
# `in_channels == z_dim`; the text width is the DiT's `context_in_dim`.
Z_DIM = 8
TEXT_HIDDEN = 32
TEXT_LAYERS = 2
TEXT_HEADS = 2
TEXT_KV_HEADS = 1
TEXT_HEAD_DIM = 16
TEXT_INTERMEDIATE = 64
VOCAB_SIZE = 256
DIT_LAYERS = 2
DIT_HEAD_DIM = 16
DIT_HEADS = 2
DIT_MLP_RATIO = 2
DIT_AXES = (4, 6, 6)

# Every word the tiny WordLevel tokenizer knows. Prompts used by the Rust tests must draw from
# this list (plus punctuation); anything else maps to `[UNK]`.
VOCAB_WORDS = [
    "[UNK]",
    "system",
    "user",
    "assistant",
    "Comprehend",
    "and",
    "analyze",
    "the",
    "provided",
    "prompt",
    ".",
    ",",
    "a",
    "red",
    "fox",
    "in",
    "forest",
    "blurry",
    "low",
    "quality",
    "photo",
    "of",
    "cat",
    "on",
    "moon",
    "neon",
    "sign",
    "night",
    "city",
    "blue",
    "green",
    "sticker",
    "transparent",
    "background",
    "This",
    "is",
    "an",
    "RGBA",
    "image",
    "with",
    "transparency",
    "The",
    "has",
    "alpha",
    "channel",
]
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
]


def seed_all(seed: int = 0) -> None:
    torch.manual_seed(seed)


def build_tiny_tokenizer():
    """A WordLevel tokenizer with the Qwen special tokens, small dense ids (< VOCAB_SIZE)."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {word: i for i, word in enumerate(VOCAB_WORDS)}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens(SPECIAL_TOKENS)
    assert tok.get_vocab_size() < VOCAB_SIZE
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="[UNK]",
        additional_special_tokens=SPECIAL_TOKENS,
        model_max_length=4096,
    )
    return fast


def real_chat_template() -> str:
    """The Qwen3-VL chat template. Read from the pinned snapshot when present (so the tiny
    processor derives `_drop_idx` through the very template the real one does), else the minimal
    system/user rendering the pipeline relies on."""
    for root in (os.environ.get("MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT"),):
        if root:
            path = Path(root) / "processor" / "chat_template.jinja"
            if path.is_file():
                return path.read_text()
    return (
        "{%- if messages[0].role == 'system' %}{{- '<|im_start|>system\\n' }}"
        "{%- for content in messages[0].content %}{%- if 'text' in content %}{{- content.text }}"
        "{%- endif %}{%- endfor %}{{- '<|im_end|>\\n' }}{%- endif %}"
        "{%- for message in messages %}{%- if message.role != 'system' %}"
        "{{- '<|im_start|>' + message.role + '\\n' }}{%- for content in message.content %}"
        "{%- if 'text' in content %}{{- content.text }}{%- endif %}{%- endfor %}"
        "{{- '<|im_end|>\\n' }}{%- endif %}{%- endfor %}"
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
    )


def assert_released_template_matches_literal_prefix() -> None:
    """The Rust port derives the system-prefix drop count by tokenizing the literal
    `<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n`; upstream derives it through
    `processor.apply_chat_template([system message])`. Prove the two agree token-for-token on the
    RELEASED tokenizer + chat template (and pin the count, 14) whenever the pinned snapshot is
    present, so the literal can never drift from the template silently."""
    from transformers import Qwen3VLProcessor

    snap = Path(os.environ["MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT"])
    proc = Qwen3VLProcessor.from_pretrained(str(snap / "processor"))
    msg = [{"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]}]
    via_template = proc.apply_chat_template(msg, tokenize=True, return_dict=False)[0]
    via_literal = proc.tokenizer(f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n").input_ids
    assert via_template == via_literal, (via_template, via_literal)
    assert len(via_literal) == 14, via_literal
    print(f"released chat template == literal prefix: {len(via_literal)} tokens {via_literal}")


def build_tiny_pipeline():
    """The tiny `QwenImage21Pipeline`: seeded components, real scheduler config, tiny tokenizer."""
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from transformers import (
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    seed_all(0)
    transformer = QwenImage21Transformer2DModel(
        patch_size=1,
        in_channels=Z_DIM,
        out_channels=Z_DIM,
        num_layers=DIT_LAYERS,
        attention_head_dim=DIT_HEAD_DIM,
        num_attention_heads=DIT_HEADS,
        context_in_dim=TEXT_HIDDEN,
        mlp_ratio=DIT_MLP_RATIO,
        axes_dims_rope=DIT_AXES,
    )
    # Zero-centered RMSNorm weight and the shared modulation both initialise to something a parity
    # test cannot distinguish from a bug (all zeros / ones), so spread every parameter.
    with torch.no_grad():
        for p in transformer.parameters():
            p.copy_(torch.randn_like(p) * 0.2)

    seed_all(1)
    # Five `dim_mult` stages -> four spatial downsamples -> the 16x compression the pipeline assumes.
    # NON-UNIFORM widths on purpose (sc-24108 review): with a uniform `dim_mult` every stage has
    # `in == out`, so `DupUp3D`'s channel-duplication index collapses to an identity gather and no
    # `conv_shortcut` exists — the port's derivation would be untested. `[1, 1, 2, 4, 4]` with
    # `decoder_base_dim != base_dim` gives the production shape: decoder up-stage 2 has
    # `repeats = 4 < 8` (the `first_chunk` temporal slot matters) and up-stage 3 has `repeats = 2`
    # (the row-offset matters), plus width-changing residual blocks in both encoder and decoder.
    vae = AutoencoderKLQwenImage21(
        base_dim=4,
        decoder_base_dim=6,
        z_dim=Z_DIM,
        dim_mult=[1, 1, 2, 4, 4],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True, True],
        latents_mean=[float(x) for x in (torch.randn(Z_DIM) * 0.5).tolist()],
        latents_std=[float(x) for x in (torch.rand(Z_DIM) * 0.5 + 0.75).tolist()],
    )
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_(torch.randn_like(p) * 0.2)

    scheduler = FlowMatchEulerDiscreteScheduler(**SCHEDULER_CONFIG)

    seed_all(2)
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": TEXT_HIDDEN,
            "intermediate_size": TEXT_INTERMEDIATE,
            "num_hidden_layers": TEXT_LAYERS,
            "num_attention_heads": TEXT_HEADS,
            "num_key_value_heads": TEXT_KV_HEADS,
            "head_dim": TEXT_HEAD_DIM,
            "vocab_size": VOCAB_SIZE,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 5000000.0,
                "mrope_section": [4, 2, 2],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": TEXT_HIDDEN,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [0],
        },
    )
    # The vision token ids must be the TINY tokenizer's, not `Qwen3VLConfig`'s released defaults
    # (151652-151656): `Qwen3VLModel.forward` matches `config.image_token_id` against the tokenized
    # ids to find the slots it splices vision features into, so a stale id makes every
    # image-conditioned call fail with "tokens: 0, features: N" (sc-24110).
    tokenizer_for_ids = build_tiny_tokenizer()
    config.image_token_id = tokenizer_for_ids.convert_tokens_to_ids("<|image_pad|>")
    config.video_token_id = tokenizer_for_ids.convert_tokens_to_ids("<|video_pad|>")
    config.vision_start_token_id = tokenizer_for_ids.convert_tokens_to_ids("<|vision_start|>")
    config.vision_end_token_id = tokenizer_for_ids.convert_tokens_to_ids("<|vision_end|>")
    text_encoder = Qwen3VLForConditionalGeneration(config).eval()
    with torch.no_grad():
        for p in text_encoder.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
        # The released final norm weight is not all-ones; give the pre-norm extraction something
        # to be wrong about (the pipeline reads the hidden state BEFORE this norm).
        text_encoder.model.language_model.norm.weight.copy_(
            torch.linspace(0.5, 2.0, TEXT_HIDDEN)
        )

    tokenizer = tokenizer_for_ids
    if os.environ.get("MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT"):
        assert_released_template_matches_literal_prefix()
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, merge_size=2, temporal_patch_size=2, min_pixels=32 * 32, max_pixels=64 * 64
        ),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        chat_template=real_chat_template(),
    )
    pipe = QwenImage21Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        processor=processor,
        transformer=transformer,
    )
    return pipe


def load_tiny_pipeline():
    """Reload the tiny pipeline from the snapshot `dump_qwen21_pipeline.py` wrote, so component
    generators read the very bytes the Rust loader reads."""
    from diffusers import QwenImage21Pipeline

    if not (SNAPSHOT_DIR / "model_index.json").is_file():
        raise SystemExit(f"run dump_qwen21_pipeline.py first ({SNAPSHOT_DIR} is missing)")
    return QwenImage21Pipeline.from_pretrained(str(SNAPSHOT_DIR), dtype=torch.float32)


def save_safetensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: v.detach().contiguous().to(torch.float32) if v.is_floating_point() else v.detach().contiguous() for k, v in tensors.items()}
    save_file(flat, str(path), metadata={k: str(v) for k, v in (metadata or {}).items()})
    print(f"wrote {path} ({len(flat)} tensors, {path.stat().st_size} bytes)")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
