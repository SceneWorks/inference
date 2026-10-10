"""Dump the Iris-3B controlled-fixture goldens for the native MLX port (sc-25679).

Everything runs the FROZEN upstream modules (see `_iris_common.py`) on CPU, with seeded miniature
configs that keep every architectural switch of the released `config.yaml` (hybrid dual→single-stream
trunk, GQA, sigmoid attention gate, sandwich RMSNorm, shared-bias adaLN, isotropic 2D RoPE, text RoPE
+ learned text position table, `lap_blocks2` layerwise adapter, post-modulation PiT pixel head with
full-resolution sincos) — only the widths shrink. Every parameter is perturbed away from its init
(adaLN-zero would otherwise make the trunk an identity map and the test vacuous).

The text encoder is driven through upstream's REAL `Qwen3VLTextEncoder.__init__` against a miniature
`Qwen3VLForConditionalGeneration` + WordLevel tokenizer snapshot, so the chat template, the separate
prefix/caption/suffix tokenization, the pad-id resolution, the 300-token (here 10-token) window with
caption truncation and the selected-layer stacking are all upstream's own code. Like the release
(`text_encoder.dtype: bfloat16`) the tower runs in bf16; the DiT and solver run in fp32.

Writes `mlx-gen-iris/tests/fixtures/`:

* `tiny-snapshot/iris/{config.yaml, model.safetensors}`  — the generation backbone, in the exact
  `scripts/export_checkpoint.py` layout (config = model/text_encoder/flow sections).
* `tiny-snapshot/text_encoder/{config.json, model.safetensors, tokenizer.json, …}` — the encoder
  resource, in the `Qwen/Qwen3-VL-4B-Instruct` snapshot layout (`save_pretrained`).
* `iris_text_golden.safetensors`   — `encode()` on three prompts (short / overflowing / empty) and
                                     `null("")`: token ids, masks, stacked selected-layer states.
* `iris_dit_golden.safetensors`    — one masked backbone + pixel-head forward (batch 2, unequal masks,
                                     non-square 12x8 image) and the layerwise-adapter output.
* `iris_solver_golden.safetensors` — `FlowDPMSolver` (order 2, CFG 2.5, shift 4) on an analytic model
                                     function: every step's state, the grid, the default 100-step grid.
* `iris_e2e_golden.safetensors`    — upstream `generate()` end to end (encoder + DiT + solver, CFG 3
                                     against the empty negative prompt) with fixed injected noise.

Run: `IRIS_SRC=/path/to/iris-3b python -I tools/dump_iris_golden.py` (see `_iris_common.py`).
"""

from __future__ import annotations

import json
import shutil
import tempfile

from _iris_common import (
    FIXTURE_DIR,
    IRIS_COMMIT,
    QWEN3_VL_REVISION,
    import_upstream,
    save_safetensors,
)

import_upstream()

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen3VLForConditionalGeneration  # noqa: E402
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig  # noqa: E402

from iris3b.config import FlowConfig, ModelConfig, PixelStageConfig, TextEncoderConfig  # noqa: E402
from iris3b.flow.solver import FlowDPMSolver  # noqa: E402
from iris3b.models.dit import IrisDiT  # noqa: E402
from iris3b.sampling import generate  # noqa: E402
from iris3b.text.qwen3_vl import Qwen3VLTextEncoder  # noqa: E402

torch.manual_seed(0)

SNAPSHOT = FIXTURE_DIR / "tiny-snapshot"
BACKBONE_DIR = SNAPSHOT / "iris"
ENCODER_DIR = SNAPSHOT / "text_encoder"

# ---- miniature geometry --------------------------------------------------------------------------
TEXT_DIM = 32
TEXT_LEN = 10  # == text_encoder.max_length (300 in the release)
# The release's exact 1-based post-block selection (`text_encoder.hidden_layers`, 12 of the 36
# Qwen3-VL-4B blocks -> `text_lap_num_layers: 12`), on a full-depth 36-layer miniature tower, so the
# fixture exercises the real 12-layer stacking law rather than a subset.
HIDDEN_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
TEXT_LAYERS = 36
# Perturbation scale of the miniature tower's weights: small enough that 36 random blocks stay
# contractive, so bf16 activation rounding does not compound into O(1) drift with depth.
TOWER_SCALE = 0.1

TINY_DIT = ModelConfig(
    block="single_stream",
    dual_depth=2,
    final_block_text="keep",
    hidden_size=64,
    depth=3,
    num_heads=4,
    num_kv_heads=2,
    gated_attention=True,
    sandwich_norm=True,
    patch_size=4,
    in_channels=3,
    mlp_ratio=4.0,
    qkv_bias=False,
    qk_norm=True,
    norm_eps=1e-6,
    modulation="shared_bias",
    timestep_max_period=10.0,
    adaln_zero_init=True,
    rope_theta=10000.0,
    rope_scale=16.0,
    rope_aspect="isotropic",
    rope_frame_pairs=0,
    text_rope=True,
    text_rope_theta=10000.0,
    text_abs_pos_embed=True,
    text_dim=TEXT_DIM,
    text_len=TEXT_LEN,
    text_adapter="lap_blocks2",
    text_lap_num_layers=len(HIDDEN_LAYERS),
    text_lap_num_heads=4,
    text_lap_mlp_ratio=1.3,
    repa_layer=0,
    pixel=PixelStageConfig(
        enabled=True,
        depth=2,
        hidden_size=8,
        attn_hidden_size=32,
        num_heads=2,
        mlp_ratio=4.0,
        modulation="post",
        abs_pos_embed=True,
    ),
)
TINY_TEXT = TextEncoderConfig(
    name="qwen3_vl",
    pretrained=str(ENCODER_DIR),
    dim=TEXT_DIM,
    max_length=TEXT_LEN,
    dtype="bfloat16",
    attn_implementation="sdpa",
    hidden_layers=list(HIDDEN_LAYERS),
    compile=False,
    on_caption_overflow="warn",
)
TINY_FLOW = FlowConfig(shift=4.0)

# A WordLevel vocabulary small enough for the tiny tower; every template word that is not listed maps
# to [UNK], which is fine — the ids only have to be the ones upstream's own tokenizer produces.
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
WORDS = ["[UNK]", "system", "user", "assistant", "Describe", "the", "image", "a", "red", "fox", "in",
         "snow", "golden", "hour", ",", ":", "of", "and", "objects", "background", "color", "shape"]
PROMPTS = {
    "short": "a red fox in the snow",
    # 12 caption tokens > budget (10 - suffix): upstream truncates the CAPTION, keeps the suffix.
    "overflow": "a red fox , a red fox , golden hour , snow in the",
    "empty": "",
}


def bf16_round_(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(p.to(torch.bfloat16).to(torch.float32))


def write_encoder_snapshot() -> None:
    if ENCODER_DIR.exists():
        shutil.rmtree(ENCODER_DIR)
    ENCODER_DIR.mkdir(parents=True)
    vocab = {tok: i for i, tok in enumerate(SPECIALS + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    fast.save_pretrained(str(ENCODER_DIR))

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


def tiny_dit() -> IrisDiT:
    model = IrisDiT(TINY_DIT).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            owner = name.rsplit(".", 1)[0].split(".")[-1]
            if "norm" in owner and name.endswith(".weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif name == "y_pos_embedding":
                p.copy_(torch.randn_like(p))
            else:
                p.copy_(0.15 * torch.randn_like(p))
    return model


def write_backbone_snapshot(model: IrisDiT) -> None:
    """`scripts/export_checkpoint.py`: model.safetensors (fp32) + config.yaml (inference sections)."""
    if BACKBONE_DIR.exists():
        shutil.rmtree(BACKBONE_DIR)
    BACKBONE_DIR.mkdir(parents=True)
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        str(BACKBONE_DIR / "model.safetensors"),
        metadata={"format": "pt"},
    )
    sections = {
        "model": OmegaConf.to_container(OmegaConf.structured(TINY_DIT)),
        "text_encoder": OmegaConf.to_container(OmegaConf.structured(TINY_TEXT)),
        "flow": OmegaConf.to_container(OmegaConf.structured(TINY_FLOW)),
    }
    del sections["text_encoder"]["null_embed_dir"]
    # the release config names the HF repo id; the miniature one names itself
    sections["text_encoder"]["pretrained"] = "tiny-snapshot/text_encoder"
    OmegaConf.save(OmegaConf.create(sections), str(BACKBONE_DIR / "config.yaml"))


def dump_text(enc: Qwen3VLTextEncoder) -> dict:
    out = {}
    tok = enc.tokenizer
    out["prefix_ids"] = enc._prefix_ids.to(torch.int32)
    out["suffix_ids"] = enc._suffix_ids.to(torch.int32)
    out["pad_id"] = torch.tensor([enc._pad_id], dtype=torch.int32)
    for name, prompt in PROMPTS.items():
        with torch.no_grad():
            enc_out = enc.encode([prompt])
        caption = tok([prompt], add_special_tokens=False)["input_ids"][0]
        out[f"{name}/caption_ids"] = torch.tensor(caption or [-1], dtype=torch.int32)
        out[f"{name}/mask"] = enc_out.mask[0].to(torch.int32)
        out[f"{name}/embeddings"] = enc_out.embeddings[0].float()
        print(f"text {name}: caption={caption} mask={enc_out.mask[0].tolist()}")
    with torch.no_grad():
        null = enc.null("")
    out["null/embeddings"] = null.embeddings[0].float()
    out["null/mask"] = null.mask[0].to(torch.int32)
    return out


def dump_dit(model: IrisDiT) -> dict:
    g = torch.Generator().manual_seed(1)
    batch, height, width = 2, 12, 8
    x = torch.randn(batch, 3, height, width, generator=g)
    t = torch.tensor([937.5, 125.25])
    y = torch.randn(batch, TEXT_LEN, len(HIDDEN_LAYERS), TEXT_DIM, generator=g)
    mask = torch.zeros(batch, TEXT_LEN, dtype=torch.int64)
    mask[0, :4] = 1
    mask[1, :TEXT_LEN] = 1
    y = y * mask[:, :, None, None]
    adapter = {}
    hook = model.y_embedder.register_forward_hook(lambda m, a, o: adapter.__setitem__("out", o))
    with torch.no_grad():
        out = model(x, t, y, y_mask=mask).x
    hook.remove()
    print(f"dit: out {tuple(out.shape)} mean|out|={out.abs().mean():.4f}")
    return {"x": x, "t": t, "y": y, "y_mask": mask.to(torch.int32), "adapter_out": adapter["out"], "out": out}


def analytic_model(x: torch.Tensor, t_model: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A smooth, x/t/y-dependent stand-in velocity the native test reimplements exactly."""
    t = (t_model / 1000.0).view(-1, 1, 1, 1)
    return torch.tanh(x) * (0.5 + t) + y.view(-1, 1, 1, 1) * (1.0 - t)


def dump_solver() -> tuple[dict, dict]:
    g = torch.Generator().manual_seed(2)
    z = torch.randn(1, 3, 4, 4, generator=g)
    cond, uncond = torch.tensor([0.7]), torch.tensor([-0.4])
    steps, order, cfg_scale, shift = 7, 2, 2.5, 4.0
    solver = FlowDPMSolver(analytic_model, num_timesteps=1000, cfg_scale=cfg_scale, cfg_interval=(0.0, 1.0))
    # replay sample()'s own loop, recording every step's state
    grid = solver.time_grid(steps, shift)
    x, history, states = z.clone(), [], []
    for i in range(1, steps + 1):
        s, t = grid[i - 1], grid[i]
        x0 = solver._pred_x0(x, s, cond, uncond)
        history.append((s, x0))
        step_order = min(i, order, steps + 1 - i)
        if step_order == 1:
            x = solver._first_order(x, s, t, x0)
        else:
            x = solver._second_order(x, history[-2], history[-1], t)
        if len(history) > 2:
            history.pop(0)
        states.append(x.clone())
    final = solver.sample(z, cond, uncond, steps=steps, order=order, shift=shift)
    torch.testing.assert_close(final, states[-1], rtol=0, atol=0)
    # The f64 time grids travel as exact JSON (repr round-trips) in the safetensors metadata: MLX
    # cannot load an F64 tensor.
    tensors = {"z": z, "states": torch.stack(states)}
    grids = {
        "grid": json.dumps(grid),
        "default_grid": json.dumps(FlowDPMSolver.time_grid(100, 4.0)),
    }
    return tensors, grids


def dump_e2e(model: IrisDiT, enc: Qwen3VLTextEncoder) -> dict:
    g = torch.Generator().manual_seed(3)
    height, width = 8, 12
    noise = torch.randn(1, 3, height, width, generator=g)
    with torch.no_grad():
        image = generate(
            model, enc, [PROMPTS["short"]], height=height, width=width, steps=6, order=2,
            cfg_scale=3.0, cfg_interval=(0.0, 1.0), shift=4.0, negative_prompt="", device="cpu",
            noise=noise,
        )
    print(f"e2e: image {tuple(image.shape)} mean={image.mean():.4f}")
    return {"noise": noise, "image": image}


def main() -> None:
    write_encoder_snapshot()
    dit = tiny_dit()
    write_backbone_snapshot(dit)
    with tempfile.TemporaryDirectory() as null_dir:
        cfg = TextEncoderConfig(**{**vars(TINY_TEXT), "null_embed_dir": null_dir})
        enc = Qwen3VLTextEncoder(cfg, device="cpu")  # upstream's real constructor
        meta = {
            "upstream_commit": IRIS_COMMIT,
            "qwen3_vl_revision": QWEN3_VL_REVISION,
            "torch": torch.__version__,
        }
        save_safetensors(FIXTURE_DIR / "iris_text_golden.safetensors", dump_text(enc), meta)
        save_safetensors(FIXTURE_DIR / "iris_dit_golden.safetensors", dump_dit(dit), meta)
        solver_tensors, grids = dump_solver()
        save_safetensors(
            FIXTURE_DIR / "iris_solver_golden.safetensors",
            solver_tensors,
            {**meta, **grids, "steps": 7, "order": 2, "cfg_scale": 2.5, "shift": 4.0, "cond": 0.7,
             "uncond": -0.4},
        )
        save_safetensors(
            FIXTURE_DIR / "iris_e2e_golden.safetensors",
            dump_e2e(dit, enc),
            {**meta, "prompt": PROMPTS["short"], "negative_prompt": "", "steps": 6, "cfg_scale": 3.0,
             "width": 12, "height": 8},
        )


if __name__ == "__main__":
    main()
