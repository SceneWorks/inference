"""Dump the FLUX 3 Image Klein **prompt upsampler** goldens for the native port (sc-26238).

Everything text-shaped runs the FROZEN upstream `pup.py` / `bbox.py` (see `_flux3_common.py` for
the tree verification) and the pinned `black-forest-labs/flux-3-image-klein-pup` package's own
tokenizer + chat template, so the mode selection, the scaffolds, the locked-box input, the
reconcile step, the bypass caption and the rendered chat text are all upstream's own bytes. The
reference-image policy runs upstream's `policy_reference` through Pillow. The controlled
multimodal run is a seeded miniature `Qwen3_5ForConditionalGeneration` (the package's
architecture: hybrid linear/full attention, a Qwen3-VL ViT with `deepstack_visual_indexes: []`,
`attn_output_gate`) with a WordLevel tokenizer that carries the pinned chat template, driven in
greedy mode over the four modes.

Writes:

* `crates/contracts/gen-core/tests/fixtures/flux3_pup/golden.json` — the text goldens: per mode
  case the `bbox.plan` output, the rendered chat text (`enable_thinking=False`), the real-tokenizer
  and tiny-tokenizer prompt lengths, the bypass caption; `bbox.reconcile` / `detect` /
  `is_finished` / `to_canvas` cases; `pup.reference_size` and the Qwen `smart_resize` token table.
* `crates/contracts/gen-core/tests/fixtures/flux3_pup/policy_reference.safetensors` — synthetic
  references (RGB / RGBA, up- and down-scaled, 1 / 2 / 3 refs) with upstream's LANCZOS output
  and the JPEG-q92 round-trip output.
* `crates/contracts/gen-core/tests/fixtures/flux3_pup/tiny-pup-package/` — the miniature package
  WITHOUT weights (config, WordLevel tokenizer, pinned chat template, processor configs, system
  prompts, an empty `model.safetensors`) for the host-side package tests.
* `mlx-gen-flux3/tests/fixtures/tiny-pup-snapshot/` — the same package WITH the seeded miniature
  weights, and `mlx-gen-flux3/tests/fixtures/flux3_pup_golden.safetensors` — per mode the expanded
  input ids, the policy-preprocessed reference pixels, the last prefill logits, the greedy ids and
  the top-1/top-2 logit margins.

Inputs: `FLUX3_SRC` (the frozen source tree) and `FLUX3_PUP` (the pinned pup package snapshot:
only its configs, tokenizer and template are read, sha256-verified; the 18.8 GB weights are not
needed). Run: `FLUX3_SRC=… FLUX3_PUP=… python -I tools/dump_flux3_pup_golden.py`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

# `python -I` keeps the script directory off `sys.path`; the shared helpers live beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _flux3_common import FIXTURE_DIR, FLUX3_COMMIT, import_upstream, save_safetensors, write_json  # noqa: E402
from _paths import require_env  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerFast  # noqa: E402
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: E402

torch.manual_seed(0)

PUP_REVISION = "8b069751eb5dcab23829d9c5eccb470eb8b8a4ed"
# sha256 of the upstream modules this oracle executes, at FLUX3_COMMIT (`_flux3_common` verifies
# the modules S1 pinned; these are the upsampler's).
PUP_SOURCE_SHA256 = {
    "src/flux3/pup.py": "e3561ae3cbff1a080ad8854c0b89bd0687c1ea9d84c74f5203b44844cbd4684d",
    "src/flux3/bbox.py": "fbea10c1c53980df1665d3cb6cc6264b8f50633399bf6809c91650f1dcf4fc37",
    "src/flux3/pup_prompts.json": "b5da3ed8bc04ab59dbdb747930f7dbff847e868903fed8028f85ee1a4ac82fb4",
}
# Verify every executable frozen module BEFORE importing it; dependencies above were loaded
# before exposing the downloaded source directory on sys.path. Keep the trusted oracle as cwd.
_frozen_source = import_upstream()
for _relative, _expected in PUP_SOURCE_SHA256.items():
    _got = hashlib.sha256((_frozen_source / _relative).read_bytes()).hexdigest()
    if _got != _expected:
        raise SystemExit(f"{_relative}: sha256 {_got} != pinned {_expected}")
from flux3 import bbox, pup  # noqa: E402

# sha256 of the pinned package files this oracle reads (revision PUP_REVISION).
PUP_PACKAGE_SHA256 = {
    "config.json": "7d7fc026d8bcb9f1fc505af1004779330fe59bb3256fe0271ec2568e8dc89db1",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
    "preprocessor_config.json": "3a159dfec9978a186a72ba085e0ad6a050f3968d8b364218d7bd13f5c89381f2",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    "system_prompts.json": "b5da3ed8bc04ab59dbdb747930f7dbff847e868903fed8028f85ee1a4ac82fb4",
    "generation_config.json": "e92258e3c39afbaa73fd53a98640c56f6d42fb091e1664697bd5396a1dcbb761",
}

REPO_ROOT = Path(__file__).resolve().parents[4]
GEN_CORE_FIXTURES = REPO_ROOT / "crates" / "contracts" / "gen-core" / "tests" / "fixtures" / "flux3_pup"
TINY_PACKAGE = GEN_CORE_FIXTURES / "tiny-pup-package"
TINY_SNAPSHOT = FIXTURE_DIR / "tiny-pup-snapshot"
MLX_GOLDEN = FIXTURE_DIR / "flux3_pup_golden.safetensors"
GREEDY_STEPS = 12

# ---- the miniature tokenizer -----------------------------------------------------------------
SPECIALS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
    "<|video_pad|>", "<think>", "</think>",
]
WORDS = [
    "[UNK]", "system", "user", "assistant", "a", "red", "bicycle", "leaning", "against", "brick", "wall",
    "two", "hot", "air", "balloons", "over", "the", "sea", "put", "bandana", "around", "dog", "s", "neck",
    "add", "small", "potted", "cactus", "on", "table", "Reference", "images", "follow", "in", "order",
    "first", "image", "second", "User", "request", "editing", "instruction", "Output", "caption",
    "attachment", "ref_image_0", "ref_image_1", "Decide", "edit", "imagine", "target", "and", "output",
    "annotation", "record", "ORIGINAL", "PROMPT", "LOCKED", "BOUNDING", "BOX", "ANNOTATIONS", "id", "bbox",
    "desc", "obj_1", "obj_2", "from", "src_bbox", "tgt_bbox", "null", ",", ":", ".", "-", "(", ")", "[", "]",
    "{", "}", "\"", "'", "...", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "1000", "y_min", "x_min",
    "y_max", "x_max", "canvas", "with", "at", "top", "left", "This", "is", "an", "authoritative", "JSON",
    "list", "of", "dictionaries", "preserve", "every", "supplied", "identity", "provenance", "value",
    "coordinate", "exactly", "striped", "balloon", "blue", "lighthouse", "cliff", "yellow",
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path, table: dict[str, str], label: str) -> None:
    for rel, want in table.items():
        got = sha256_of(root / rel)
        if got != want:
            raise SystemExit(f"{label}/{rel}: sha256 {got} != pinned {want}")


def package_root() -> Path:
    root = Path(require_env("FLUX3_PUP", f"the pinned black-forest-labs/flux-3-image-klein-pup snapshot at {PUP_REVISION}"))
    verify(root, PUP_PACKAGE_SHA256, "FLUX3_PUP")
    return root


def upstream_root() -> Path:
    root = Path(require_env("FLUX3_SRC", "the frozen source tree"))
    verify(root, PUP_SOURCE_SHA256, "FLUX3_SRC")
    return root


# ---- cases -----------------------------------------------------------------------------------
def load_example_boxes(src: Path, name: str) -> list[dict]:
    return json.loads((src / "examples" / "inputs" / name).read_text(encoding="utf-8"))


def plan_cases(src: Path) -> list[dict]:
    balloons = load_example_boxes(src, "balloons_boxes.json")
    cactus = load_example_boxes(src, "cactus_boxes.json")
    rows = [{"id": "obj_1", "bbox": [80, 60, 480, 380], "desc": "a red balloon"},
            {"id": "obj_2", "bbox": [200, 620, 420, 800], "desc": "a blue balloon"}]
    table = json.dumps(rows, ensure_ascii=False)
    unicode_boxes = [{"id": "sign_1", "desc": "a sign reading \"CAFÉ ☕\" with a \\ backslash\nand a newline",
                      "bbox": [0.0825, 0.0835, 0.5, 0.77]}]
    return [
        {"name": "t2i", "prompt": "a red bicycle leaning against a brick wall", "boxes": None, "n_images": 0},
        {"name": "t2i_bbox", "prompt": " two hot-air balloons over the sea\n", "boxes": balloons, "n_images": 0},
        {"name": "i2i", "prompt": "put a red bandana around the dog's neck", "boxes": None, "n_images": 1},
        {"name": "i2i_bbox", "prompt": "add a small potted cactus on the table", "boxes": cactus, "n_images": 2},
        {"name": "t2i_detected", "prompt": f"a red balloon <obj_1> and a blue balloon {table}", "boxes": None, "n_images": 0},
        {"name": "t2i_finished", "prompt": f"a red balloon <obj_1> and a blue balloon <obj_2> {table}", "boxes": None, "n_images": 0},
        {"name": "t2i_locked_input", "prompt": bbox.upsampling_input("  two balloons  ", rows), "boxes": None, "n_images": 0},
        {"name": "i2i_detected", "prompt": f"move the balloon <obj_1> {table}", "boxes": None, "n_images": 3},
        {"name": "t2i_unicode", "prompt": "a café ☕ sign, \"quoted\" and \\ backslash", "boxes": unicode_boxes, "n_images": 0},
        {"name": "t2i_empty_boxes", "prompt": "an empty box list is no boxes", "boxes": [], "n_images": 0},
        {"name": "t2i_trailing_list", "prompt": "a caption ending in [SALE]", "boxes": None, "n_images": 0},
    ]


def reconcile_cases() -> list[dict]:
    locked = [{"id": "obj_1", "bbox": [80, 60, 480, 380], "desc": "a red balloon"},
              {"id": "obj_2", "bbox": [200, 620, 420, 800], "desc": "a blue balloon"}]
    edit_locked = [{"id": "obj_1", "from": "target", "src_bbox": None, "tgt_bbox": [550, 620, 850, 850], "desc": "a small potted cactus"}]
    odd_locked = [{"z": 1, "id": "obj_1", "src_bbox": [1, 2, 3, 4], "from": "ref_image_0", "desc": None}]
    return [
        {"name": "rename_adopt_append", "locked": locked,
         "text": "Full-color photograph of a red balloon <balloon_1> and a blue balloon <obj_2> over a sea <sea_1>. "
                 "[{\"id\": \"balloon_1\", \"bbox\": [82, 60, 480, 380], \"desc\": \" A red balloon with stripes \"}, "
                 "{\"id\": \"obj_2\", \"bbox\": [1, 1, 2, 2], \"desc\": \"\"}, "
                 "{\"id\": \"sea_1\", \"bbox\": [600, 0, 1000, 1000], \"desc\": \"grey sea\"}, "
                 "{\"id\": \"ghost_1\", \"bbox\": [0, 0, 10, 10], \"desc\": \"uncited\"}, 7]"},
        {"name": "no_table", "locked": locked, "text": "  just prose, no table  "},
        {"name": "prose_with_brackets", "locked": locked, "text": "a sign reading \"[SALE]\" <obj_1> [not json"},
        {"name": "edit_rows", "locked": edit_locked,
         "text": "Add a cactus <obj_1> beside the cup <cup_1> from <ref_image_0>, keep the lamp <lamp_1>. "
                 "[{\"id\": \"obj_1\", \"from\": \"target\", \"src_bbox\": null, \"tgt_bbox\": [550, 620, 850, 850], \"desc\": \"a cactus in a clay pot\"}, "
                 "{\"id\": \"cup_1\", \"from\": \"ref_image_0\", \"src_bbox\": [300, 100, 500, 250], \"tgt_bbox\": [300, 100, 500, 250], \"desc\": \"a white mug\"}, "
                 "{\"id\": \"lamp_1\", \"from\": \"ref_image_0\", \"src_bbox\": [0, 0, 1, 1], \"tgt_bbox\": null, \"desc\": \"bad src\"}, "
                 "{\"id\": \"bad_1\", \"from\": \"target\", \"src_bbox\": [1, 2, 3, 4], \"tgt_bbox\": [1, 2, 3, 4], \"desc\": \"target with src\"}]"},
        {"name": "odd_locked_keys", "locked": odd_locked, "text": "x <obj_1> [{\"id\": \"obj_1\", \"desc\": \"d\"}]"},
        {"name": "unicode_desc", "locked": locked,
         "text": "Un café ☕ <obj_1> <obj_2> [{\"id\": \"obj_1\", \"bbox\": [80, 60, 480, 380], \"desc\": \"ballon « rouge » \\u00e9\\n\"}]"},
        {"name": "iou_rename_float_box", "locked": locked,
         "text": "a <b_1> b <obj_2> [{\"id\": \"b_1\", \"bbox\": [80.0, 60.5, 480, 380], \"desc\": \"float box\"}, {\"id\": \"obj_2\", \"bbox\": [200, 620, 420, 800], \"desc\": \"blue\"}]"},
        {"name": "non_dict_rows_only", "locked": locked, "text": "caption [1, 2, 3]"},
    ]


def detect_cases() -> list[dict]:
    table = "[{\"id\": \"obj_1\", \"bbox\": [80, 60, 480, 380], \"desc\": \"b\"}]"
    return [
        {"prompt": f"a balloon <obj_1> {table}"},
        {"prompt": f"a balloon {table}"},
        {"prompt": "a [{\"id\": \"obj_1\", \"bbox\": [0.1, 0.1, 0.5, 0.5]}]"},
        {"prompt": "a [{\"id\": \"1obj\", \"bbox\": [80, 60, 480, 380]}]"},
        {"prompt": "a [{\"id\": \"obj_1\", \"tgt_bbox\": [80, 60, 480, 380]}, 5]"},
        {"prompt": "a [{\"id\": \"obj_1\", \"src_bbox\": [80, 60, 480, 380]}]  "},
        {"prompt": "[]"},
        {"prompt": "a caption ending in [SALE]"},
        {"prompt": bbox.upsampling_input("  two balloons  ", json.loads(table))},
        {"prompt": "ORIGINAL PROMPT:\nno annotations line " + table},
        {"prompt": "x [1] y " + table},
        {"prompt": "\u001fpad\u001f " + table},
    ]


def bypass_cases(src: Path) -> list[dict]:
    balloons = load_example_boxes(src, "balloons_boxes.json")
    return [
        {"prompt": " p \n", "boxes": balloons},
        {"prompt": " p \n", "boxes": None},
        {"prompt": "p", "boxes": []},
        {"prompt": "é", "boxes": [{"desc": "ünïcode", "tgt_bbox": [0.5, 0.5, 1, 1], "from": "target"}]},
    ]


def to_canvas_error_cases() -> list[dict]:
    return [
        {"boxes": [{"bbox": [0.1, 0.2]}]},
        {"boxes": [{"id": "x", "desc": "y"}]},
        {"boxes": [{"id": "ok", "bbox": [0, 0, 1, 1]}, {"src_bbox": [1, 2, 3]}]},
    ]


def reference_size_cases() -> list[dict]:
    sizes = [(1024, 1024, 1), (1000, 1000, 1), (2000, 1000, 1), (1024, 1024, 2), (1024, 1024, 3), (10, 10, 1),
             (48, 48, 1), (80, 80, 1), (4000, 32, 10), (16000, 80, 1), (201, 1, 1), (1, 200, 2), (3000, 15, 1),
             (1536, 2048, 1), (1536, 2048, 2), (1536, 2048, 10), (33, 31, 5), (1023, 1025, 1), (724, 724, 2),
             (725, 725, 2), (7, 1400, 1), (640, 480, 4)]
    out = []
    for w, h, n in sizes:
        try:
            out.append({"w": w, "h": h, "n": n, "expect": list(pup.reference_size(w, h, n))})
        except ValueError as error:
            out.append({"w": w, "h": h, "n": n, "error": str(error)})
    return out


def vision_token_cases(processor) -> list[dict]:
    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size
    out = []
    for w, h in [(1024, 1024), (32, 32), (512, 512), (704, 704), (1440, 704), (992, 992), (64, 64), (32, 64), (192, 384), (12800, 64), (31, 33)]:
        hb, wb = smart_resize(h, w, factor=factor, min_pixels=ip.size["shortest_edge"], max_pixels=ip.size["longest_edge"])
        out.append({"w": w, "h": h, "resized": [wb, hb], "tokens": (hb // factor) * (wb // factor)})
    return out


# ---- synthetic reference images ----------------------------------------------------------------
def synthetic_images() -> dict[str, tuple[np.ndarray, str, int]]:
    rng = np.random.default_rng(2626)

    def rgb(w, h):
        yy, xx = np.mgrid[0:h, 0:w]
        base = np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)), ((xx + yy) * 7 % 256)], axis=-1)
        return np.clip(base + rng.integers(-20, 20, size=(h, w, 3)), 0, 255).astype(np.uint8)

    def rgba(w, h):
        a = rgb(w, h)
        yy, xx = np.mgrid[0:h, 0:w]
        alpha = ((xx * 255) // max(w - 1, 1)).astype(np.uint8)
        alpha[yy % 7 == 0] = 0
        alpha[yy % 11 == 0] = 255
        return np.concatenate([a, alpha[..., None]], axis=-1)

    # Small on purpose (the fixture is committed): the one down-scaled case sits just over the
    # three-reference 512² budget rather than over the 1 MP one.
    return {
        "rgb_70x50_n1": (rgb(70, 50), "RGB", 1),
        "rgba_100x64_n1": (rgba(100, 64), "RGBA", 1),
        "rgb_600x500_n3": (rgb(600, 500), "RGB", 3),
        "rgb_40x60_n2": (rgb(40, 60), "RGB", 2),
        "rgb_20x20_n3": (rgb(20, 20), "RGB", 3),
        "rgb_64x32_n2": (rgb(64, 32), "RGB", 2),
        "rgba_33x31_n5": (rgba(33, 31), "RGBA", 5),
    }


def policy_fixture() -> tuple[dict, dict, dict[str, np.ndarray]]:
    from PIL import ImageOps

    tensors, meta, outputs = {}, {}, {}
    for name, (arr, mode, n) in synthetic_images().items():
        image = Image.fromarray(arr, mode)
        # Upstream's steps up to the resize, so the resampler is checked bit-exactly on its own.
        pre = ImageOps.exif_transpose(image)
        if pre.mode in ("RGBA", "LA", "PA"):
            pre_rgba = pre.convert("RGBA")
            pre = Image.alpha_composite(Image.new("RGBA", pre_rgba.size, (255, 255, 255, 255)), pre_rgba)
        pre = pre.convert("RGB")
        size = pup.reference_size(pre.width, pre.height, n)
        resized = pre.resize(size, Image.Resampling.LANCZOS) if pre.size != size else pre
        out = pup.policy_reference(image, n)
        assert out.size == size, (out.size, size)
        tensors[f"in_{name}"] = torch.from_numpy(arr.copy())
        if mode != "RGB":
            # The RGB conversion is only a step of its own for an alpha input.
            tensors[f"rgb_{name}"] = torch.from_numpy(np.asarray(pre).copy())
        if resized.width * resized.height <= 128 * 128:
            # The resampler's bit-exact witness, kept for the small cases only (fixture size).
            tensors[f"resized_{name}"] = torch.from_numpy(np.asarray(resized).copy())
        tensors[f"out_{name}"] = torch.from_numpy(np.asarray(out).copy())
        meta[f"n_refs_{name}"] = n
        meta[f"mode_{name}"] = mode
        outputs[name] = np.asarray(out).copy()
    return tensors, meta, outputs


# ---- the miniature VLM -------------------------------------------------------------------------
def tiny_tokenizer(template: str) -> PreTrainedTokenizerFast:
    vocab = {tok: i for i, tok in enumerate(SPECIALS + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        additional_special_tokens=SPECIALS,
        chat_template=template,
    )
    return fast


def tiny_config(vocab_size: int, ids: dict[str, int]) -> Qwen3_5Config:
    return Qwen3_5Config(
        text_config={
            "vocab_size": vocab_size,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "attn_output_gate": True,
            "full_attention_interval": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "mrope_section": [2, 1, 1], "mrope_interleaved": True, "partial_rotary_factor": 0.25},
            "mtp_num_hidden_layers": 0,
            "tie_word_embeddings": False,
            "eos_token_id": ids["<|endoftext|>"],
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
            "initializer_range": 0.02,
        },
        image_token_id=ids["<|image_pad|>"],
        video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"],
        vision_end_token_id=ids["<|vision_end|>"],
        eos_token_id=ids["<|im_end|>"],
        pad_token_id=ids["<|endoftext|>"],
        tie_word_embeddings=False,
    )


def seed_model(model: torch.nn.Module) -> None:
    g = torch.Generator().manual_seed(2638)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or "norm" in name.split(".")[-2:]:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif p.dim() <= 1:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.2 * torch.randn(p.shape, generator=g) / (p.shape[-1] ** 0.5))


def write_package_files(dest: Path, package: Path, fast: PreTrainedTokenizerFast, with_weights: torch.nn.Module | None) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    fast.save_pretrained(str(dest))
    for stale in ("special_tokens_map.json", "added_tokens.json", "vocab.txt"):
        if (dest / stale).exists() and stale != "special_tokens_map.json":
            (dest / stale).unlink()
    # The pinned template as the sidecar file the serving engines prefer.
    shutil.copyfile(package / "chat_template.jinja", dest / "chat_template.jinja")
    for name in ("preprocessor_config.json", "processor_config.json", "system_prompts.json", "generation_config.json"):
        shutil.copyfile(package / name, dest / name)
    if with_weights is not None:
        with_weights.save_pretrained(str(dest), safe_serialization=True)
    else:
        config = with_weights_config_placeholder()
        (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        from safetensors.torch import save_file

        save_file({}, str(dest / "model.safetensors"), metadata={"format": "pt"})
    # generation_config.json of the tiny model: the tiny EOS ids (the package's name the real vocab).
    ids = {t: fast.convert_tokens_to_ids(t) for t in SPECIALS}
    (dest / "generation_config.json").write_text(json.dumps({
        "eos_token_id": [ids["<|im_end|>"], ids["<|endoftext|>"]], "pad_token_id": ids["<|endoftext|>"],
    }, indent=2) + "\n", encoding="utf-8")


_TINY_CONFIG_JSON: dict | None = None


def with_weights_config_placeholder() -> dict:
    assert _TINY_CONFIG_JSON is not None
    return _TINY_CONFIG_JSON


def expand_image_tokens(ids: list[int], image_token: int, counts: list[int]) -> list[int]:
    out, k = [], 0
    for i in ids:
        if i == image_token:
            out.extend([image_token] * counts[k])
            k += 1
        else:
            out.append(i)
    assert k == len(counts), (k, counts)
    return out


class GeneratedOnlyPresence(LogitsProcessor):
    """vLLM's additive/count-independent penalty excludes the input prompt (unlike repetition).

    HF's generic generate call does not apply vLLM presence penalties by itself. Keep this
    oracle explicit so text-mode goldens exercise pup.PRESENCE_PENALTY, not bare argmax.
    """

    def __init__(self, prompt_length: int, penalty: float):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        for row, history in enumerate(input_ids):
            seen = history[self.prompt_length:].unique()
            scores[row, seen] -= self.penalty
        return scores


def verify_presence_processor():
    # Independent discriminating examples: prompt ids do not count, generated ids count once,
    # and editing's zero penalty is the identity. Fails if a dump silently reverts to bare argmax.
    scores = torch.tensor([[4.0, 3.0, 2.0]])
    processor = GeneratedOnlyPresence(3, 1.5)
    assert torch.equal(processor(torch.tensor([[0, 0, 1]]), scores), scores)
    assert torch.equal(processor(torch.tensor([[0, 0, 1, 0, 0, 2]]), scores),
                       torch.tensor([[2.5, 3.0, 0.5]]))
    assert torch.equal(GeneratedOnlyPresence(3, 0.0)(torch.tensor([[0, 0, 1, 2]]), scores), scores)
    assert torch.equal(scores, torch.tensor([[4.0, 3.0, 2.0]])), "processor mutated caller's logits"


def run_tiny(model, fast, processor_cfg: dict, prompts: dict, case: dict, references: list[np.ndarray]) -> dict:
    system_key, text, locked = bbox.plan(case["prompt"], case["boxes"], len(references))
    if references:
        scaffold = pup.I2I_SCAFFOLD[locked is not None].format(instruction=text)
        content = [{"type": "image"} for _ in references] + [{"type": "text", "text": scaffold}]
        messages = [{"role": "system", "content": prompts[system_key]}, {"role": "user", "content": content}]
    else:
        messages = [{"role": "system", "content": prompts[system_key]}, {"role": "user", "content": text}]
    chat = fast.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = fast.encode(chat, add_special_tokens=False)
    image_token = fast.convert_tokens_to_ids("<|image_pad|>")
    inputs = {}
    if references:
        ip = Qwen2VLImageProcessor(**{k: v for k, v in processor_cfg.items() if k != "image_processor_type"})
        pixels = ip(images=[Image.fromarray(r, "RGB") for r in references], return_tensors="pt")
        grid = pixels["image_grid_thw"]
        counts = [int(t * h * w) // (ip.merge_size ** 2) for t, h, w in grid.tolist()]
        ids = expand_image_tokens(ids, image_token, counts)
        inputs["pixel_values"] = pixels["pixel_values"]
        inputs["image_grid_thw"] = grid
        # What the processor returns beside `input_ids` (transformers >= 5): 1 marks an image token
        # for the M-RoPE position computation.
        inputs["mm_token_type_ids"] = torch.tensor([[1 if i == image_token else 0 for i in ids]], dtype=torch.long)
    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        prefill = model(input_ids=input_ids, **inputs)
        last_logits = prefill.logits[0, -1].float()
        penalty = pup.PRESENCE_PENALTY["i2i" if references else "t2i"]
        processors = LogitsProcessorList([GeneratedOnlyPresence(input_ids.shape[1], penalty)])
        gen = model.generate(input_ids=input_ids, **inputs, max_new_tokens=GREEDY_STEPS, do_sample=False,
                             logits_processor=processors, output_scores=True,
                             return_dict_in_generate=True, eos_token_id=[fast.convert_tokens_to_ids("<|im_end|>"), fast.convert_tokens_to_ids("<|endoftext|>")])
    generated = gen.sequences[0, input_ids.shape[1]:].tolist()
    margins = []
    for score in gen.scores:
        top = torch.topk(score[0].float(), 2).values
        margins.append(float(top[0] - top[1]))
    return {
        "system_key": system_key, "chat_text": chat, "ids": ids, "last_logits": last_logits,
        "generated": generated, "margins": margins[: len(generated)], "inputs": inputs,
        "decoded": fast.decode(generated, skip_special_tokens=True),
    }


def main() -> None:
    verify_presence_processor()
    global _TINY_CONFIG_JSON
    src = upstream_root()
    package = package_root()
    processor = AutoProcessor.from_pretrained(str(package), local_files_only=True)
    tok = processor.tokenizer
    prompts = json.loads((package / "system_prompts.json").read_text(encoding="utf-8"))
    template = (package / "chat_template.jinja").read_text(encoding="utf-8")
    fast = tiny_tokenizer(template)
    import PIL
    import transformers

    golden: dict = {
        "upstream": {
            "code_commit": FLUX3_COMMIT,
            "pup_revision": PUP_REVISION,
            "source_sha256": PUP_SOURCE_SHA256,
            "package_sha256": PUP_PACKAGE_SHA256,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "pillow": PIL.__version__,
        },
        "sampling": dict(pup.SAMPLING, presence_penalty=pup.PRESENCE_PENALTY, max_model_len=pup.ENGINE["max_model_len"],
                         max_references=pup.MAX_REFERENCES, stop_tokens=list(pup.STOP_TOKENS),
                         reference_multiple=pup.REFERENCE_MULTIPLE, max_aspect_ratio=pup.MAX_ASPECT_RATIO,
                         jpeg_quality=pup.JPEG_QUALITY, max_pixels_by_refs={str(k): v for k, v in pup.MAX_PIXELS_BY_REFS.items()}),
        "scaffolds": {"plain": pup.I2I_SCAFFOLD[False], "locked": pup.I2I_SCAFFOLD[True]},
        "stop_token_ids": [tok.convert_tokens_to_ids(t) for t in pup.STOP_TOKENS],
        "plan": [],
        "reconcile": [],
        "detect": [],
        "bypass": [],
        "to_canvas_errors": [],
        "reference_size": reference_size_cases(),
        "vision_tokens": vision_token_cases(processor),
    }

    for case in plan_cases(src):
        n = case["n_images"]
        system_key, text, locked = bbox.plan(case["prompt"], case["boxes"], n)
        finished = not case["boxes"] and bbox.is_finished(case["prompt"])
        if n:
            scaffold = pup.I2I_SCAFFOLD[locked is not None].format(instruction=text)
            content = [{"type": "image"} for _ in range(n)] + [{"type": "text", "text": scaffold}]
            messages = [{"role": "system", "content": prompts[system_key]}, {"role": "user", "content": content}]
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            user_text = scaffold
        else:
            messages = [{"role": "system", "content": prompts[system_key]}, {"role": "user", "content": text}]
            chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            user_text = text
            assert chat == processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        golden["plan"].append({
            "name": case["name"], "prompt": case["prompt"], "n_images": n,
            # Inputs and outputs that are JSON *text* upstream travel as strings (the golden file
            # is re-serialized with sorted keys, which would lose the row key order).
            "boxes_json": json.dumps(case["boxes"], ensure_ascii=False) if case["boxes"] is not None else None,
            "system_key": system_key, "user_text": user_text, "locked": json.dumps(locked, ensure_ascii=False) if locked is not None else None,
            "finished": finished, "chat_text": chat,
            "chat_tokens_real": len(tok.encode(chat, add_special_tokens=False)),
            "chat_tokens_tiny": len(fast.encode(chat, add_special_tokens=False)),
            "bypass": f"{case['prompt'].rstrip()} {json.dumps(bbox.to_canvas(case['boxes']), ensure_ascii=False)}" if case["boxes"] else case["prompt"],
        })

    for case in reconcile_cases():
        golden["reconcile"].append({
            "name": case["name"], "text": case["text"],
            "locked_json": json.dumps(case["locked"], ensure_ascii=False),
            "expect": bbox.reconcile(case["text"], json.loads(json.dumps(case["locked"]))),
        })
    for case in detect_cases():
        found = bbox.detect(case["prompt"])
        caption, rows = bbox.split(case["prompt"])
        golden["detect"].append({
            "prompt": case["prompt"],
            "expect": {"prompt": found[0], "rows_json": json.dumps(found[1], ensure_ascii=False)} if found else None,
            "finished": bbox.is_finished(case["prompt"]),
            "split": {"caption": caption, "rows_json": json.dumps(rows, ensure_ascii=False) if rows is not None else None},
        })
    for case in bypass_cases(src):
        boxes = case["boxes"]
        golden["bypass"].append({
            "prompt": case["prompt"],
            "boxes_json": json.dumps(boxes, ensure_ascii=False) if boxes is not None else None,
            "expect": f"{case['prompt'].rstrip()} {json.dumps(bbox.to_canvas(boxes), ensure_ascii=False)}" if boxes else case["prompt"],
        })
    for case in to_canvas_error_cases():
        try:
            bbox.to_canvas(case["boxes"])
            raise SystemExit(f"expected to_canvas to refuse {case}")
        except ValueError as error:
            golden["to_canvas_errors"].append({"boxes_json": json.dumps(case["boxes"], ensure_ascii=False), "error": str(error)})

    GEN_CORE_FIXTURES.mkdir(parents=True, exist_ok=True)
    write_json(GEN_CORE_FIXTURES / "golden.json", golden)

    tensors, meta, policy_out = policy_fixture()
    save_safetensors(GEN_CORE_FIXTURES / "policy_reference.safetensors", tensors, meta)

    # ---- the miniature VLM: seeded weights, greedy runs over the four modes ------------------
    ids = {t: fast.convert_tokens_to_ids(t) for t in SPECIALS}
    cfg = tiny_config(len(fast), ids)
    model = Qwen3_5ForConditionalGeneration(cfg).eval()
    seed_model(model)
    processor_cfg = json.loads((package / "preprocessor_config.json").read_text(encoding="utf-8"))
    cases = {c["name"]: c for c in plan_cases(src)}
    refs = {
        "t2i": [], "t2i_bbox": [],
        "i2i": [policy_out["rgb_70x50_n1"]],
        "i2i_bbox": [policy_out["rgb_40x60_n2"], policy_out["rgb_64x32_n2"]],
    }
    mlx_tensors, mlx_meta = {}, {"greedy_steps": GREEDY_STEPS, "pup_revision": PUP_REVISION,
                                 "code_commit": FLUX3_COMMIT, "presence_penalty_scope": "generated_only"}
    for name, images in refs.items():
        run = run_tiny(model, fast, processor_cfg, prompts, cases[name], images)
        mlx_tensors[f"ids_{name}"] = torch.tensor(run["ids"], dtype=torch.int32)
        mlx_tensors[f"logits_{name}"] = run["last_logits"]
        mlx_tensors[f"generated_{name}"] = torch.tensor(run["generated"], dtype=torch.int32)
        mlx_tensors[f"margins_{name}"] = torch.tensor(run["margins"], dtype=torch.float32)
        for i, image in enumerate(images):
            mlx_tensors[f"ref_{name}_{i}"] = torch.from_numpy(image.copy())
        if "image_grid_thw" in run["inputs"]:
            mlx_tensors[f"grid_{name}"] = run["inputs"]["image_grid_thw"].to(torch.int32)
        mlx_meta[f"system_key_{name}"] = run["system_key"]
        mlx_meta[f"chat_text_{name}"] = run["chat_text"]
        mlx_meta[f"decoded_{name}"] = run["decoded"]
        mlx_meta[f"n_refs_{name}"] = len(images)
        mlx_meta[f"presence_penalty_{name}"] = pup.PRESENCE_PENALTY["i2i" if images else "t2i"]
        print(f"{name}: {len(run['ids'])} prompt ids, generated {run['generated']} (margins {[round(m, 4) for m in run['margins']]})")
    _TINY_CONFIG_JSON = json.loads(model.config.to_json_string())
    write_package_files(TINY_SNAPSHOT, package, fast, model)
    write_package_files(TINY_PACKAGE, package, fast, None)
    mlx_meta["tokenizer_sha256"] = sha256_of(TINY_SNAPSHOT / "tokenizer.json")
    assert sha256_of(TINY_PACKAGE / "tokenizer.json") == mlx_meta["tokenizer_sha256"]
    save_safetensors(MLX_GOLDEN, mlx_tensors, mlx_meta)


if __name__ == "__main__":
    main()
