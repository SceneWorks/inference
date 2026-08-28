"""Is the LTX-2.5 packed text encoder usable as a prompt *enhancer*? — sc-18764.

sc-18764 was planned on the premise that LTX-2.5 prompt enhancement runs on the **same**
self-contained `gemma4-12b-with-proj` checkpoint the 2.5 text encoder is built from, the way
LTX-2.3's censored enhancer reuses its already-loaded Gemma 3 backbone
(`mlx-gen-ltx/src/enhance.rs`). This probe is the evidence that it does not, and it exists so the
finding is reproducible rather than a claim in a story comment.

It drives the **upstream** enhancer — Lightricks/LTX-2 @ `d151147788a9284cca791edc6ce898007e727fe6`
(v1.2.0), `ltx_core.text_encoders.gemma.encoders.base_encoder.LTXGemmaTextEncoder.enhance_t2v`,
with upstream's own `GEMMA4_ENHANCE_GENERATION_KWARGS` (greedy, `no_repeat_ngram_size=5`) and
upstream's own loader (`get_gemma_ops` + `GemmaTextEncoderConfigurator` + the module ops) — over
two checkpoints, and prints what each generates:

* `LTX25_TE_FILE` — the packed LTX-2.5 encoder (`gemma4-12b-with-proj`, or a SceneWorks tier's
  `text_encoder.safetensors`, which carries the same tensors).
* `GEMMA4_INSTRUCT_ROOT` — *optional* control: a stock generative Gemma 4 instruct root, e.g. an
  HF snapshot of `google/gemma-4-12B-it`. Same `model_type`, same code path.

Measured 2026-08-27 on CPU/bf16, prompt `"a red fox darting through a snowy pine forest at dawn"`,
`max_new_tokens=24`, seed 42:

* stock `google/gemma-4-12B-it`  → `"A wide shot captures a vibrant red fox with a thick, bushy
  tail and pointed ears darting rapidly through a dense pine"`
* LTX-2.5 packed TE              → `"SSSS…"` (one token repeated)

Both go through the identical loader and the identical `_enhance`, so the loader, the chat
template, the processor and the generation config are all controlled for. The packed TE is an
encoder fine-tune: its early layers are bit-identical to stock, while `embed_tokens` (which is the
tied LM head), `model.norm` and the late blocks are retuned — so it no longer carries a usable
generative head.

This is also what upstream itself encodes: `get_gemma_ops` documents `gemma4_unified` as
"encode only" and `gemma4` (dense instruct) as "enhance only", and
`ltx_pipelines/utils/blocks.py` refuses `enhance_first_prompt` when the encode root is not
`gemma3`, directing the caller at `--prompt-enhancer-gemma-root`. LTX-2.5 prompt enhancement
therefore needs a **separate** instruct checkpoint, staged the way LTX-2.3 stages its
`uncensored_enhancer` component — an asset/manifest decision, not a port.

One deliberate patch is applied: upstream's `_default_generation_kwargs` /
`_default_system_prompt` accept only `model_type in {"gemma3", "gemma4"}` and raise for
`"gemma4_unified"`. Both checkpoints probed here are `gemma4_unified`, so that allowlist is
widened — and nothing else is, which is why the stock control still produces a coherent rewrite.

Run:
    LTX2_SRC=<checkout>/packages/ltx-core/src \\
    LTX25_TE_FILE=<...>/text_encoder.safetensors \\
    GEMMA4_INSTRUCT_ROOT=<...>/models--google--gemma-4-12B-it/snapshots/<hash> \\
      python3 tools/probe_ltx25_enhancer_capability.py

Needs torch + `transformers==5.14.1` (5.0.0 has no gemma4). Prints only; writes nothing.
"""

from __future__ import annotations

import os
import time

import torch

from _ltx25_diffvae_ref import REFERENCE_COMMIT, ltx_core_on_path
from _paths import require_env

#: The one prompt both checkpoints are asked to rewrite.
PROMPT = "a red fox darting through a snowy pine forest at dawn"

#: Short enough to be affordable on CPU, long enough that a degenerate model is unmistakable.
MAX_NEW_TOKENS = 24

#: Upstream's `enhance_t2v` seed. Inert under greedy decoding; passed so the call matches upstream.
SEED = 42

ltx_core_on_path()

from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader  # noqa: E402
from ltx_core.text_encoders.gemma import (  # noqa: E402
    GemmaTextEncoderConfigurator,
    get_gemma_ops,
)
from ltx_core.text_encoders.gemma.encoders.base_encoder import (  # noqa: E402
    GEMMA4_ENHANCE_GENERATION_KWARGS,
    LTXGemmaTextEncoder,
    default_gemma4_i2v_system_prompt,
    default_gemma4_t2v_system_prompt,
)
from ltx_core.text_encoders.gemma.gemma_assets import (  # noqa: E402
    resolve_gemma_weight_paths,
)


def widen_allowlist_to_unified() -> None:
    """Accept `gemma4_unified` wherever upstream accepts only dense `gemma4` — see the module docs.

    Only the two `model_type` dispatches are widened. The generation kwargs and the system prompts
    they return are upstream's own, unmodified, so a checkpoint that *can* enhance still does.
    """
    real_kwargs = LTXGemmaTextEncoder._default_generation_kwargs
    real_prompt = LTXGemmaTextEncoder._default_system_prompt

    def kwargs(self):
        if self._model_type() == "gemma4_unified":
            return dict(GEMMA4_ENHANCE_GENERATION_KWARGS)
        return real_kwargs(self)

    def prompt(self, *, t2v: bool):
        if self._model_type() == "gemma4_unified":
            return default_gemma4_t2v_system_prompt() if t2v else default_gemma4_i2v_system_prompt()
        return real_prompt(self, t2v=t2v)

    LTXGemmaTextEncoder._default_generation_kwargs = kwargs
    LTXGemmaTextEncoder._default_system_prompt = prompt


def load_encoder(path: str, dtype: torch.dtype = torch.bfloat16) -> LTXGemmaTextEncoder:
    """Upstream's own load path: `get_gemma_ops` → configurator → `assign=True` → module ops."""
    loader = SafetensorsModelStateDictLoader()
    sd_ops, module_ops = get_gemma_ops(path)
    encoder = GemmaTextEncoderConfigurator.with_gemma_model_path(path).from_metadata({})

    sd: dict[str, torch.Tensor] = {}
    for weight_path in resolve_gemma_weight_paths(path):
        sd.update(loader.load(weight_path, sd_ops).sd)
    # Cast in place: a 26 GB encoder cannot hold two copies of the dict at once.
    for key in list(sd):
        sd[key] = sd.pop(key).to(dtype)

    missing, unexpected = encoder.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if "inv_freq" not in k and not k.endswith("embed_scale")]
    if missing or unexpected:
        raise SystemExit(f"{path}: missing={missing[:8]} unexpected={unexpected[:8]}")

    for ops in module_ops:
        if ops.matcher(encoder):
            encoder = ops.mutator(encoder)
    return encoder.to(dtype=dtype).eval()


def probe(label: str, path: str) -> None:
    print(f"\n=== {label}\n    {path}", flush=True)
    encoder = load_encoder(path)
    print(f"    model_type={encoder._model_type()!r}", flush=True)
    started = time.time()
    enhanced = encoder.enhance_t2v(PROMPT, max_new_tokens=MAX_NEW_TOKENS, seed=SEED)
    print(f"    {time.time() - started:.0f}s -> {enhanced!r}", flush=True)


def main() -> None:
    te_file = require_env(
        "LTX25_TE_FILE",
        "the packed LTX-2.5 text encoder (gemma4-12b-with-proj, or a tier's "
        "text_encoder.safetensors)",
    )
    print(f"[ref] Lightricks/LTX-2 @ {REFERENCE_COMMIT} (v1.2.0)")
    print(f"[ref] prompt={PROMPT!r} max_new_tokens={MAX_NEW_TOKENS} greedy")
    widen_allowlist_to_unified()

    if instruct_root := os.environ.get("GEMMA4_INSTRUCT_ROOT"):
        probe("CONTROL: stock Gemma 4 instruct", instruct_root)
    else:
        print(
            "\n[skip] GEMMA4_INSTRUCT_ROOT is unset, so the generative control is NOT running — "
            "the LTX result below then has nothing to be compared against."
        )
    probe("LTX-2.5 packed text encoder", te_file)


if __name__ == "__main__":
    main()
