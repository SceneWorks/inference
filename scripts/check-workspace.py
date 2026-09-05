#!/usr/bin/env python3
"""Fail when the normalized inference workspace drifts from its graph invariants."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MEMBER_COUNT = 95
INTERNAL_PACKAGES = {
    "candle-audio",
    "candle-audio-catalog",
    "candle-audio-kokoro",
    "candle-audio-moss-sfx",
    "candle-gen-catalog",
    "core-llm",
    "core-llm-testkit",
    "mlx-gen-catalog",
    "mlx-llm",
    "candle-llm",
    "sceneworks-gen-core",
    "sceneworks-gen-core-testkit",
    "runtime-catalog",
    "runtime-macos",
    "runtime-cpu",
    "runtime-cuda",
}
PINNED_WORKSPACE_DEPENDENCIES = {
    "mlx-rs": ("pmetal-mlx-rs", "bd8f0e3c757195b17b2c34fae3073ab826fb7bc1"),
    "mlx-sys": ("pmetal-mlx-sys", "bd8f0e3c757195b17b2c34fae3073ab826fb7bc1"),
    "candle-core": ("candle-core", "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"),
    "candle-nn": ("candle-nn", "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"),
    "candle-transformers": ("candle-transformers", "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"),
    "candle-flash-attn": ("candle-flash-attn", "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"),
}
DEFAULT_GRAPH_PINNED_PACKAGES = {
    package_name: revision
    for dependency_name, (package_name, revision) in PINNED_WORKSPACE_DEPENDENCIES.items()
    if dependency_name != "candle-flash-attn"
}
FORBIDDEN_GRAPH_PACKAGES = {
    # Provider composition is ordinary, value-scoped source code. Reintroducing this crate would
    # make linker participation part of the supported runtime graph again.
    "inventory",
}
# Directory names whose subtrees are not part of this checkout's single workspace: the git store
# and build output (.git, target), plus agent tooling that nests its own gitignored worktrees --
# each a separate checkout carrying its own Cargo.lock/manifest (.claude, .codex). They must not
# be swept into the single-lockfile / single-manifest invariants below.
IGNORED_TREE_PARTS = frozenset({".git", "target", ".claude", ".codex"})

# --- epic 13657 guardrail: inference never fetches weights and never derives a download-cache
# location. Every model component is a caller-provisioned local path (WeightsSource::Dir / File);
# fetching and cache placement are the consumer's job, so user-supplied models at arbitrary paths
# must load. The three assertions below (network-client ban, HF-cache source lint, deleted
# env-side-channel pins) turn that contract into an enforced boundary. See
# docs/architecture/inference-rearchitecture.md.

# No network/HTTP client may resolve anywhere in the graph. Mirrors the `inventory` whole-graph
# ban above: a client reachable through an enabled third-party feature (e.g. a candle `hf-hub`
# feature) would reopen self-fetch, so pin the feature off rather than weaken this. All nine are
# confirmed absent on main; extend this set, never trim it. If a denied crate ever becomes a
# legitimate *transitive-only* dep of a build tool, narrow to the direct/feature-activated
# workspace-member scope reported by check_network_clients rather than deleting the entry.
FORBIDDEN_NETWORK_CLIENT_PACKAGES = {
    "hf-hub",
    "reqwest",
    "ureq",
    "curl",
    "curl-sys",
    "git2",
    "isahc",
    "attohttpc",
    "hyper",
}

# Whole-tree substring bans over every workspace member's Rust (src, tests, examples, testkits --
# NO allow-list, per the sharpened rule). These name an HF download cache or its client, which
# inference must never reference. Precision notes:
#   * `.cache/huggingface` is the SPECIFIC HF cache path -- NOT a blanket `.cache/` ban, so the
#     legitimate `~/.cache/mlx-gen-seedvr2-golden` test-golden dir does not trip.
#   * bare `HUGGINGFACE` is deliberately NOT banned: it false-positives on legitimate
#     `https://huggingface.co/...` `source_url` attribution, `huggingface-cli` doc prose, repo IDs,
#     and license text. `.cache/huggingface` + `hf_hub` already cover the real cache-derivation cases.
#   * `Api::new` is the hf_hub API constructor; no non-hf `Api::new` exists on the clean tree, so
#     the bare token is safe. If a legitimate unrelated `Api::new` ever appears, qualify it to the
#     `hf_hub`-scoped form rather than dropping the ban.
RUST_BANNED_SUBSTRINGS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    ".cache/huggingface",
    "hf_hub",
    "Api::new",
)

# Env vars that were DELETED as production self-fetch / cache-derivation side channels. They must
# not return as production reads. Scoped to production `crates/**/src/**` EXCLUDING `#[cfg(test)]`
# modules, and matched only as actual `env::var("NAME")` reads (not doc-comment prose that merely
# names a removed var), because:
#   * MOSS_XY_TOKENIZER_SNAPSHOT / MOSS_AUDIO_TOKENIZER_SNAPSHOT legitimately persist TEST-SIDE
#     (sc-13660/sc-13662): each moss crate's tests/conformance.rs reads them as explicit passed-in
#     snapshot paths, and they are keys in release/real-weight-models.toml + real-weights.yml that
#     provision the weekly runner. A passed-in test path is allowed; cache DERIVATION is not.
#   * LTX_GEMMA_DIR is read only inside a `#[cfg(test)]` real-weight harness (mlx-gen-ltx
#     src/training.rs) as a test-only convenience path; its production fallback was deleted.
#   * SENSENOVA_DISTILL_LORA / PULID_FLUX_WEIGHTS / LTX_UNCENSORED_GEMMA_DIR still appear in `src/`
#     ONLY as doc prose or `#[cfg(test)]` assertions proving the deleted var is NOT resurrected --
#     never as an env::var read -- so shape-matching + cfg(test) stripping keeps them green.
# Legitimate passed-in-path env vars (MLX_LLM_TEST_MODEL, per-crate *_SNAPSHOT/*_SNAPSHOT_DIR,
# MLX_GEN_MODELS_ROOT, CANDLE_GEN_MODELS_ROOT, tuning knobs LTX_MAX_LATENT_TOKENS/LTX_VAE_BUDGET_GIB)
# are NOT banned: this targets cache-location derivation + the deleted side channels, not all reads.
DELETED_ENV_SIDE_CHANNELS = (
    "PERTH_SNAPSHOT",
    "MOSS_XY_TOKENIZER_SNAPSHOT",
    "MOSS_AUDIO_TOKENIZER_SNAPSHOT",
    "SENSENOVA_DISTILL_LORA",
    "LTX_GEMMA_DIR",
    "PULID_FLUX_WEIGHTS",
    "LTX_UNCENSORED_GEMMA_DIR",
    "PULID_EVA_WEIGHTS",
    "PULID_FACE_WEIGHTS_DIR",
)

# The crate that owns the shared PiD decode seam and, since sc-15775, the shared per-route decode
# domain. A direct dependency on it is what makes a provider PiD-eligible.
PID_SEAM_CRATE = "mlx-gen-pid"

# Evidence that a provider actually CALLS the shared per-route decode API — not that it mentions it.
#
# The first revision of this gate matched the bare substrings "DecodeRoutes" / "decode_routes", which
# the adversarial review defeated three ways: a `// TODO(sc-15775): should use DecodeRoutes one day.`
# on an otherwise non-conforming adopter passed; so did `#[allow(unused)] use
# mlx_gen_pid::decode_routes;`. A gate whose stated purpose is to replace doc-comment enforcement must
# not be satisfiable by a doc comment, so matching now happens on COMMENT-STRIPPED source and requires
# call syntax.
#
# `DecodeRoutes::new` is the *checked* constructor: since sc-15775 it returns `Result` and refuses a
# native ladder that reaches into the PiD student's tile domain, so calling it (or
# `assert_decode_routes`, its panicking test-suite form) is at once the construction evidence and the
# conformance evidence. There is deliberately no third, separately-skippable conformance call to look
# for — requiring one would be ceremony now that construction cannot be performed unchecked.
PID_DECODE_ROUTE_CONSTRUCTION_MARKERS = ("DecodeRoutes::new", "assert_decode_routes")

# The admission half. Declaring the routes and then not gating on them leaves the hazard wide open, so
# a construction call alone is not adoption. Matched as a call on a route-named receiver rather than a
# bare `.validate(` so an unrelated `foo.validate(...)` elsewhere in the crate cannot stand in for it.
PID_DECODE_ROUTE_ADMISSION_CALL = re.compile(
    r"(?:decode_routes|DecodeRoutes|routes)\b[^;{}]{0,200}?\.\s*validate\s*\(",
    re.DOTALL,
)

# Any one of these in the crate's own sources means it implements rung 2 (bounded decode) and can
# therefore reach the seam with a decode geometry.
#
# Broader than the `decode_tile_edges` field name alone, which the review defeated as attack (c): an
# adopter that builds its `MemoryParameterRanges` through a shared helper never writes that literal.
# What it cannot delegate is the executable half — its own `MemoryRequestScope::configure_decode`, the
# hook the shared runtime drives to apply a bounded-decode selection — so that spelling is in the set
# too. Deliberately fail-closed: over-triggering asks a provider for a three-line declaration, while
# under-triggering ships the defect.
PID_RUNG_TWO_MARKERS = (
    "decode_tile_edges",
    "decode_overlaps",
    "MemoryParameterRanges",
    "BoundedDecode",
    "configure_decode",
)

# --- Cross-backend geometry parity (sc-19419) -----------------------------------------------------
#
# A model family implemented on two backends declares the same published geometry twice, in crates
# that cannot see each other: `mlx-gen-*` builds on macOS only, so no dependency edge — in either
# direction — can exist between the pair. Every existing guard is therefore per-crate, and a per-crate
# assertion is structurally incapable of noticing that its sibling declares something else.
#
# It found exactly that: `candle-gen-minimax-h3` advertised `SIZE_MULTIPLE = 16` while
# `mlx-gen-minimax-h3` advertised 32, and the candle crate's own test *asserted the 16* under a doc
# comment claiming both backends pinned the same geometry. 16 is the VAE-only alignment, correct while
# that crate was VAE-only and stale from the moment its DiT landed in sc-17155; 32 is
# `vae_spatial_compression_ratio * patch_size[2]`. A 16-aligned canvas survives the VAE and then has an
# odd number of latent columns with no patched representation at all.
#
# This gate lives here rather than in either crate because `scripts/ci/select_lanes.py` marks the
# `workspace` lane "never skip" — it is the only enforcement point that runs on every change, on Linux,
# without a macOS runner. A Rust test in the candle crate would be skipped by path-filtered lanes on an
# mlx-only edit, and one in the mlx crate would not run on Linux at all.
# sc-19496 replaced the hand-maintained pair table and file list this gate shipped with. Both were
# the same defect shape the gate exists to catch: a curated list is a claim about coverage that
# nothing checks, and it shrinks silently. A family left out of the table, or a declaration moved to
# a file outside the list, was simply not compared — and the gate still printed OK.
#
# So both are derived now:
#
#   * the families come from `cargo metadata --no-deps --offline` — every package named
#     `candle-gen-X` whose `mlx-gen-X` sibling is also a workspace member. No family can be omitted,
#     because omitting one would mean deleting the crate.
#   * the declarations come from every `*.rs` under each crate's `src/`. Nothing to keep in sync,
#     and moving a constant between files cannot move it out of the gate's reach.
#
# Only the *shared* names are compared: the two backends legitimately declare different constants
# (mlx-gen carries Metal memory registrations and tile geometry candle has no analogue for, and vice
# versa), so name-set equality across `src/` would be a permanent red rather than a signal. Value
# agreement on the names both sides *do* declare is the checkable claim.
CROSS_BACKEND_GEOMETRY_EXEMPT_FAMILIES: dict[str, str] = {}

# A family whose two crates share *no* constant name is compared against nothing, and every other
# clause below still passes — the exact inert shape this gate exists to refuse, one family at a time
# and printing OK the whole way. So zero shared comparisons is a failure, and a family that genuinely
# has none says so here, with a reason the gate can invalidate: listing a family that *does* share a
# constant is itself a violation, so this table cannot outlive its subject either.
#
# Both entries are real, and both were reached by reading the crates rather than by finding the gate
# inconvenient.
CROSS_BACKEND_GEOMETRY_NO_SHARED_CONSTANTS: dict[str, str] = {
    "joycaption": (
        "`mlx-gen-joycaption` declares no constant at all, of any visibility: its whole surface is "
        "`mlx_gen::register_captioner! { pub(crate) const REGISTRATION = descriptor => load }` "
        "(`src/model.rs`), a macro that expands to the registration and reads the prompt menu out "
        "of the shared `joycaption-prompts` crate. The candle side's eight constants are that menu "
        "and its templates, which mlx consumes rather than redeclares — so there is one declaration "
        "of this family's geometry, not two, and nothing for a cross-backend comparison to hold. If "
        "the mlx crate ever declares its own copy, this entry goes stale and the gate says so."
    ),
    "sam3": (
        "both crates declare their ~40 shared geometry names (`INPUT_SIZE`, `NUM_HEADS`, "
        "`MEM_ATTN_LAYERS`, `ROPE_THETA`, `LN_EPS`, the tracker thresholds) as *private* module "
        "`const`s, which `RUST_PUB_CONST` deliberately does not read — see its comment for why "
        "visibility is the line. They are not invisible on purpose and they are not a divergence: "
        "measured across the pair, every one of those ~40 names agrees today. Reading private "
        "consts workspace-wide is sc-19696, which has to triage 45 other families' divergences "
        "first. Until then this family is uncompared, and that is recorded here rather than "
        "printed as OK."
    ),
}

# Constants a family declares on both sides that genuinely differ. Each was read with its doc
# comment before being listed; the reason is what that reading found. Nothing lands here for being
# inconvenient — a divergence with no reason is a defect (sc-19419 was exactly one).
#
# The gate fails on a *stale* entry too, wherever it can see one: an exemption for a family that no
# longer exists, or for a constant no longer declared on both sides, always fails. An exemption for a
# constant that "now agrees" can only fail where both sides canonicalize to the same form, which
# excludes three entries below. `mage.ATTENTION_CHUNK_SIZE` is one: candle's value is the
# cross-crate `gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET` (64 Mi today, at
# `crates/contracts/gen-core/src/attention_budget.rs`), and `_const_numbers` resolves same-crate
# names only — so changing that budget to mlx's 16 Mi would make this exemption stale with nothing
# to notice. Both `wan` `*_VAE_TILING` entries are the other two: each side's value is an associated
# const reached through its own provider type, and the qualifier is kept as text, so they compare
# unequal by spelling whatever the geometry does. `wan.VAE_TILING` itself is now compared directly,
# which covers the substance; the two re-exports are still text.
CROSS_BACKEND_GEOMETRY_EXEMPTIONS: dict[tuple[str, str], str] = {
    ("catalog", "BESPOKE_UTILITY_CRATES"): (
        "each backend's own inventory of platform-owned crates, not a shared declaration: candle "
        "ships `pulid`, mlx ships it as `pulid_flux` and ships `sam2`, which candle has no port of. "
        "Both doc comments already say so."
    ),
    ("chroma", "DEFAULT_SAMPLER"): (
        "candle advertises the legacy `flow_match` alias, mlx advertises `euler`. Both resolve to "
        "the same integrator — `candle_gen::menu_with_aliases` documents `flow_match` as falling "
        "back to euler — but the advertised strings differ, and which one is right against the "
        "released checkpoint is sc-19495. Exempted so the value question stays on that story "
        "instead of being settled by whichever backend someone copies."
    ),
    ("flux", "DECODE_OVERLAP"): (
        "per-backend measured VAE tile overlap: candle 128 on CUDA, mlx 64 on Metal. Measured "
        "quantities, not a published geometry either backend could read off the checkpoint."
    ),
    ("flux", "DECODE_TILE_EDGES"): (
        "candle decodes the whole frame at one edge (`&[DECODE_TILE_EDGE]`, with a doc comment "
        "explaining that FLUX GroupNorm makes independent spatial tiles numerically unsafe); mlx "
        "runs the shared head-once/tail-tiled decoder over three candidate edges."
    ),
    ("flux2", "CALIBRATION_FINGERPRINT"): (
        "a calibration *identity*, deliberately per-backend — it names the hardware and the "
        "measurement campaign the memory record came from. Two backends sharing one would be the "
        "defect."
    ),
    ("flux2", "DECODE_OVERLAP"): (
        "candle's 1 is documented as an inert sentinel: at its 1024px full-edge cell no neighboring "
        "tiles exist, and the shared contract requires a positive overlap. mlx's 128 is a real "
        "measured overlap between real tiles."
    ),
    ("flux2", "DECODE_TILE_EDGE"): (
        "candle 1024 is the full output edge of the calibration cell — deliberately not a spatial "
        "partition, per its doc comment. mlx 512 is a real tile edge."
    ),
    ("flux2", "DECODE_TILE_EDGES"): (
        "follows DECODE_TILE_EDGE: candle's list is the single full edge, mlx's is three measured "
        "tile edges."
    ),
    ("kolors", "SIZE_MULTIPLE"): (
        "candle validates only the SDXL VAE /8 stride; mlx enforces the structural "
        "`PRODUCTION_SPATIAL_MULTIPLE` of 32 (`8 * 2^2`, the U-Net's two exact skip joins) — the "
        "same deliberate strict/loose split as sdxl's SIZE_MULTIPLE below."
    ),
    ("lens", "MEMORY_CALIBRATION_FINGERPRINT"): (
        "a calibration identity, per-backend by construction — the candle value names the CUDA "
        "campaign and the mlx value names the Metal one."
    ),
    ("ltx", "MODEL_ID"): (
        "the two backends ship different checkpoints of the family and register different ids: "
        "candle the single-stage `ltx_2_3_distilled`, mlx the two-stage `ltx_2_3`. Both doc "
        "comments state which."
    ),
    ("ltx", "MODEL_25_ID"): (
        "LTX 2.5 deliberately preserves the same backend-split engine-id contract as 2.3: "
        "candle registers `ltx_2_5_distilled`, while mlx registers `ltx_2_5`. Shortcut story "
        "sc-18778 records these exact public ids, and SceneWorks maps the shared model to each "
        "backend-specific id."
    ),
    ("ltx", "CALIBRATION_FINGERPRINT"): (
        "a calibration identity is backend-specific: Candle names the released CUDA q4 I2V cell, "
        "while MLX names the Metal base/Eros I2V cell. Sharing a fingerprint would cross the "
        "physical artifact and backend evidence domains."
    ),
    ("ltx", "CALIBRATED_TIER"): (
        "the companion of the CALIBRATION_FINGERPRINT exemption above, one level down: this names "
        "WHICH tier that backend-specific identity was measured on, so it is backend-specific for "
        "exactly the same reason. Candle's released cell is the CUDA q4 I2V one; MLX's is the Metal "
        "q8 base cell its retained anchor is filed under. Making the two agree would either "
        "relabel a measured record or claim a cell that lane never swept (sc-22737)."
    ),
    ("ltx", "MEMORY_REGISTRATION"): (
        "the registration is structurally identical but resolves its provider id through each "
        "backend's distinct MODEL_ID; the companion MODEL_ID exemption records the released "
        "checkpoint split, so this aggregate necessarily differs too."
    ),
    ("ltx", "MEMORY_BEHAVIOR"): (
        "the behavior registration is structurally identical but resolves its provider id through "
        "each backend's distinct MODEL_ID; its differing identity is deliberate rather than a "
        "cross-backend geometry disagreement."
    ),
    ("mage", "ATTENTION_CHUNK_SIZE"): (
        "candle takes the shared `gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET` "
        "(64 Mi score elements); mlx pins 16 Mi, a quarter of it, under the SC-15509 Apple/Metal "
        "calibration paragraph immediately above it. A measured Metal operating point."
    ),
    ("mage", "MODEL_ID"): (
        "different kinds of id that collide on the name: candle's is the generator id "
        "(`config.rs`), mlx's is the *trainer* id in `training.rs`, which by the "
        "`TrainerDescriptor::id` convention is the Base checkpoint's `mage_flow_base`."
    ),
    ("qwen-image", "DEFAULT_STEPS"): (
        "candle 30, mlx 4, and both carry a long doc comment arguing its own value. mlx keeps the "
        "fork's verbatim function-signature default rather than silently diverging from it, and "
        "records that raising it to the fork's documented 30 is an owner decision (sc-4139)."
    ),
    ("qwen-image", "MODEL_ID"): (
        "per-variant registry ids, one per module. Each backend ships a different set of variants, "
        "so the sets of values differ: candle `qwen_image`; mlx also `qwen_image_control` and "
        "`qwen_image_edit`."
    ),
    ("qwen-image", "TRANSFORMER_WINDOW_SIZES"): (
        "each backend's own *published* rung-4 window ladder, not a checkpoint geometry. mlx "
        "advertises `&[TRANSFORMER_WINDOW_SIZE]` = `&[1]` under a doc comment recording SC-16353: "
        "1/2/4/8 plus an unbounded 60-block control were measured at 1024 across Q4/Q8/BF16 and only "
        "window 1 materially lowered the denoise counter, so publishing more would advertise "
        "candidates the Metal kernel does not distinguish. candle publishes the CUDA ladder "
        "`&[1, 2, 4, 8, 15, 30]`, behind `#[cfg(any(feature = \"cuda\", test))]` and with no doc "
        "comment of its own. A caller picks from whichever backend it is running on."
    ),
    ("sdxl", "SIZE_MULTIPLE"): (
        "8 vs 32, both documented on purpose: candle validates the VAE /8 stride only; mlx "
        "enforces `8 * 2^2 = 32` because its U-Net mirrors exact skip concatenations and an odd "
        "intermediate extent would break the joins. Same split as kolors, which shares the "
        "SDXL U-Net."
    ),
    ("wan", "VAE_TILING"): (
        "the z48 halves agree (`VaeTiling::WAN22` on both); the z16 halves differ in "
        "`causal_temporal`, and `candle-gen-wan/src/vae16.rs` says so in the doc comment directly "
        "above the declaration: 'This deliberately differs from MLX's non-causal z16 temporal "
        "geometry while retaining the same 96-channel full-resolution write bound.' candle spells "
        "the z16 tiling inline with `causal_temporal: true`; mlx takes the shared "
        "`VaeTiling::WAN`, whose own doc comment in `gen-core/src/tiling.rs` documents it as "
        "non-causal (`T → T·4`). Same spatial scale, same temporal scale, same 96-channel bound."
    ),
    ("wan", "WAN_Z16_VAE_TILING"): (
        "an associated const re-exported off each backend's own provider type "
        "(`wan14b::ProviderVae` vs `model::A14bProviderVae`), not a value either side spells out. "
        "The paths differ because the types do."
    ),
    ("wan", "WAN_Z48_VAE_TILING"): (
        "as WAN_Z16_VAE_TILING: `Ti2vProviderVae::VAE_TILING` reached through each backend's own "
        "module path."
    ),
    ("z-image", "DECODE_OVERLAP"): (
        "the same per-backend measured VAE tile overlap as `flux.DECODE_OVERLAP`: candle 128 on "
        "CUDA, mlx 64 on Metal, both paired with the same 512 px `DECODE_TILE_EDGE`. mlx's doc "
        "comment records that 64 is the only native overlap it advertises, so that moving the tile "
        "ladder and the overlap at once cannot make a calibration row un-attributable; candle's 128 "
        "carries no comment of its own. A measured operating point, not a value either backend "
        "could read off the checkpoint."
    ),
    ("z-image", "DEFAULT_STEPS"): (
        "different *sets*, because the two backends ship different variants — the same split the "
        "`MODEL_ID` entry below describes. Both declare 4 for the guidance-distilled turbo "
        "checkpoint; mlx additionally ships undistilled Base, whose `model_base.rs` declares 50 "
        "under a doc comment citing the reference `ZImagePipeline` example "
        "(`num_inference_steps=50`). candle has no Base port to declare a default for."
    ),
    ("z-image", "MODEL_ID"): (
        "per-variant registry ids, one per module, and mlx ships two Fun-Controlnet-Union variants "
        "candle has no port of. Every one of these doc comments already says the ids coexist."
    ),
    ("z-image", "TRANSFORMER_WINDOW_SIZES"): (
        "as `qwen-image.TRANSFORMER_WINDOW_SIZES`, and for the measurement recorded on this "
        "family's own mlx declaration: window 1 reaches 2.072 GiB against a 2.247 GiB never-bounds "
        "control, and the sweep found the window itself worth at most ~175 MiB of a ~2.8 GiB "
        "saving, so mlx publishes `&[1]` — 'one candidate is the honest domain either way'. candle "
        "publishes the CUDA ladder `&[1, 2, 4, 8, 15, 30]`."
    ),
}

# The same thing as CROSS_BACKEND_GEOMETRY_EXEMPTIONS, for a constant whose value is an *aggregate*:
# the exemption names the sub-fields that diverge instead of the whole constant.
#
# Whole-constant exemptions are the wrong instrument for the encoder/tokenizer/prompt-execution
# contracts. Each is one `const` holding twenty-odd behavior-bearing fields, so exempting the
# constant to record that `loaded_hidden_layers` differs also exempts `hidden_size`, `head_dim`,
# `vocab_size`, every required token id and every prompt length policy inside it — for exactly the
# families whose two backends are known to have drifted once. A future divergence in any of them
# would be swallowed by an exemption written about something else, which is the same shape of hole
# sc-19419 was.
#
# A path addresses a sub-value the way the Rust reads: `.field` into a struct, `[key]` into a slice
# whose elements name themselves (`purpose`, `role`), `[index]` into one whose elements do not. A
# path present on only one side is that side's extra element. `Some(..)` is transparent — an
# optional aggregate is addressed by its own fields — but `Some(x)` against `None` is one whole
# value, because `None` has no fields to name.
#
# The staleness rule is per path, not per constant: a listed path whose two sides now agree fails,
# and a diverging path nobody listed fails. Both halves are what make this narrower than the
# whole-constant table rather than a re-spelling of it.
CROSS_BACKEND_GEOMETRY_FIELD_EXEMPTIONS: dict[tuple[str, str], dict[str, str]] = {
    ("flux2", "DEV_ENCODER_CONTRACT"): {
        ".loaded_hidden_layers": (
            "per-backend encoder realization of the same Mistral checkpoint (sc-18306): candle "
            "materializes only the 30 layers its deepest selected hidden-state tap needs, mlx "
            "loads the full 40-layer stack through its shared decoder loader. Loaded subset, not "
            "geometry."
        ),
        ".requires_final_norm": (
            "follows `loaded_hidden_layers`: candle stops at its deepest tap and never reads the "
            "final decoder norm; mlx's caption-upsample route does."
        ),
        ".requires_lm_head": (
            "follows `loaded_hidden_layers`: only the MLX lane wires caption upsampling, which is "
            "what constructs the LM head."
        ),
        ".tokenizer.required_tokens[mistral_eos]": (
            "one of the four special tokens only the MLX lane's caption-upsample execution "
            "consumes; candle has no execution that emits or stops on `</s>`. See "
            "`DEV_TOKENIZER_CONTRACT`, which this contract embeds."
        ),
        ".tokenizer.required_tokens[pixtral_image]": (
            "as `mistral_eos`: a pixtral image token only the MLX caption-upsample execution needs."
        ),
        ".tokenizer.required_tokens[pixtral_image_break]": (
            "as `mistral_eos`: a pixtral image token only the MLX caption-upsample execution needs."
        ),
        ".tokenizer.required_tokens[pixtral_image_end]": (
            "as `mistral_eos`: a pixtral image token only the MLX caption-upsample execution needs."
        ),
        ".prompt_executions[flux2_dev_caption_upsample]": (
            "the caption-upsample execution only the MLX lane wires. See `DEV_PROMPT_EXECUTIONS`, "
            "which this contract embeds."
        ),
    },
    ("flux2", "DEV_PROMPT_EXECUTIONS"): {
        "[flux2_dev_caption_upsample]": (
            "mlx declares one extra execution — the caption-upsample path only the MLX lane wires. "
            "The executions the two backends share are compared field by field and agree."
        ),
    },
    ("flux2", "DEV_TOKENIZER_CONTRACT"): {
        ".required_tokens[mistral_eos]": (
            "mlx audits four special tokens candle does not, all of them consumed by the "
            "caption-upsample execution only the MLX lane wires: `</s>` here, the three pixtral "
            "image tokens below. Every token the two backends both declare — including the "
            "mistral `<pad>` candle's padding path emits — is identical, id and literal."
        ),
        ".required_tokens[pixtral_image]": (
            "as `mistral_eos`: pixtral `[IMG]`, needed only by the MLX caption-upsample execution."
        ),
        ".required_tokens[pixtral_image_break]": (
            "as `mistral_eos`: pixtral `[IMG_BREAK]`, needed only by the MLX caption-upsample "
            "execution."
        ),
        ".required_tokens[pixtral_image_end]": (
            "as `mistral_eos`: pixtral `[IMG_END]`, needed only by the MLX caption-upsample "
            "execution."
        ),
    },
    ("flux2", "KLEIN_ENCODER_CONTRACT"): {
        ".loaded_hidden_layers": (
            "same shape as `DEV_ENCODER_CONTRACT.loaded_hidden_layers`: candle loads 27 of Qwen3's "
            "36 layers (its deepest tap), mlx loads all 36 through its shared decoder loader. "
            "Every other field of this contract, tokenizer and executions included, agrees."
        ),
    },
    ("krea", "ENCODER_CONTRACT"): {
        ".packing.supports_file": (
            "per-backend packing capability, not geometry: mlx's packed loader accepts a "
            "single-file artifact, candle requires a directory artifact. Every other field of the "
            "packing contract — group size, embedding and LM-head packing — agrees."
        ),
        ".dense_storage_dtype_probe": (
            "follows `packing.supports_file`: candle probes "
            "`language_model.layers.0.input_layernorm.weight` to classify dense storage, mlx's "
            "packed loader needs no probe."
        ),
    },
    # The krea prompt-execution `length` exemptions (`[krea_t2i]` / `[krea_edit]`, on both this
    # constant and `PROMPT_EXECUTIONS`) are gone: mlx used to leave both executions unbounded while
    # candle rejected above 1024 / 8192, and the sc-17137 sync settled it by giving mlx the same
    # fail-loud admission the repo chose in sc-9047 — `MAX_TEXT_TOKENS` / `MAX_EDIT_TOKENS` in
    # `mlx-gen-krea/src/text_encoder/tokenizer.rs`, enforced by `check_len` and named directly by
    # `PROMPT_EXECUTIONS`. The two lanes now agree, so an exemption here would itself red the gate.
    ("qwen-image", "ENCODER_CONTRACT"): {
        ".tokenizer.artifact_candidates[1]": (
            "the encoder contract embeds the tokenizer contract, so the loader search path "
            "recorded on `TOKENIZER_CONTRACT` surfaces here too. Every geometry field agrees."
        ),
    },
    ("qwen-image", "TOKENIZER_CONTRACT"): {
        ".artifact_candidates[1]": (
            "candle resolves the tokenizer from two artifact candidates "
            "(`tokenizer/tokenizer.json`, then `processor/tokenizer.json`); mlx ships only the "
            "first, so only candle declares a second. A loader search path, not geometry — the "
            "first candidate and every required token agree."
        ),
    },
    ("z-image", "ENCODER_CONTRACT"): {
        ".prompt_executions[z_image_prompt].padding": (
            "batching posture, not geometry — see `PROMPT_EXECUTIONS`, which this contract embeds. "
            "`qk_norm_eps` used to be listed here as an open question and is now settled at the "
            "checkpoint's 1e-6 on both lanes (sc-17137 sync review); this gate reds if it diverges "
            "again."
        ),
    },
    ("z-image", "PROMPT_EXECUTIONS"): {
        "[z_image_prompt].padding": (
            "the `z_image_prompt` execution pads differently by lane: candle applies no padding, "
            "mlx right-pads to the 512-token max with the qwen pad token. Verified functionally "
            "equivalent against the diffusers reference (sc-17137 sync review), not an open "
            "question: `pipeline_z_image.py:229-249` pads to 512 purely so a batch can be one "
            "tensor, then strips each sample back to its valid length with the attention mask "
            "before the DiT ever sees it. mlx mirrors that mechanically — pad, then `slice_valid` "
            "back to `num_valid` (`mlx-gen-z-image/src/pipeline.rs:600-615`); candle reaches the "
            "same place by skipping the pad round-trip entirely. Both feed the DiT identical "
            "valid-token conditioning, and both truncate identically above 512 through the shared "
            "`gen-core/src/tokenizer.rs:196-202`. The exemption stays because the declared "
            "constants still differ — a batching posture, not geometry — but the question of "
            "whether that difference changes conditioning is settled: it does not. The "
            "`z_image_empty_negative` execution agrees field for field."
        ),
    },
}

# Agreement alone is not correctness — the defect this gate exists for was two crates that would have
# agreed perfectly had anyone "fixed" the red by copying the wrong value across. So the reference
# values are pinned too, on BOTH sides, and they come from the released checkpoint read through the
# diffusers reference (0.40.0.dev0 @ 7564fb01), never from either backend:
#
#   VAE_RATIO   = prod(vae/config.json spatial_downsample_factors [2,2,2,2,1,1])       = 16
#   VAE_RATIO_T = prod(vae/config.json temporal_downsample_factors [1,2,2,1,1,1])      = 4
#   SIZE_MULTIPLE = MiniMaxH3ModularPipeline.canvas_multiple, i.e.
#                   vae_spatial_compression_ratio * transformer patch_size[2] = 16 * 2 = 32
#                   (modular_pipeline.py:236; transformer/ and transformer_ref/ both ship [1, 2, 2])
#   FRAMES_PER_CHUNK  = vae.config.clip_length          = 17  } the `17n + 5` lattice
#   LATENTS_PER_CHUNK = vae.tokens_chunk_size           = 5   } (modular_pipeline.py:96-129)
#   LATENT_CHANNELS   = vae/config.json latent_channels = 24
#   TOKEN_DROP        = vae/config.json token_drop      = 3
#   CLIP_LENGTH       = vae/config.json clip_length     = 17
#   MINIMAX_H3_FPS    = modular_pipeline.py:31          = 24
#   LEGAL_FRAME_COUNTS = the 14 `17n + 5` counts inside the 5.0-15.0 s envelope at 24 fps.
CROSS_BACKEND_GEOMETRY_REFERENCE: dict[str, dict[str, tuple[float, ...]]] = {
    "minimax-h3": {
        "SIZE_MULTIPLE": (32.0,),
        "VAE_RATIO": (16.0,),
        "VAE_RATIO_T": (4.0,),
        "LATENT_CHANNELS": (24.0,),
        "TOKEN_DROP": (3.0,),
        "CLIP_LENGTH": (17.0,),
        "FRAMES_PER_CHUNK": (17.0,),
        "LATENTS_PER_CHUNK": (5.0,),
        "MINIMAX_H3_FPS": (24.0,),
        "ENCODER_SPATIAL_DOWNSAMPLE_FACTORS": (2.0, 2.0, 2.0, 2.0, 1.0, 1.0),
        "ENCODER_TEMPORAL_DOWNSAMPLE_FACTORS": (1.0, 2.0, 2.0, 1.0, 1.0, 1.0),
        "LEGAL_FRAME_COUNTS": tuple(
            float(17 * n + 5) for n in range(60) if 120 <= 17 * n + 5 <= 360
        ),
    },
}

# The second half of sc-19496. Both MiniMax-H3 crates hand-maintained the *fixture* geometry too, in
# `tests/common/mod.rs`, inside functions — `dit_fixture_config`, `fixture_config`,
# `encode_fixture_config`, `audio_fixture_config` and the `DIT_LAYOUT` literal — under doc comments
# reading "the same set of numbers the MLX lane's `dit_fixture_config` uses" and "Identical to the
# MLX lane's". Nothing checked either sentence, and this gate could not see inside a function body.
#
# That mattered concretely rather than theoretically: the two crates commit *byte-identical* fixture
# files (`video_vae_decode.safetensors`, `video_vae_encode.safetensors`, `audio_vae_decode.safetensors`,
# `dit_block.safetensors`, `av_denoise.safetensors` all hash the same on both sides). Both lanes load
# the same bytes through their own hand-typed geometry, so a drift in either config leaves both lanes
# internally consistent, both parity suites green, and the two backends comparing tensors dumped at
# one shape against a model built at another.
#
# The numbers are now `pub const SHARED_FIXTURE_*` declarations that those functions *construct
# from*, so the gate compares the values the tests actually use rather than text that resembles them.
# Both the name set and the values must match, so a number lifted on one side only is caught as the
# same drift one step earlier.
CROSS_BACKEND_FIXTURE_PREFIX = "SHARED_FIXTURE_"

# MiniMax-H3 was not the only family in that shape — it was one of five. Every family below commits at
# least one fixture file byte-identically on both sides AND hand-types the geometry it is loaded
# through, so each carried the same silent-drift hole. The `SHARED_FIXTURE_*` treatment now applies to
# all of them, and a family listed here whose crates declare no `SHARED_FIXTURE_*` constants at all is
# a failure: that is what makes the requirement a gate rather than a convention someone remembered to
# follow once.
#
# Membership is not taken on trust. `_shares_a_fixture_file` re-derives the premise — a same-named
# file under both crates' `tests/fixtures/` with the same bytes — every run, so a family whose
# fixtures stopped being shared cannot keep drawing a requirement that no longer describes it.
#
# What that check does NOT claim is that every same-named fixture file matches: `sana` commits a
# `dcae_encode_golden.safetensors` on each side that differ, two independent diffusers dumps of the
# encode reference. Each lane's `#[ignore]`d encode gate is sound against its own; they are simply not
# one shared reference, and that is recorded here rather than gated, because nothing here can tell a
# deliberate second dump from a drifted copy.
#
# `bernini` earns its place with the smallest surface of the five, and deliberately so: most of its
# fixture geometry is carried in each golden's own safetensors `__metadata__` and read back by both
# lanes, which needs no gate at all — one copy of a number cannot disagree with itself. Only what the
# goldens do not record is declared as `SHARED_FIXTURE_*`.
CROSS_BACKEND_FIXTURE_FAMILIES: dict[str, str] = {
    "anima": (
        "six committed golden JSONs, byte-identical on both sides. The synthetic-DiT golden records "
        "only `shape`/`count`/moments, so the tiny `DitConfig`, the FNV-1a-of-key seeds, the LCG "
        "recurrence and the synthetic inputs must agree or the two lanes summarise different models."
    ),
    "bernini": (
        "ten byte-identical goldens. Most geometry rides each golden's `__metadata__` and is read by "
        "both lanes; the assembly backbone shape, the ViT-guidance weights and the template task "
        "matrix are the remainder both lanes hand-type."
    ),
    "krea": (
        "six byte-identical fixtures, including `variant5_native_keys.txt`. Both lanes hand-type the "
        "tiny `Krea2Config` and `KreaTeConfig` the DiT and TE goldens are loaded through."
    ),
    "ltx": (
        "two byte-identical fixtures — `ltx25_distilled_dit_tensors.json` (the real LTX-2.5 "
        "distilled DiT header) and `ltx_duration_head_golden.safetensors`. Both lanes hand-type the "
        "DiT depth and the two video-FFN bias widths the 2.3↔2.5 delta is reconstructed from, and "
        "the duration-head golden's three modality case names and relative tolerance."
    ),
    "minimax-h3": (
        "five byte-identical safetensors goldens loaded through hand-typed video-VAE, audio-VAE and "
        "DiT geometry — the family sc-19496 was written for."
    ),
    "sana": (
        "`sana_transformer_golden` and `sana_sprint_trunk_golden`, byte-identical on both sides, both "
        "loaded through a hand-typed tiny `SanaTransformerConfig`."
    ),
}

# Relative max-abs-diff, never a norm, a cosine or a checksum — those went blind to real defects in
# this family seven separate times. Effectively exact for source literals; its only job is to let
# `1e-5` and `0.00001`, or `0.858_090_34` and `0.85809034`, compare equal.
CROSS_BACKEND_GEOMETRY_TOLERANCE = 1e-12

# A `const NAME: TYPE = VALUE;` declaration carrying a visibility modifier, at any indentation.
#
# Values run to the first `;` and may span lines (the 24-element `LATENTS_MEAN`), hence DOTALL and
# `[^;]*`. The type may itself contain a `;` — `[f64; 5]` — so only the value is `;`-bounded; the type
# is merely `=`-bounded, which no Rust type contains.
#
# The `^pub const` this replaced was column-0-only, which across the 33 swept families left 68
# indented `pub const` and 257 `pub(crate) const` declarations unread — 24 of them declared under the
# same name on *both* backends, and every one of those a cross-backend geometry or sampling default:
# `bernini`'s nine `Defaults::{STEPS, NUM_FRAMES, OMEGA_*, ETA, MOMENTUM, NORM_THRESHOLD}`,
# `sensenova`'s nine, `qwen-image`'s two, and one each in `ltx`, `krea`, `wan` and minimax-h3 itself
# (`DitProjections::TENSOR_COUNT`). An associated const on a type is reached the same way, so
# `WanVae16::VAE_TILING` is now compared as a value rather than only through the two `lib.rs`
# re-exports that name it.
#
# A *private* `const` is deliberately still not read, and that restriction is a claim about what this
# gate compares rather than a convenience: the premise is "one published geometry, declared twice",
# and a bare `const` is not published — two crates that happen to give an unexported module constant
# the same name have a name collision, not a shared declaration. `boogu`'s `IMG` (a token id in
# candle's text encoder, an unrelated 999 in mlx's) is exactly that shape. Extending to private
# consts adds 387 more shared names and 45 divergences that need per-item owner reads, several of
# which look like real defects; that sweep is sc-19696, not this gate's silence.
RUST_PUB_CONST = re.compile(
    r"^[ \t]*pub(?:\s*\([^()]*\))?[ \t]+const (\w+)\s*:\s*([^=]+?)\s*=\s*([^;]*);[ \t]*$",
    re.M | re.S,
)

# `VAE_RATIO as u32 * 2` and `AUDIO_TOKEN_RATE_HZ as i32` differ from their siblings only in the cast,
# because the two backends spell the same number in different Rust types (`usize`/`i32`, `f64`/`f32`).
# The cast carries no value, so it is removed before comparison. Applied by
# `_normalize_const_value` to code only — inside a string `as` is just a word, and stripping it made
# `boogu`'s `SYSTEM_PROMPT_T2I` ("…the instructions are as follows.") and `sensenova`'s
# `SYSTEM_MESSAGE_FOR_GEN` ("…as input") compare equal to prose that genuinely differs.
RUST_CONST_CAST = re.compile(r"\bas\s+\w+")

# A digit separator inside a numeric literal. Two things keep it there, and it needs both.
#
# The lookbehind stops it starting mid-identifier: the `(?<=\d)_(?=\d)` form this replaced rewrote
# `SD3_5_LARGE_ID` to `SD35_LARGE_ID`, which then resolved against nothing and dropped `sd3`'s
# `MODEL_ID` to a false divergence.
#
# `_normalize_const_value` running it over code only is the other half, and the lookbehind cannot do
# that half's job: a digit run inside a string starts after `/` or `-` just as legitimately as after
# a space, so `"model/2_3"` still collapsed to `"model/23"` and `"SceneWorks/wan-2_2"` to
# `"SceneWorks/wan-22"` — HF-style repo ids are precisely where that bites, and comparing equal to a
# genuinely different id is the same false green the `"ltx_2_3"` case was.
RUST_NUMERIC_LITERAL = re.compile(r"(?<![A-Za-z0-9_])\d[\d_]*(?:\.\d[\d_]*)?")

# `candle-gen-*` reaches the shared backend surface through `candle_gen::`, `mlx-gen-*` through
# `mlx_gen::`. `candle_gen::gen_core::MemoryRegistration { .. }` and
# `mlx_gen::gen_core::MemoryRegistration { .. }` are the same item named through each backend's own
# shim, so the shim segment carries no value and is folded to a common token before comparison.
RUST_BACKEND_SHIM = re.compile(r"\b(?:candle_gen|mlx_gen)::")

# A module path in front of a SCREAMING_CASE name: `config::SPATIAL_SCALE`,
# `crate::denoise::LEGAL_FRAME_COUNTS`. The path is how the constant is *reached* from where it is
# used, not what it is, and the two backends organize their modules differently. Stripping it lets
# the same-crate resolver fold a path-qualified reference the way it already folds a bare one. Only
# lowercase segments are stripped, so an associated const on a type (`ProviderVae::VAE_TILING`) keeps
# its qualifier and is compared as text.
RUST_PATH_QUALIFIER = re.compile(r"\b(?:[a-z_][A-Za-z0-9_]*::)+(?=[A-Z_][A-Z0-9_]*\b)")

# `NAME[i]` and `NAME.len()` over an array constant declared in the same crate. `MIN_DURATION_SECONDS
# = LEGAL_FRAME_COUNTS[0] as f64 / MINIMAX_H3_FPS` is a real declaration in both MiniMax-H3 crates; a
# gate that cannot fold it compares two backends' durations as text and passes on any pair of
# spellings.
RUST_CONST_INDEX = re.compile(r"\b([A-Z_][A-Z0-9_]*)\s*\[\s*([^\[\]]*?)\s*\]")
RUST_CONST_LEN = re.compile(r"\b([A-Z_][A-Z0-9_]*)\s*\.\s*len\s*\(\s*\)")

# An identifier to resolve, and never the `e` of a float literal: without the lookbehind, `1e-5`
# reads as the identifier `e`, fails to resolve, and drops the constant to text-only comparison —
# which silently costs it the reference pin and would false-red `1e-5` against `0.00001`.
RUST_IDENTIFIER = re.compile(r"(?<![0-9.])[A-Za-z_][A-Za-z0-9_]*")

_RAW_STRING_START = re.compile(r'(?:b?r)(#*)"')

_CFG_ATTR_START = re.compile(r"#\s*(!?)\s*\[\s*cfg\s*\(")


def fail(message: str) -> None:
    raise AssertionError(message)


def _rust_chunks(source: str) -> list[tuple[str, str]]:
    """Split Rust source into ``("code" | "comment" | "literal", text)`` runs, in order.

    One scanner, two callers. ``strip_rust_comments`` blanks the comment runs (and optionally the
    literal ones); ``_normalize_const_value`` leaves the literal runs untouched while rewriting the
    code around them. Sharing the walk is the point — a second, looser notion of "inside a string"
    is how a normalizer ends up rewriting the contents of one (sc-19496 review).

    Raw strings (``r"..."``, ``r#"..."#``, ``br#"..."#``) are literals, since containing otherwise-
    significant characters is their whole purpose. Block comments nest, as they do in Rust. ``'a``
    lifetimes are distinguished from ``'a'`` char literals.
    """
    chunks: list[tuple[str, str]] = []

    def emit(kind: str, text: str) -> None:
        if not text:
            return
        if chunks and chunks[-1][0] == kind:
            chunks[-1] = (kind, chunks[-1][1] + text)
            return
        chunks.append((kind, text))

    index = 0
    length = len(source)
    while index < length:
        char = source[index]
        if char == "/" and source.startswith("//", index):
            end = source.find("\n", index)
            end = length if end < 0 else end
            emit("comment", source[index:end])
            index = end
            continue
        if char == "/" and source.startswith("/*", index):
            depth = 0
            end = index
            while end < length:
                if source.startswith("/*", end):
                    depth += 1
                    end += 2
                    continue
                if source.startswith("*/", end):
                    depth -= 1
                    end += 2
                    if depth == 0:
                        break
                    continue
                end += 1
            emit("comment", source[index:end])
            index = end
            continue
        raw = _RAW_STRING_START.match(source, index)
        if raw and not (index and (source[index - 1].isalnum() or source[index - 1] == "_")):
            terminator = '"' + raw.group(1)
            end = source.find(terminator, raw.end())
            end = length if end < 0 else end + len(terminator)
            emit("literal", source[index:end])
            index = end
            continue
        if char == '"':
            end = index + 1
            while end < length:
                if source[end] == "\\":
                    end += 2
                    continue
                if source[end] == '"':
                    end += 1
                    break
                end += 1
            emit("literal", source[index:end])
            index = end
            continue
        if char == "'":
            # `'\n'` / `'\u{1f600}'`: an escape runs to the closing quote.
            if source.startswith("'\\", index):
                end = source.find("'", index + 2)
                end = length if end < 0 else end + 1
                emit("literal", source[index:end])
                index = end
                continue
            # `'a'` is a char literal; `'a` (no closing quote) is a lifetime, so emit just the tick and
            # let the identifier be scanned normally.
            if index + 2 < length and source[index + 2] == "'":
                emit("literal", source[index : index + 3])
                index += 3
                continue
            emit("code", char)
            index += 1
            continue
        emit("code", char)
        index += 1
    return chunks


def strip_rust_comments(source: str, *, strip_literals: bool = False) -> str:
    """Blank out Rust comments so a source-text policy check cannot be satisfied by a comment.

    Comments become runs of spaces (newlines preserved) rather than being deleted, so line and column
    positions still line up with the original if a caller ever reports them.

    String and character literals are honoured, because ``"// not a comment"`` and ``"/* nor this */"``
    are real code. When ``strip_literals`` is true, their spans are blanked too, so a policy looking
    for call syntax cannot be satisfied by a diagnostic or test fixture string.
    """

    def blank(text: str) -> str:
        return "".join("\n" if char == "\n" else " " for char in text)

    return "".join(
        blank(text) if kind == "comment" or (kind == "literal" and strip_literals) else text
        for kind, text in _rust_chunks(source)
    )


def cargo_metadata(offline: bool) -> dict:
    command = ["cargo", "metadata", "--locked", "--format-version", "1"]
    if offline:
        command.append("--offline")
    # cargo emits UTF-8 on every platform, so decode explicitly. text=True would decode with
    # the locale encoding instead, which fails on Windows (cp1252) as soon as any package in
    # the resolved graph carries non-ASCII metadata -- today a dependency author name.
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        fail(f"cargo metadata failed with exit code {result.returncode}")
    return json.loads(result.stdout.decode("utf-8"))


def _within_workspace(path: Path) -> bool:
    """True when a discovered path belongs to this checkout's own workspace tree.

    The check is on the path RELATIVE to ROOT, so running the gate from inside a nested worktree
    (whose own absolute path contains e.g. ``.claude/worktrees/...``) still counts that worktree's
    own root Cargo.lock/manifest -- only subtrees *below* ROOT named in IGNORED_TREE_PARTS drop out.
    """
    return IGNORED_TREE_PARTS.isdisjoint(path.relative_to(ROOT).parts)


def check_filesystem() -> None:
    lockfiles = sorted(
        path.relative_to(ROOT)
        for path in ROOT.rglob("Cargo.lock")
        if _within_workspace(path)
    )
    if lockfiles != [Path("Cargo.lock")]:
        fail(f"expected only the root Cargo.lock, found: {lockfiles}")

    workspace_manifests = []
    for manifest in ROOT.rglob("Cargo.toml"):
        if not _within_workspace(manifest):
            continue
        if any(
            line.strip() == "[workspace]"
            for line in manifest.read_text(encoding="utf-8").splitlines()
        ):
            workspace_manifests.append(manifest.relative_to(ROOT))
    if workspace_manifests != [Path("Cargo.toml")]:
        fail(f"expected one active root workspace manifest, found: {workspace_manifests}")

    for required in (Path(".cargo/config.toml"), Path("rust-toolchain.toml")):
        if not (ROOT / required).is_file():
            fail(f"missing root-owned configuration: {required}")

    root_manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    dependencies = root_manifest["workspace"]["dependencies"]
    for dependency_name, (package_name, revision) in PINNED_WORKSPACE_DEPENDENCIES.items():
        dependency = dependencies.get(dependency_name)
        if not isinstance(dependency, dict):
            fail(f"missing structured root pin for {dependency_name}")
        if dependency.get("rev") != revision:
            fail(f"{dependency_name} is not declared at {revision}: {dependency}")
        if dependency.get("package", dependency_name) != package_name:
            fail(f"{dependency_name} no longer aliases package {package_name}: {dependency}")


def check_graph(metadata: dict) -> None:
    packages = metadata["packages"]
    packages_by_id = {package["id"]: package for package in packages}
    member_ids = metadata["workspace_members"]
    members = [packages_by_id[member_id] for member_id in member_ids]

    if len(member_ids) != EXPECTED_MEMBER_COUNT:
        fail(f"expected {EXPECTED_MEMBER_COUNT} workspace members, found {len(member_ids)}")
    if len(set(member_ids)) != len(member_ids):
        fail("workspace member IDs are not unique")

    for package in members:
        manifest = Path(package["manifest_path"]).resolve()
        if package["source"] is not None:
            fail(f"workspace member {package['name']} unexpectedly has source {package['source']}")
        if ROOT / "crates" not in manifest.parents:
            fail(f"workspace member is outside crates/: {manifest}")

    for name in sorted(INTERNAL_PACKAGES):
        matches = [package for package in packages if package["name"] == name]
        if len(matches) != 1:
            fail(f"expected one {name} package resolution, found {len(matches)}")
        if matches[0]["source"] is not None:
            fail(f"internal package {name} is not a path source: {matches[0]['source']}")

    resolved_names = {package["name"] for package in packages}
    forbidden = sorted(FORBIDDEN_GRAPH_PACKAGES & resolved_names)
    if forbidden:
        fail(f"explicit composition forbids these graph packages: {forbidden}")

    for package in members:
        for dependency in package["dependencies"]:
            if dependency["name"] not in INTERNAL_PACKAGES:
                continue
            if dependency["source"] is not None or dependency.get("path") is None:
                fail(
                    f"{package['name']} -> {dependency['name']} is not a workspace path edge: "
                    f"source={dependency['source']!r}, path={dependency.get('path')!r}"
                )

    for name, revision in DEFAULT_GRAPH_PINNED_PACKAGES.items():
        matches = [package for package in packages if package["name"] == name]
        if len(matches) != 1:
            fail(f"expected one {name} resolution, found {len(matches)}")
        source = matches[0]["source"] or ""
        if not source.endswith(f"#{revision}"):
            fail(f"{name} does not resolve at {revision}: {source}")

    tokenizer_minors = {
        ".".join(package["version"].split(".")[:2])
        for package in packages
        if package["name"] == "tokenizers"
    }
    if tokenizer_minors != {"0.21", "0.22"}:
        fail(f"expected intentional tokenizers 0.21/0.22 split, found {tokenizer_minors}")


def check_network_clients(metadata: dict) -> None:
    """No network/HTTP client may resolve in the workspace graph (epic 13657 self-fetch ban)."""
    packages = metadata["packages"]
    packages_by_id = {package["id"]: package for package in packages}
    resolved_names = {package["name"] for package in packages}

    present = sorted(FORBIDDEN_NETWORK_CLIENT_PACKAGES & resolved_names)
    if not present:
        return

    # Attribute each present client to the workspace member(s) that declare it directly, so the
    # error names the reintroduction site (the common case: someone adds it to a member manifest).
    direct = sorted(
        {
            (packages_by_id[member_id]["name"], dependency["name"])
            for member_id in metadata["workspace_members"]
            for dependency in packages_by_id[member_id]["dependencies"]
            if dependency["name"] in FORBIDDEN_NETWORK_CLIENT_PACKAGES
        }
    )
    detail = f"; direct workspace-member deps: {direct}" if direct else "; transitive only"
    fail(
        "inference never self-fetches weights: no network/HTTP client may resolve in the graph, "
        f"found {present}{detail}"
    )


def _match_brace(text: str, open_index: int) -> int:
    """Index just past the ``}`` matching the ``{`` at ``open_index``.

    Rust string/char literals (including raw strings) and comments are skipped so a brace inside a
    format string or a comment cannot unbalance the count.
    """
    depth = 0
    i = open_index
    n = len(text)
    while i < n:
        char = text[i]
        if char == "{":
            depth += 1
            i += 1
        elif char == "}":
            depth -= 1
            i += 1
            if depth == 0:
                return i
        elif char == "/" and i + 1 < n and text[i + 1] == "/":
            newline = text.find("\n", i)
            i = n if newline == -1 else newline
        elif char == "/" and i + 1 < n and text[i + 1] == "*":
            close = text.find("*/", i + 2)
            i = n if close == -1 else close + 2
        elif char == "r" and i + 1 < n and text[i + 1] in '#"':
            hashes = 0
            cursor = i + 1
            while cursor < n and text[cursor] == "#":
                hashes += 1
                cursor += 1
            if cursor < n and text[cursor] == '"':
                terminator = '"' + "#" * hashes
                close = text.find(terminator, cursor + 1)
                i = n if close == -1 else close + len(terminator)
            else:
                i += 1
        elif char == '"':
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                elif text[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
        elif char == "'":
            # Char literal or a lifetime. Skip an escaped or single-char literal; otherwise advance
            # one (a lifetime such as `'a` has no closing quote).
            if i + 1 < n and text[i + 1] == "\\":
                close = text.find("'", i + 2)
                i = n if close == -1 else close + 1
            elif i + 2 < n and text[i + 2] == "'":
                i += 3
            else:
                i += 1
        else:
            i += 1
    return n


def _split_cfg_args(expression: str) -> list[str]:
    """Split one cfg predicate's arguments at top-level commas."""
    args: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(expression):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(expression[start:index].strip())
            start = index + 1
    args.append(expression[start:].strip())
    return [arg for arg in args if arg]


def _cfg_truth_without_test(expression: str) -> tuple[bool, bool]:
    """Return ``(can_be_false, can_be_true)`` with the Rust ``test`` cfg fixed false.

    Other predicates are treated as independently unknown. That is deliberately conservative: an
    item is removed from a production scan only when its cfg expression cannot possibly be true in
    a non-test build.
    """
    expression = expression.strip()
    if expression == "test":
        return True, False
    call = re.fullmatch(r"(all|any|not)\s*\((.*)\)", expression, re.DOTALL)
    if call is None:
        return True, True
    kind, body = call.groups()
    values = [_cfg_truth_without_test(arg) for arg in _split_cfg_args(body)]
    if kind == "all":
        return any(can_false for can_false, _ in values), all(
            can_true for _, can_true in values
        )
    if kind == "any":
        return all(can_false for can_false, _ in values), any(
            can_true for _, can_true in values
        )
    if len(values) != 1:
        return True, True
    can_false, can_true = values[0]
    return can_true, can_false


def _cfg_attributes_requiring_test(text: str) -> list[tuple[int, int, bool]]:
    """Attributes whose cfg expression cannot be true when ``test`` is false.

    The third tuple member identifies inner ``#![...]`` attributes, which gate the enclosing
    module/file rather than a following construct.
    """
    attributes: list[tuple[int, int, bool]] = []
    for start in _CFG_ATTR_START.finditer(text):
        open_paren = start.end() - 1
        depth = 1
        cursor = open_paren + 1
        while cursor < len(text) and depth:
            if text[cursor] == "(":
                depth += 1
            elif text[cursor] == ")":
                depth -= 1
            cursor += 1
        if depth:
            continue
        close_paren = cursor - 1
        after = cursor
        while after < len(text) and text[after].isspace():
            after += 1
        if after >= len(text) or text[after] != "]":
            continue
        _, can_be_true = _cfg_truth_without_test(text[open_paren + 1 : close_paren])
        if not can_be_true:
            attributes.append((start.start(), after + 1, start.group(1) == "!"))
    return attributes


def _cfg_item_end(syntax: str, start: int) -> int | None:
    """Find the end of the Rust item following a cfg attribute.

    ``syntax`` has comments and literals blanked, so only Rust delimiters remain. Braces inside a
    function header (notably const-generic arguments such as ``Foo<{ 1 }>``) are skipped; only a
    top-level brace starts the item's body.
    """
    prefix = syntax[start:]
    prefix = re.sub(r"^\s*(?:#\s*\[[^\]]*\]\s*)*", "", prefix)
    prefix = re.sub(r"^pub(?:\s*\([^)]*\))?\s+", "", prefix)
    # These constructs own every brace in their initializer/type/path and end at a semicolon. A
    # function body, including a qualified `const fn`, ends at its top-level brace instead. String
    # literal blanking leaves an extern ABI as whitespace, so the same pattern covers `extern fn`
    # and `extern "ABI" fn`.
    const_function = (
        re.match(r"const\s+(?:async\s+)?(?:unsafe\s+)?(?:extern\s+)?fn\b", prefix)
        is not None
    )
    semicolon_item = (
        re.match(r"(?:const|static|type|use|let)\b", prefix) is not None
        and not const_function
    )

    paren_depth = 0
    bracket_depth = 0
    angle_depth = 0
    cursor = start
    while cursor < len(syntax):
        char = syntax[cursor]
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "<" and paren_depth == 0 and bracket_depth == 0:
            angle_depth += 1
        elif (
            char == ">"
            and paren_depth == 0
            and bracket_depth == 0
            and not (cursor > 0 and syntax[cursor - 1] == "-")
        ):
            angle_depth = max(0, angle_depth - 1)
        elif char == "{":
            end = _match_brace(syntax, cursor)
            if (
                not semicolon_item
                and paren_depth == 0
                and bracket_depth == 0
                and angle_depth == 0
            ):
                return end
            cursor = end
            continue
        elif (
            char == ";"
            and paren_depth == 0
            and bracket_depth == 0
            # A bare comparison such as `if 1 < 2` is indistinguishable from a generic opener to
            # this lightweight scanner. Recognized semicolon items still end here; their braced
            # initializers were skipped above, and nested delimiters remain protected.
            and (semicolon_item or angle_depth == 0)
        ):
            return cursor + 1
        cursor += 1
    return None


def _cfg_test_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of items that cannot compile outside a Rust test build.

    This covers functions and other items as well as modules, including predicates such as
    ``cfg(all(test, feature = "fixture"))``. An ``any(test, feature = "shipping")`` item is retained
    because it can compile in production when the other predicate is true.
    """
    spans: list[tuple[int, int]] = []
    syntax = strip_rust_comments(text, strip_literals=True)
    for attribute_start, attribute_end, inner in _cfg_attributes_requiring_test(syntax):
        if inner:
            # File-scope inner cfg gates the rest of the file. An inline-module inner cfg could be
            # narrowed to that module's closing brace, but blanking farther is safely fail-closed:
            # trigger discovery deliberately uses a separate cfg-unblanked syntax stream.
            spans.append((attribute_start, len(text)))
            continue
        # Additional attributes and visibility modifiers are intentionally included in the scan
        # from the cfg attribute onward.
        end = _cfg_item_end(syntax, attribute_end)
        if end is None:
            continue
        spans.append((attribute_start, end))
    return spans


def _blank_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Blank selected source spans while preserving newlines and character offsets."""
    if not spans:
        return text
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    out: list[str] = []
    cursor = 0
    for start, end in merged:
        out.append(text[cursor:start])
        out.append("".join("\n" if char == "\n" else " " for char in text[start:end]))
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _configure_decode_hooks_are_unconditional_rejections(text: str) -> bool:
    """Whether every concrete ``configure_decode`` hook is a single typed ``Err`` expression.

    ``MemoryRequestScope`` requires the method even from a provider whose contract declares rung 2
    Missing. Such a hook is not evidence that native tile geometry can reach the PiD seam. Keep the
    exemption deliberately narrow: any statement, delegation, ``?``, or success arm before the
    rejection remains an adoption signal and must use ``DecodeRoutes``.
    """
    starts = list(re.finditer(r"\bfn\s+configure_decode\b", text))
    if not starts:
        return False
    for start in starts:
        open_index = text.find("{", start.end())
        if open_index < 0:
            return False
        end = _match_brace(text, open_index)
        if not _is_single_err_expression(text[open_index + 1 : end - 1]):
            return False
    return True


# Constructs that make an `Err(...)` argument capable of *not* rejecting. `Err(match p() { Ok(())
# => return Ok(()), Err(e) => e })` and `Err(if ok { return Ok(()); } else { nope() })` both compile
# and both read as unconditional rejections to a shape-only reader; a macro can hide either
# (`Err(or_accept!(plan(e, o)))`) and `rustfmt` does not expand macro bodies, so the brace-wrapping
# that would otherwise make multi-line forms conspicuous cannot be relied on.
_CONDITIONAL_IN_ERR_ARGUMENT = re.compile(r"\breturn\b|\bmatch\b|\bif\b|\?|\w\s*!\s*[\(\[{]")


def _is_single_err_expression(body: str) -> bool:
    """Whether ``body`` is exactly one *unconditional* ``Err(...)`` rejection.

    Shared by the two decode-hook shapes a provider can write, so "unconditional rejection" has one
    definition rather than one per seam.

    Shape alone is not enough, and the SC-15525 probe proved it: `Err(` + balanced `)` + end-of-body
    is satisfied by an argument that early-returns `Ok`. So the argument is additionally required to
    contain no control flow that could escape it — no ``return``, ``?``, ``match`` or ``if``, and no
    macro invocation, since a macro body is opaque here and to ``rustfmt`` alike.

    The cost of that strictness is a provider that wants a computed rejection *reason* written with a
    ``match``; it can hoist the reason into a named function, which is what the shipping provider
    already does (``Err(refuse_decode(id, Some(edge), Some(overlap)))``).
    """
    body = body.strip()
    prefix = re.match(r"Err\s*\(", body)
    if prefix is None:
        return False
    outer_open = body.find("(", prefix.start())
    if _match_paren(body, outer_open) != len(body):
        return False
    argument = body[outer_open + 1 : len(body) - 1]
    return _CONDITIONAL_IN_ERR_ARGUMENT.search(argument) is None


# The shared MLX request-scope constructor. Its LAST positional argument is the provider's decode
# validator — the closure the runtime drives to apply (or refuse) a bounded-decode selection.
#
# SC-15525 review: this is the shape that actually matters, and the first revision of the rung-2
# exemption missed it entirely. **No MLX provider writes `fn configure_decode`** — that method lives
# once, on `mlx_gen::request_scope::MlxRequestScopeCore`, and every family reaches it by handing this
# constructor a closure. So an exemption that only inspected the trait-method form was vacuous for the
# entire MLX provider family: flip the closure's `Err` to `Ok` and a native geometry reaches the PiD
# seam with the gate silent.
MLX_REQUEST_SCOPE_CONSTRUCTOR = "MlxRequestScopeConfig::new"

# The config TYPE itself. Every one of its fields is `pub`, so the constructor is a convention rather
# than a chokepoint — see `_decode_validators_are_unconditional_rejections` for the two shapes that
# exploited that in the first revision.
MLX_REQUEST_SCOPE_CONFIG_TYPE = "MlxRequestScopeConfig"

# ANY field access on the validator, not just `=` assignment. A provider that declares rung 2
# `Missing` has no legitimate reason to touch `decode_validator` at all — it hands its rejection to
# the constructor positionally — so the field access itself is the signal.
#
# This was `\.\s*decode_validator\s*=` in the first hardening pass, which reads as "catch the
# post-construction overwrite" but only catches one spelling of it:
# `mem::replace(&mut cfg.decode_validator, Box::new(|_u, _e, _o| Ok(())))` installs an accepting
# validator with no `=` anywhere near the field, and so does any `&mut` handed to a helper. Matching
# the field rather than one syntax for writing it closes the class instead of one member of it.
_DECODE_VALIDATOR_FIELD_ACCESS = re.compile(r"\.\s*decode_validator\b")


def _opens_a_closure(text: str, arg_start: int, bar_index: int) -> bool:
    """Whether the ``|`` at ``bar_index`` begins a closure's parameter list.

    True only when everything between the argument's start and the bar is whitespace, optionally
    preceded by ``move``. Anything else is a bitwise/pattern or, which must not be treated as a
    delimiter — doing so swallowed the following argument separators and produced a false red on a
    call site whose validator genuinely rejects.
    """
    return re.fullmatch(r"\s*(?:move\s+)?", text[arg_start:bar_index]) is not None


def _split_top_level_args(text: str) -> list[str]:
    """Split a Rust argument list on top-level commas, ignoring nesting and string literals.

    A Rust trailing comma is idiomatic and `rustfmt` inserts one, so the empty tail it produces is
    dropped — otherwise every well-formatted call site would look like it passed nothing last.
    """
    args: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[start:i])
            start = i + 1
        elif char == '"':
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 1
                elif text[i] == '"':
                    break
                i += 1
        elif char == "|" and depth == 0 and _opens_a_closure(text, start, i):
            # A closure parameter list at argument level. Skip to its closing `|` so a `,` between
            # closure parameters is not mistaken for an argument separator. Only when the `|` STARTS
            # the argument (optionally after `move`): a bitwise-or in an earlier argument used to
            # swallow the real closure's delimiters and route-check an honestly-rejecting call site.
            close = text.find("|", i + 1)
            if close != -1:
                i = close
        i += 1
    args.append(text[start:])
    while len(args) > 1 and not args[-1].strip():
        args.pop()
    return args


def _mentions_scope_config(text: str) -> bool:
    """Whether the crate names the MLX request-scope config type at all, under any spelling."""
    return MLX_REQUEST_SCOPE_CONFIG_TYPE in text


def _scope_config_aliases(text: str) -> set[str]:
    """Every local name the scope-config type answers to in this crate.

    The SC-15525 probe defeated a literal-substring reader with `use … as ScopeConfig`, `type Scope =
    MlxRequestScopeConfig`, and the qualified `<path::MlxRequestScopeConfig>::new` form — each one
    `rustfmt`-stable and each one compiling today. Enumerating spellings is a losing game, so the
    type is resolved instead: collect the local aliases, then match construction against all of them.
    """
    aliases = {MLX_REQUEST_SCOPE_CONFIG_TYPE}
    pattern = re.escape(MLX_REQUEST_SCOPE_CONFIG_TYPE)
    for match in re.finditer(
        r"\buse\s+[^;]*?\b" + pattern + r"\s+as\s+(\w+)\s*[;,}]", text
    ):
        aliases.add(match.group(1))
    for match in re.finditer(
        r"\btype\s+(\w+)\s*(?:<[^=]*>)?\s*=\s*[^;]*?\b" + pattern + r"\b[^;]*;", text
    ):
        aliases.add(match.group(1))
    return aliases


def _decode_validators_are_unconditional_rejections(text: str) -> bool:
    """Whether every decode validator this crate installs refuses outright.

    The closure twin of :func:`_configure_decode_hooks_are_unconditional_rejections`, and the one an
    MLX family actually writes. Deliberately narrow, in the same way: only a literal closure whose
    entire body is one ``Err(...)`` counts. A named function, a delegation, a ``match``, or any
    closure with a success arm is an adoption signal and must use ``DecodeRoutes``.

    Returns ``False`` — never a silent pass — when a validator reaches the scope by a route this
    reader cannot resolve. Three routes exist, because **every field of the config type is `pub`**,
    which makes ``::new`` a convention rather than a chokepoint:

    1. the checked constructor's last positional argument (what every shipping provider writes);
    2. **assignment to** ``.decode_validator`` after construction. `mlx_gen_sdxl::memory_strategy`
       already mutates the config two lines after ``::new`` (``config.attention_chunk_size = None``),
       so this shape is one line from code that ships — and invisible to a reader that only inspects
       the call site;
    3. a **struct literal**, which bypasses ``::new`` entirely. The first revision returned ``True``
       *vacuously* here: no ``::new`` match meant an empty loop meant "everything refuses", flatly
       contradicting this docstring.

    (2) and (3) are refused outright rather than parsed. A provider that declares rung 2 ``Missing``
    has no legitimate reason to install a validator by either route, so "cannot prove it refuses" and
    "should not be doing this at all" have the same answer.
    """
    aliases = _scope_config_aliases(text)
    names = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))

    # (3) A struct literal bypasses the constructor entirely. Under any of its names.
    if re.search(r"(?:" + names + r")\s*\{", text):
        return False
    # (2) Any post-construction reach for the field defeats all the care taken at the call site —
    #     an `=` overwrite, a `mem::replace`, or an `&mut` handed to a helper.
    if _DECODE_VALIDATOR_FIELD_ACCESS.search(text):
        return False

    # (1) Every recognized construction site, resolved by TYPE rather than by one literal spelling.
    # `Alias::new(`, `Type::new(` and the qualified `<path::Type>::new(` form all resolve here.
    sites = list(re.finditer(r"(?:<[^<>]*?(?:" + names + r")>|(?:" + names + r"))\s*::\s*new\s*\(", text))
    if not sites:
        # The crate never names the scope config under any alias — a rung-4-only or non-MLX adopter
        # with no validator to be wrong about. Anything else reached the `names` branch and is
        # handled above or below; a crate that NAMES the type but whose construction this reader
        # cannot resolve falls through to the `False` here, which is the inversion the SC-15525 probe
        # asked for: an unrecognized construction ARMS the gate rather than disarming it.
        return not _mentions_scope_config(text)
    for start in sites:
        open_index = text.rfind("(", start.start(), start.end())
        end = _match_paren(text, open_index)
        if end > len(text):
            return False
        args = _split_top_level_args(text[open_index + 1 : end - 1])
        validator = args[-1].strip().rstrip(",").strip()
        closure = re.match(r"(?:move\s+)?\|[^|]*\|", validator)
        if closure is None:
            return False
        if not _is_single_err_expression(validator[closure.end() :]):
            return False
    return True


# The support expression a `BoundedDecode` arm resolves to.
_MISSING_SUPPORT_EXPR = re.compile(r"^(?:MemoryStrategySupport::)?Missing$")
_CONST_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _bounded_decode_arm_tails(text: str) -> list[str]:
    """Every ``BoundedDecode`` occurrence's arm tail — the token through to end of line.

    Taken as a whole line rather than parsed, because the shapes that defeated the first regex all
    hid *between* the token and the support expression: an or-pattern (``BoundedDecode |
    BoundedAttention => Implemented``) and a guard whose ``>=`` the old ``[^=]*?`` could not cross
    (``BoundedDecode if self.tiles >= 2 => Implemented``). A line is coarse, and coarse is the right
    direction here: it can only over-capture, and over-capturing fails closed.
    """
    tails = []
    for match in re.finditer(r"\bBoundedDecode\b", text):
        newline = text.find("\n", match.end())
        tails.append(text[match.start() : len(text) if newline < 0 else newline])
    return tails


def _declares_bounded_decode_missing(files: list[str]) -> bool:
    """Whether this provider's contract textually declares ``BoundedDecode`` support **Missing**.

    Keyed on the **positive claim**. The first revision keyed the exemption on the *absence* of the
    ``decode_tile_edges`` / ``decode_overlaps`` literals, which review defeat (B) broke by building
    ``MemoryParameterRanges`` in another crate while declaring rung 2 **Implemented**: absence of
    evidence was doing the work of evidence.

    Takes the per-file evidence rather than the concatenated crate, because the const hop must be
    **scoped**. Resolving ``DECODE_SUPPORT`` by a crate-wide search let a dead or deprecated sibling
    const of the same name satisfy a live ``Implemented`` arm in another module (SC-15525 probe, root
    cause D). One hop, same file, or no exemption.

    Scoping to the file was necessary and **not sufficient**, which the first hardening pass claimed
    and a second review caught: the hop still searched for the ``= Missing`` *spelling*, so an inner
    ``mod deprecated_v1`` holding a dead ``DECODE_SUPPORT = Missing`` beside a live
    ``DECODE_SUPPORT = Implemented`` in the same file satisfied it. The hop now collects every
    declaration of that name in the file and requires **exactly one**, then checks that one's value.
    Two declarations mean this reader cannot know which the arm binds — it resolves no module paths —
    and "cannot prove it is Missing" gets the same answer as "not exempt".

    Three rules, each closing a defeated shape:

    1. **No ``Implemented`` may appear in any ``BoundedDecode`` arm line.** This is what catches the
       or-pattern and the ``>=`` guard, without this reader having to parse either.
    2. **Exactly one support arm may exist in the whole crate.** A dead ``Missing`` arm parked beside
       a live one — in a ``deprecated_v1`` module, or generated by a macro — is how root cause E
       turned a real ``Implemented`` declaration into an exemption.
    3. The single arm resolves to ``Missing`` inline, or one hop through a ``const`` **declared in the
       same file**.

    Named limitation: a macro that *generates* the support arms leaves no ``BoundedDecode`` token to
    inspect. Rule 2 turns that into "zero arms found" and therefore no exemption, which is the safe
    answer; but a macro-generated `Implemented` paired with a hand-written `Missing` arm elsewhere
    would satisfy rule 2. The registry conformance walk covers the semantic declaration, and that is
    where a macro adopter has to be caught — a text gate cannot expand macro bodies.
    """
    support_arms: list[tuple[str, str]] = []  # (rhs expression, owning file text)
    for source in files:
        for tail in _bounded_decode_arm_tails(source):
            if "=>" not in tail:
                # A bare mention — `contract.engages(…, MemoryStrategy::BoundedDecode)` and friends.
                continue
            if "Implemented" in tail:
                return False
            rhs = tail.rsplit("=>", 1)[1].strip().rstrip(",").strip()
            support_arms.append((rhs, source))
    if len(support_arms) != 1:
        return False
    expr, owning_file = support_arms[0]
    if _MISSING_SUPPORT_EXPR.match(expr):
        return True
    if not _CONST_NAME.match(expr):
        return False
    # Every `const <NAME>: MemoryStrategySupport = …;` in the arm's own file, captured by VALUE.
    # Searching for the `= Missing` spelling directly — which is what the first hardening pass did —
    # answers "does a Missing one exist?" when the question is "is the one this arm resolves to
    # Missing?". Those differ as soon as the file holds two: an inner `mod deprecated_v1` carrying a
    # dead `DECODE_SUPPORT = Missing` beside a live `DECODE_SUPPORT = Implemented` satisfied the old
    # search and bought an exemption for a crate that declares rung 2 Implemented. Scoping the hop to
    # the file (root cause D) narrowed that from crate-wide to file-wide; it did not close it.
    #
    # This reader cannot resolve Rust module paths, so it does not try to pick the right one. Two
    # declarations of the same name mean it cannot know which the arm binds, and "cannot prove it is
    # Missing" is the same answer as "not exempt".
    declarations = re.findall(
        r"\bconst\s+" + re.escape(expr) + r"\s*:\s*MemoryStrategySupport\s*=\s*([^;]+);",
        owning_file,
    )
    if len(declarations) != 1:
        return False
    return _MISSING_SUPPORT_EXPR.match(declarations[0].strip()) is not None


def _match_paren(text: str, open_index: int) -> int:
    """Index just past the ``)`` matching ``open_index``, ignoring Rust strings and comments."""
    depth = 0
    i = open_index
    while i < len(text):
        char = text[i]
        if char == "(":
            depth += 1
            i += 1
        elif char == ")":
            depth -= 1
            i += 1
            if depth == 0:
                return i
        elif char == '"':
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
        elif char == "/" and i + 1 < len(text) and text[i + 1] == "/":
            newline = text.find("\n", i)
            i = len(text) if newline == -1 else newline
        elif char == "/" and i + 1 < len(text) and text[i + 1] == "*":
            close = text.find("*/", i + 2)
            i = len(text) if close == -1 else close + 2
        else:
            i += 1
    return len(text) + 1


def check_rust_sources(root: Path) -> None:
    """Fail on any HF-cache reference in workspace Rust, or a production read of a deleted env
    side channel. See RUST_BANNED_SUBSTRINGS / DELETED_ENV_SIDE_CHANNELS for the precise scoping."""
    crates = root / "crates"
    if not crates.is_dir():
        return

    side_channel_reads = {
        name: re.compile(r"env::var(?:_os)?\s*\(\s*\"" + re.escape(name) + r"\"\s*\)")
        for name in DELETED_ENV_SIDE_CHANNELS
    }
    violations: list[str] = []
    for path in sorted(crates.rglob("*.rs")):
        relative = path.relative_to(root)
        if not IGNORED_TREE_PARTS.isdisjoint(relative.parts):
            continue
        text = path.read_text(encoding="utf-8")

        # Whole-tree HF-cache bans: every .rs of every member, tests and examples included.
        for needle in RUST_BANNED_SUBSTRINGS:
            index = text.find(needle)
            while index != -1:
                violations.append(
                    f"{relative}:{_line_of(text, index)}: banned HF-cache reference {needle!r}"
                )
                index = text.find(needle, index + 1)

        # Deleted env side channels: production `src/` reads only, `#[cfg(test)]` blocks excluded.
        if "src" in relative.parts:
            spans = _cfg_test_spans(text)
            for name, pattern in side_channel_reads.items():
                for match in pattern.finditer(text):
                    if any(start <= match.start() < end for start, end in spans):
                        continue
                    violations.append(
                        f"{relative}:{_line_of(text, match.start())}: production read of deleted "
                        f"env side channel {name!r} (inference receives model paths from the caller)"
                    )

    if violations:
        joined = "\n  ".join(violations)
        fail(f"inference source must not reference HF caches or deleted env side channels:\n  {joined}")


def check_pid_decode_route_adoption(metadata: dict, root: Path) -> None:
    """A PiD-eligible provider that adopts bounded decode must declare its routes through the shared
    ``mlx_gen_pid::DecodeRoutes`` (sc-15775).

    ``mlx_gen_pid::engine::selected_decode_tiling`` is a shared, provider-agnostic seam. Native VAE
    tile edges (256-768 output px) and PiD tile edges (512px-aligned, 1024..=4096) are disjoint by
    construction, because the student decodes a ``scale x`` super-resolved output. A provider that
    emits its native ladder into ``GenerationMemory::decode_tile_edge`` therefore turns a working
    ``use_pid`` + bounded-decode request into a hard ``budget::validate_tile`` rejection at generate
    time — and the seam must not paper over it, because silently re-planning would execute a different
    strategy than the selector chose.

    ``DecodeRoutes::new`` accepts only the native ladder and, since sc-15775, *refuses to construct* one
    that reaches into the PiD domain — so an overlapping declaration never becomes a value. That helps
    only a provider that calls it, so this is the tripwire for one that does not: it lives here, in a
    lane that fires on any workspace change, so the next adopter cannot opt out by simply not reaching
    for the shared type.

    Armed by two facts: a direct ``mlx-gen-pid`` dependency (PiD-eligible) and a
    ``register_memory_strategy`` call (its contract is resolvable, so a selector can choose a strategy
    for it), plus any sign of rung-2 adoption in the crate's own sources
    (``PID_RUNG_TWO_MARKERS``). A provider missing the first two cannot reach the hazard and is not
    asked to declare routes.

    What it verifies, precisely — and no more:

    * the crate's comment-stripped sources contain a *call* to the checked constructor, and
    * a *call* to ``validate`` on a route-named receiver.

    Both are call-syntax matches on production code with comments and literals blanked, which closes
    the three defeats the
    adversarial review found in the first revision (a ``// TODO`` mentioning ``DecodeRoutes``, an unused
    ``use``, and a trigger set narrow enough to miss a helper-built ``MemoryParameterRanges``).

    It remains a *static text* check, so its limits are named rather than implied away:

    1. It cannot prove the ``validate`` call is reached on every admission path — only that one exists.
       The weights-free registry conformance walk verifies that behaviour directly through
       ``MemoryRegistration::safety_check`` and the contract's ``pid_decode_routes`` declaration.
    2. The rung-2 trigger is a *text* proxy for the semantic fact "this provider's contract publishes
       non-empty ``decode_tile_edges``". A provider could in principle delegate every textual trace of
       rung 2 — the ranges, the strategy name, and its ``configure_decode`` hook — to another crate. The
       registry walk covers the semantic declaration; the trigger set remains deliberately wide
       (fail-closed) as an earlier diagnostic.
    3. The two *exemptions* below are narrower than the trigger set, and each is keyed on a positive,
       checkable claim rather than on an absence — because SC-15525's review showed an
       absence-keyed exemption is defeated by moving the absent text into another crate. Every
       exemption additionally requires that **both** decode seams a provider can write refuse
       unconditionally: the ``configure_decode`` trait method *and* the closure handed to
       ``MlxRequestScopeConfig::new``. The latter is the one that matters in practice — no MLX
       provider writes the former.
    4. It is a **text** gate, and two things stay outside its reach by construction. A ``macro_rules``
       expansion that generates the support arms or the ``configure_decode`` impl leaves no tokens to
       inspect; and a helper in another crate that builds a scope config is not this crate's source.
       Both are handled by *arming* rather than exempting — an unresolvable construction, or zero
       support arms, both fail closed — but neither can be positively verified here. The weights-free
       registry conformance walk is what covers the semantic declaration.

    String/character literals and constructs that cannot compile without the Rust ``test`` cfg are
    excluded from evidence matching. Trigger matching uses a separate cfg-unblanked syntax stream,
    so a parser overmatch can fail closed but cannot silently disarm the gate.
    """
    packages_by_id = {package["id"]: package for package in metadata["packages"]}
    violations: list[str] = []
    for member_id in metadata["workspace_members"]:
        package = packages_by_id[member_id]
        if package["name"] == PID_SEAM_CRATE:
            continue
        if not any(
            dependency["name"] == PID_SEAM_CRATE for dependency in package["dependencies"]
        ):
            continue
        manifest_dir = Path(package["manifest_path"]).parent
        if not manifest_dir.is_dir():
            continue
        # Comments/literals are stripped before all matching. Trigger discovery deliberately does
        # NOT use cfg-item blanking: if that lightweight parser ever overblanks an unusual Rust
        # construct, the gate fails loudly for missing evidence instead of silently disarming itself
        # by erasing a real registration or rung-2 marker.
        trigger_sources: list[str] = []
        evidence_sources: list[str] = []
        for path in sorted(manifest_dir.rglob("*.rs")):
            if not IGNORED_TREE_PARTS.isdisjoint(path.relative_to(root).parts):
                continue
            source = path.read_text(encoding="utf-8")
            trigger_sources.append(strip_rust_comments(source, strip_literals=True))
            production = _blank_spans(source, _cfg_test_spans(source))
            evidence_sources.append(strip_rust_comments(production, strip_literals=True))
        triggers = "\n".join(trigger_sources)
        evidence = "\n".join(evidence_sources)
        if "register_memory_strategy" not in triggers:
            continue
        rung_two_markers = [marker for marker in PID_RUNG_TWO_MARKERS if marker in triggers]
        if not rung_two_markers:
            continue
        # SC-15800: `MemoryRequestScope` requires `configure_decode` even for a rung-4-only adopter.
        # A single-expression typed rejection cannot emit a native tile into the PiD seam, so do not
        # mistake that trait-completeness method for a bounded-decode implementation. Every broader
        # hook shape still fails closed through the route checks below.
        #
        # Both exemptions below share one precondition, and it is the one the SC-15525 review found
        # missing: **every** decode hook the crate writes — the trait method AND the closure handed to
        # the shared MLX request scope — must be an unconditional rejection. See
        # `_decode_validators_are_unconditional_rejections` for why the closure is the load-bearing
        # half on MLX.
        # Matched with the SAME regex the helper uses. The first revision short-circuited on the
        # literal `"fn configure_decode"`, which `fn  configure_decode` (two spaces) and a
        # newline-split signature both slip past — defeating this gate's own pinned test by one
        # keystroke.
        defines_configure_decode = re.search(r"\bfn\s+configure_decode\b", triggers) is not None
        hooks_all_refuse = (
            not defines_configure_decode
            or _configure_decode_hooks_are_unconditional_rejections(triggers)
        ) and _decode_validators_are_unconditional_rejections(triggers)
        if (
            rung_two_markers == ["configure_decode"]
            and _configure_decode_hooks_are_unconditional_rejections(evidence)
            and hooks_all_refuse
        ):
            continue
        # SC-15525: the same exemption, for the provider shape that declares rung 2 **Missing** after
        # measuring it. Such a provider must still NAME `BoundedDecode` (to declare the support) and
        # `MemoryParameterRanges` (the field's type, which it populates for rung 4), so the trigger
        # set alone cannot distinguish it from an adopter.
        #
        # It is keyed on the **positive claim**, not on the absence of one. The first revision keyed it
        # on the *absence* of the `decode_tile_edges` / `decode_overlaps` literals — a proxy for
        # "publishes no domain" that review defeat (B) broke by building `MemoryParameterRanges` in
        # another crate while declaring `BoundedDecode` **Implemented**: absence of evidence was doing
        # the work of evidence. Now the crate must SAY rung 2 is Missing
        # (`_declares_bounded_decode_missing`), and a provider that says so cannot also be publishing a
        # domain — so the two domain markers stay as a corroborating, fail-closed necessary condition
        # rather than as the key.
        #
        # The support declaration is read off EVIDENCE (cfg(test)-blanked): a `#[cfg(test)]` fixture
        # asserting `BoundedDecode => Missing` must not be able to buy a production exemption. The
        # domain and hook checks are read off TRIGGERS (unblanked) for the mirror-image reason — a
        # `#![cfg(test)]` file would otherwise erase the very domain (or the very accepting closure)
        # that arms the gate. Each stream is chosen so that a parser slip over-arms rather than
        # disarms.
        publishes_decode_domain = any(
            marker in triggers for marker in ("decode_tile_edges", "decode_overlaps")
        )
        if (
            _declares_bounded_decode_missing(evidence_sources)
            and not publishes_decode_domain
            and hooks_all_refuse
        ):
            continue
        missing: list[str] = []
        if not any(marker in evidence for marker in PID_DECODE_ROUTE_CONSTRUCTION_MARKERS):
            spellings = " or ".join(
                f"`{marker}`" for marker in PID_DECODE_ROUTE_CONSTRUCTION_MARKERS
            )
            missing.append(f"never calls the checked constructor ({spellings})")
        if not PID_DECODE_ROUTE_ADMISSION_CALL.search(evidence):
            missing.append("never calls `validate` on a declared route set to gate admission")
        if not missing:
            continue
        violations.append(
            f"{package['name']} depends on {PID_SEAM_CRATE}, registers a memory-strategy contract, "
            f"and implements bounded decode, but {', and '.join(missing)}"
        )

    if violations:
        joined = "\n  ".join(violations)
        fail(
            "a PiD-eligible provider that adopts bounded decode must construct its native ladder "
            "through the checked mlx_gen_pid::DecodeRoutes::new (or assert_decode_routes) AND gate "
            "admission on the resulting `validate` (sc-15775) — native VAE tile edges are not legal "
            "PiD tiles, so emitting the native ladder into a use_pid request is refused at generate "
            "time rather than re-planned. A mention in a comment or an unused import is not "
            "adoption; this gate matches call syntax on comment-stripped source:\n  "
            f"{joined}"
        )


# A path literal shaped like a model/cache store — the thing a `$HOME` read is being used to build.
# Deliberately narrow: a `$HOME` read that joins nothing store-shaped (a bare `home()` accessor, a
# `~/` expander) is not the defect and must not be flagged, or the lint gets switched off.
SNAPSHOT_STORE_LITERAL = re.compile(
    r'"\.cache/|"Library/Application Support/|"Repos/|"\.\w+/models'
)
HOME_READ = re.compile(r'env::var(?:_os)?\("HOME"\)')
# Any env read that is NOT `"HOME"` — a named override, or one whose name arrives as a binding
# (`env::var(var)`, `env::var(format!(…))`). Its presence means the chain HAS an override.
NON_HOME_ENV_READ = re.compile(r'env::var(?:_os)?\(\s*(?!"HOME")')


def _enclosing_fn(lines: list[str], index: int) -> tuple[int, list[str]]:
    """The `fn` containing line ``index``, as (start line, body lines)."""
    for start in range(index, -1, -1):
        if re.match(r"\s*(pub\s+)?(async\s+)?fn\s+\w+", lines[start]):
            depth, cursor, opened = 0, start, False
            while cursor < len(lines):
                depth += lines[cursor].count("{") - lines[cursor].count("}")
                if "{" in lines[cursor]:
                    opened = True
                if opened and depth <= 0:
                    return start, lines[start : cursor + 1]
                cursor += 1
            return start, lines[start:]
    return index, [lines[index]]


def check_snapshot_path_derivation(root: Path) -> None:
    """Fail when a snapshot/cache path is derived from ``$HOME`` with no override in the chain.

    ``check_rust_sources`` already bans HF-cache *references* under epic 13657, but only in
    production ``src/``. The same defect survived in test harnesses, where nothing linted it: a
    resolver that reads ``$HOME`` unconditionally means pointing an env var at a real store reads
    somewhere else entirely, and the suite skips or mis-resolves **while still reporting green**.
    That is the failure mode worth a gate — a red row says "fix me", a silently-skipped one says
    nothing at all.

    Two shapes hide behind a naive ``grep`` for ``env::var("HOME")`` and only one is a defect:

    * **Fallback** — an override is read first and ``$HOME`` is only the default. Harmless, and the
      shape this repo's own passing harnesses use.
    * **Derivation** — no override anywhere in the resolution chain.

    What this enforces, precisely, and no more: a ``$HOME`` read in a function that **also** joins a
    store-shaped literal (``SNAPSHOT_STORE_LITERAL``) and reads **no** other env var. All three
    conditions are needed. Dropping the store-literal one re-flags every bare ``fn home()`` accessor
    and every ``~/``-expander; dropping the env-read one re-flags every legitimate
    ``env::var(NAME).unwrap_or_else(|_| home.join(…))``.

    Two residual gaps, named rather than implied away:

    1. It is **function-scoped**, so a resolver that reads its override in one function and joins
       ``$HOME`` in another reads as a derivation. No such site exists today — the accessor shapes
       in `mlx-gen-scail2` and `mlx-gen-krea-realtime` join nothing and are excluded by the store
       literal — but a future one would need the override moved into the joining function, or an
       exemption argued here.
    2. It cannot tell a *derived cache* (correctly given a ``$HOME`` fallback) from a *provided
       input* (which should hard-fail with the epic-13657 message). Both satisfy it. Which of the
       two a path is remains a judgement the author makes; see `mlx-gen-boogu`'s
       ``BOOGU_VISION_TEST_IMAGE`` for the required shape and `converted_root()` for the fallback.
    """
    violations: list[str] = []
    for path in sorted(root.rglob("*.rs")):
        relative = path.relative_to(root)
        if not IGNORED_TREE_PARTS.isdisjoint(relative.parts):
            continue
        lines = strip_rust_comments(path.read_text(encoding="utf-8")).split("\n")
        for index, line in enumerate(lines):
            if not HOME_READ.search(line):
                continue
            start, body = _enclosing_fn(lines, index)
            text = "\n".join(body)
            if not SNAPSHOT_STORE_LITERAL.search(text):
                continue
            if NON_HOME_ENV_READ.search(text):
                continue
            violations.append(f"{relative}:{start + 1}: {lines[start].strip()}")

    if violations:
        joined = "\n  ".join(violations)
        fail(
            "a snapshot/cache path is derived from $HOME with no env override in the resolution "
            "chain, so pointing a variable at a real store cannot reach it and the row skips while "
            "looking green. Add the override (keep $HOME as the default for a derived cache; hard-"
            "fail with the epic-13657 message for a provided input):\n  " + joined
        )


# --- test-fixture temp roots (sc-17704 / sc-17755 / sc-17768 / sc-17791) -------------------------
TEMP_DIR_READ = re.compile(r"env::temp_dir\(\)")
TEMPFILE_GUARD = re.compile(r"tempfile::|TempDir")
# The one legitimate `env::temp_dir()` in a test: a *deliberately persistent* artifact the author
# wants to open afterwards (a rendered WAV, a preview PNG, a converted snapshot). Deleting it would
# defeat the point of writing it. Two ways that shape appears, and the lint has to see both:
#
#   * the function reads the override itself — `env::var("X_WAV_OUT")…`; or
#   * the `$TMPDIR` path is the *fallback arm* of a lookup that happens elsewhere, which is how the
#     `preview_real_weights.rs` suites are written:
#         `env_path("SDXL_PREVIEW_ARTIFACT_DIR").unwrap_or_else(|| env::temp_dir().join(…))`
#     A repo-local reader (`env_path`) hides the `env::var` from a function-scoped regex, so match
#     the fallback position instead. It is a reliable signal on its own: the defect shape is a
#     `let dir = env::temp_dir().join(…)` statement, never an `unwrap_or_else` arm.
ENV_OVERRIDE_READ = re.compile(r"env::var(?:_os)?\(")
FALLBACK_CALL = re.compile(r"\.(?:unwrap_or_else|unwrap_or|ok_or_else)\s*\($")


def _inside_a_fallback_arm(text: str, index: int) -> bool:
    """Is ``index`` lexically inside a ``.unwrap_or_else( … )`` argument?

    Positional, not function-wide: an unrelated ``unwrap_or_else`` earlier in the same function
    must not launder an ordinary ``let dir = env::temp_dir().join(..)`` fixture root, which is what
    a function-scoped search did.
    """
    depth = 0
    cursor = index
    while cursor > 0:
        cursor -= 1
        char = text[cursor]
        if char in ")]}":
            depth += 1
        elif char in "([{":
            if depth == 0:
                return bool(FALLBACK_CALL.search(text[:cursor + 1]))
            depth -= 1
    return False
# Cargo's own test targets: every function in them is test code.
TEST_TARGET_DIRS = frozenset({"tests", "benches", "examples"})


def _inline_cfg_test_spans(text: str) -> list[tuple[int, int]]:
    """Byte ranges of `#[cfg(test)] mod .. { .. }` blocks (not `#[cfg(test)] mod name;`)."""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"#\[cfg\(test\)\]", text):
        tail = text[match.end() :]
        if re.match(r"\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*;", tail):
            continue  # a whole-file test module; `_test_only_files` claims it instead
        opening = text.find("{", match.end())
        if opening < 0 or ";" in text[match.end() : opening]:
            continue
        spans.append((match.start(), _match_brace(text, opening) + 1))
    return spans


def _test_only_files(root: Path) -> set[Path]:
    """Files pulled in by a `#[cfg(test)] mod name;` declaration — test code end to end."""
    files: set[Path] = set()
    for path in root.rglob("*.rs"):
        if not IGNORED_TREE_PARTS.isdisjoint(path.relative_to(root).parts):
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(
            r"#\[cfg\(test\)\]\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;", text
        ):
            name = match.group(1)
            for candidate in (
                path.with_name(f"{name}.rs"),
                path.parent / name / "mod.rs",
                path.parent / path.stem / f"{name}.rs",
            ):
                if candidate.exists():
                    files.add(candidate)
    return files


def check_test_temp_dir_guards(root: Path) -> None:
    """Fail when test code builds a fixture path from ``env::temp_dir()`` instead of a guard.

    The defect this closes is narrow and was measured, not assumed. A helper that writes
    ``env::temp_dir().join(format!("{prefix}-{pid}"))`` and cleans up with a trailing
    ``remove_dir_all`` looks tidy on a green run — every cleanup fires — and leaks on exactly two
    paths:

    1. **The panic path.** A failing test never reaches its trailing cleanup. Measured on `mlx-llm`
       for sc-17768: eleven panicking probes left ten trees behind before the fix and none after.
    2. **Same-PID collisions.** Two `#[test]`s in one binary get the *same* ``{prefix}{pid}`` path,
       so one test's cleanup deletes a fixture another is still reading. Observed in
       `mlx-llm/tests/contract_roundtrip.rs`.

    A `tempfile` guard fixes both at once: it removes the tree while unwinding, and its suffix comes
    from OS randomness rather than a PID the OS will hand out again.

    Deliberately scoped to **test code** — `#[cfg(test)]` blocks, `#[cfg(test)] mod name;` files, and
    cargo's `tests/` `benches/` `examples/` targets. Production code that materializes into `$TMPDIR`
    (`mlx-gen-seedvr2`'s bundled negative embedding, `candle-gen`'s device-format cache root) is
    making a deliberate, reviewed choice about a process- or host-lifetime file, which is a different
    question from a fixture that should not outlive its test.

    **Scope: every unguarded test temp root, whatever its name.** An earlier revision fired only on
    a *per-run varying* name (``{pid}``, a counter, a clock) on the theory that a stable name is
    bounded at one entry and therefore usually deliberate. The measurement taken while fixing
    sc-17791 refutes that, and is the reason this is scoped as it is:

    ==========================  ==============  =====================
    lane (green run on `main`)  per-run leaked  **stable-name leaked**
    ==========================  ==============  =====================
    ``candle-audio*``                        0                   **14**
    ``candle-gen*`` + contracts             14                   **16**
    ``mlx-gen*``                            30                        0
    ==========================  ==============  =====================

    **The entire audio family's leak was stable-named — none of it was deliberate.** A per-run-only
    lint would not catch a regression of the exact thing that story fixed there. Stable names are
    also the *worse* of the two for the second hazard: a ``{pid}`` path collides only between tests
    sharing a process, while a fixed name collides across every concurrent test *and* every
    concurrent ``cargo test``, which is what deleted a live fixture mid-run in
    `mlx-llm/tests/contract_roundtrip.rs`.

    Nothing syntactic separates a deliberate artifact from a fixture root someone forgot to clean —
    only the author knows — so the author has to say. That is what the exemption below is for, and
    the sixteen genuinely-deliberate sites in the tree now carry it.

    That override is also the exemption: an ``env::var(..)``-then-fall-back reads as "the file is
    the point", whether the read is in this function or the ``unwrap_or_else`` sits downstream of a
    helper that does it.
    """
    test_only = _test_only_files(root)
    violations: list[str] = []
    for path in sorted(root.rglob("*.rs")):
        relative = path.relative_to(root)
        if not IGNORED_TREE_PARTS.isdisjoint(relative.parts):
            continue
        text = strip_rust_comments(path.read_text(encoding="utf-8"))
        whole_file_is_test = path in test_only or not TEST_TARGET_DIRS.isdisjoint(relative.parts)
        spans = [] if whole_file_is_test else _inline_cfg_test_spans(text)
        lines = text.split("\n")
        for match in TEMP_DIR_READ.finditer(text):
            if not whole_file_is_test and not any(
                start <= match.start() < end for start, end in spans
            ):
                continue
            index = text.count("\n", 0, match.start())
            _, body = _enclosing_fn(lines, index)
            enclosing = "\n".join(body)
            if TEMPFILE_GUARD.search(enclosing) or ENV_OVERRIDE_READ.search(enclosing):
                continue
            # ...or the `$TMPDIR` path IS the fallback arm of a lookup made elsewhere.
            if _inside_a_fallback_arm(text, match.start()):
                continue
            violations.append(f"{relative}:{index + 1}: {lines[index].strip()}")

    if violations:
        joined = "\n  ".join(violations)
        fail(
            "test code builds a fixture path from env::temp_dir() with no tempfile guard in the "
            "enclosing function. That tree survives a panicking test and collides with a sibling "
            "test at the same PID (sc-17704 / sc-17755 / sc-17768 / sc-17791). Mint it from "
            "`tempfile::tempdir()` and delete the trailing remove_dir_all — the guard is the "
            "cleanup. A deliberately persistent artifact keeps its `env::var(..)` override in "
            "front:\n  " + joined
        )


def _rust_const_declarations(source: str) -> dict[str, str]:
    """Map every visibility-carrying ``const`` in a Rust source to its value text.

    Module-level or associated, at any indentation, under any of `pub`, `pub(crate)`, `pub(super)`
    or `pub(in path)`.

    Comment-stripped first, so a doc comment quoting a declaration — which these crates do
    constantly, since every constant carries its provenance in prose — cannot be mistaken for one.

    ``const _: () = assert!(..);`` is a compile-time assertion rather than a named declaration, and
    two crates asserting different things under the anonymous name would be compared as if they were
    the same constant, so it is dropped.
    """
    stripped = strip_rust_comments(source)
    return {
        match.group(1): " ".join(match.group(3).split())
        for match in RUST_PUB_CONST.finditer(stripped)
        if set(match.group(1)) != {"_"}
    }


def _normalize_const_value(value: str) -> str:
    """Strip what only spells a value (casts, digit separators, whitespace), never what it is.

    Applied to the *code* around string and character literals and never inside one. Every rule here
    is a claim about Rust syntax, and none of them holds inside a literal: `as` is a keyword outside
    one and the word "as" inside one, `2_3` is a digit separator outside one and two characters of an
    HF repo id inside one, and a space is layout outside one and content inside one. Run over the
    whole text they collapse `"…the instructions are as follows."` to `"…theinstructionsare."`,
    `"SceneWorks/wan-2_2"` to `"SceneWorks/wan-22"` and `"euler a"` to `"eulera"` — three ways for two
    backends whose strings genuinely differ to compare equal. That is the same false-green class this
    gate exists to refuse, and this PR is what makes it reachable: the normalizer used to see one
    family's two files and now sees every declaration in 66 crates, 30 of which carry prose.
    """
    out: list[str] = []
    for kind, text in _rust_chunks(value):
        if kind == "literal":
            out.append(text)
            continue
        text = RUST_CONST_CAST.sub("", text)
        text = RUST_NUMERIC_LITERAL.sub(lambda match: match.group(0).replace("_", ""), text)
        out.append(text.replace(" ", ""))
    return "".join(out)


def _arithmetic_value(node: ast.AST) -> float | None:
    """Fold a literal arithmetic AST to a float, or None if it is anything richer than that.

    Deliberately an explicit walk rather than ``eval`` on an allowlisted tree: nothing in this
    script should be able to execute text read out of a crate's sources, even text that passed a
    filter.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return None
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        operand = _arithmetic_value(node.operand)
        if operand is None:
            return None
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        return None
    if isinstance(node, ast.BinOp):
        left = _arithmetic_value(node.left)
        right = _arithmetic_value(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return None if right == 0.0 else left / right
        return None
    return None


def _const_numbers(
    value: str, declarations: dict[str, str], depth: int = 0
) -> tuple[float, ...] | None:
    """Resolve a constant's value text to numbers, following identifiers *within the same crate*.

    Same-crate resolution is the point: `SIZE_MULTIPLE = VAE_RATIO as u32 * 2` has to fold against
    the `VAE_RATIO` its own backend declares, so that a crate which quietly redefines `VAE_RATIO`
    still reports the `SIZE_MULTIPLE` it actually compiles to.

    Returns None — never a default, never a zero — for anything that is not purely numeric (strings,
    bools, enum paths, nested aggregates). Callers treat None as "cannot verify", which is a
    violation wherever a number was required, and a fallback to text equality wherever it was not.
    """
    if depth > 8:
        return None
    text = _normalize_const_value(value)
    if not text:
        return None

    is_array = text.startswith("[") and text.endswith("]")
    is_tuple = text.startswith("(") and text.endswith(")") and "," in text
    if is_array or is_tuple:
        inner = text[1:-1]
        if "[" in inner or "(" in inner:
            return None  # nested aggregate — out of scope, compared as text instead
        numbers: list[float] = []
        for element in inner.split(","):
            if not element:
                continue  # trailing comma
            resolved = _const_numbers(element, declarations, depth + 1)
            if resolved is None or len(resolved) != 1:
                return None
            numbers.append(resolved[0])
        return tuple(numbers) if numbers else None

    unresolved = False

    def _substitute(match: re.Match[str]) -> str:
        nonlocal unresolved
        name = match.group(0)
        resolved = (
            _const_numbers(declarations[name], declarations, depth + 1)
            if name in declarations
            else None
        )
        if resolved is None or len(resolved) != 1:
            unresolved = True
            return "0"
        return repr(resolved[0])

    substituted = RUST_IDENTIFIER.sub(_substitute, text)
    if unresolved:
        return None
    try:
        tree = ast.parse(substituted, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    folded = _arithmetic_value(tree.body)
    return None if folded is None else (folded,)


def _relative_max_abs_diff(
    left: tuple[float, ...], right: tuple[float, ...]
) -> float:
    """Worst elementwise ``|l - r| / max(|l|, |r|)``.

    Relative max-abs-diff and nothing else. A norm, a cosine or a checksum over these vectors would
    be blind to exactly the single-element defects this gate exists to catch — that failure has been
    paid for repeatedly in this model family.
    """
    worst = 0.0
    for lhs, rhs in zip(left, right):
        difference = abs(lhs - rhs)
        scale = max(abs(lhs), abs(rhs))
        worst = max(worst, difference if scale == 0.0 else difference / scale)
    return worst


def _dual_backend_families(metadata: dict) -> list[tuple[str, Path, Path]]:
    """Every family with both a ``candle-gen-X`` and an ``mlx-gen-X`` workspace member.

    Derived from `cargo metadata`, never from a list kept alongside it. A curated pair table is the
    same shape of claim this gate exists to refuse — it asserts a coverage nothing verifies, and a
    family added to the workspace without being added to the table is silently uncompared while the
    gate keeps printing OK.
    """
    crates: dict[str, Path] = {}
    for package in metadata.get("packages", []):
        name = package.get("name", "")
        if name.startswith("candle-gen-") or name.startswith("mlx-gen-"):
            crates[name] = Path(package["manifest_path"]).parent

    families: list[tuple[str, Path, Path]] = []
    for name, directory in sorted(crates.items()):
        if not name.startswith("candle-gen-"):
            continue
        family = name[len("candle-gen-") :]
        sibling = crates.get(f"mlx-gen-{family}")
        if sibling is not None:
            families.append((family, directory, sibling))
    if not families:
        fail(
            "cross-backend geometry: cargo metadata reported no candle-gen-X/mlx-gen-X pair at all. "
            "The workspace has 33; a gate that finds none compares nothing while reporting green."
        )
    return families


def _crate_pub_consts(
    crate: Path, subdirectory: str, *, prefix: str | None = None
) -> dict[str, set[str]]:
    """Every visibility-carrying ``const`` under ``crate/subdirectory``, name to distinct value texts.

    A name may be declared more than once in one crate — `MODEL_ID` is declared per variant module
    in several families — so the value is a *set*. Comparing sets rather than picking one keeps the
    comparison honest: two backends agree when they declare the same values under a name, whichever
    module each puts them in.
    """
    declarations: dict[str, set[str]] = {}
    for path in sorted((crate / subdirectory).rglob("*.rs")):
        for constant, value in _rust_const_declarations(path.read_text(encoding="utf-8")).items():
            if prefix is not None and not constant.startswith(prefix):
                continue
            declarations.setdefault(constant, set()).add(value)
    return declarations


def _shares_a_fixture_file(candle_crate: Path, mlx_crate: Path) -> bool:
    """Do the two crates commit at least one same-named fixture file with the same bytes?

    This is the premise `CROSS_BACKEND_FIXTURE_FAMILIES` membership rests on, re-derived rather than
    asserted: the reason a family's hand-typed geometry has to be compared at all is that both lanes
    load the *same bytes* through it. A membership claim nothing re-checks is exactly the shape of
    defect this gate exists for, so a family whose fixtures are no longer shared stops satisfying
    its own entry and the entry has to go.

    One shared file is enough to establish it, and no more is claimed: same-named files that differ
    are outside what this can judge (see the `sana` note on the table).
    """
    for path in sorted((candle_crate / "tests" / "fixtures").rglob("*")):
        if not path.is_file():
            continue
        sibling = mlx_crate / "tests" / "fixtures" / path.relative_to(
            candle_crate / "tests" / "fixtures"
        )
        if sibling.is_file() and sibling.read_bytes() == path.read_bytes():
            return True
    return False


def _single_valued(declarations: dict[str, set[str]]) -> dict[str, str]:
    """The subset a same-crate reference can resolve against — one declaration, one value."""
    return {name: next(iter(values)) for name, values in declarations.items() if len(values) == 1}


def _fold_const_arithmetic(text: str, single: dict[str, str]) -> str:
    """Fold ``NAME.len()`` and ``NAME[i]`` over same-crate array constants."""

    def elements(name: str) -> list[str] | None:
        if name not in single:
            return None
        value = RUST_PATH_QUALIFIER.sub("", single[name])
        if "[" not in value or not value.rstrip().endswith("]"):
            return None
        inner = value[value.index("[") + 1 : value.rindex("]")]
        if "[" in inner:
            return None
        return [element.strip() for element in inner.split(",") if element.strip()]

    def fold_len(match: re.Match[str]) -> str:
        found = elements(match.group(1))
        return str(len(found)) if found is not None else match.group(0)

    def fold_index(match: re.Match[str]) -> str:
        found = elements(match.group(1))
        if found is None:
            return match.group(0)
        try:
            tree = ast.parse(_normalize_const_value(match.group(2)), mode="eval")
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            return match.group(0)
        index = _arithmetic_value(tree.body)
        if index is None or index != int(index):
            return match.group(0)
        index = int(index)
        return found[index] if -len(found) <= index < len(found) else match.group(0)

    for folder, pattern in ((fold_len, RUST_CONST_LEN), (fold_index, RUST_CONST_INDEX)):
        for _ in range(4):
            folded = pattern.sub(folder, text)
            if folded == text:
                break
            text = folded
    return text


def _canonical_const_value(value: str, declarations: dict[str, set[str]], depth: int = 0) -> str:
    """A constant's value reduced to a form two backends can be compared on.

    Numbers fold to numbers, so `usize`/`i32` and `1e-5`/`0.00001` compare equal. Everything else
    reduces to normalized text with same-crate identifiers resolved through, so that a backend
    naming a constant (`crate::config::SD3_5_LARGE_ID`) compares equal to a backend spelling the
    same string out. What cannot be resolved is returned as text and compared as text — never
    dropped, because a value the gate stops comparing is a value nothing checks.
    """
    if depth > 8:
        return "text:" + _normalize_const_value(value)
    single = _single_valued(declarations)
    text = RUST_BACKEND_SHIM.sub("backend::", RUST_PATH_QUALIFIER.sub("", value))
    # The shared contracts crate is reachable both through a backend shim
    # (`candle_gen::gen_core::X` / `mlx_gen::gen_core::X`) and bare (`use …::gen_core;` then
    # `gen_core::X`). The sc-18306 encoder-contract sweep writes it bare on the candle side and
    # shimmed on the mlx side, so without this fold every contract constant reads as divergent on
    # spelling alone. `backend::` is kept for non-gen_core paths, where a bare name really can be a
    # different (crate-local) item.
    text = text.replace("backend::gen_core::", "gen_core::")
    text = _fold_const_arithmetic(text, single)

    numbers = _const_numbers(text, single)
    if numbers is not None:
        return "number:" + repr(numbers)

    normalized = _normalize_const_value(text)
    if normalized in single and single[normalized] != value:
        return _canonical_const_value(single[normalized], declarations, depth + 1)

    def resolve(match: re.Match[str]) -> str:
        name = match.group(0)
        if name in single and single[name] != value:
            return _canonical_const_value(single[name], declarations, depth + 1)
        return name

    # Resolution is a claim about *code*, for the same reason `_normalize_const_value`'s rewrites
    # are: an ALL-CAPS word inside a string literal is prose, not a reference to a crate-local
    # constant that happens to share its spelling. flux2's `SYSTEM_MESSAGE_UPSAMPLING_T2I` says
    # "Put ALL text in quotation marks"; once the candle crate gained a `pub const ALL`, running
    # this substitution over the literal rewrote that word into the constant's value on one side
    # only and reported two byte-identical constants as divergent (sc-11045).
    return "text:" + "".join(
        chunk if kind == "literal" else re.sub(r"\b[A-Z_][A-Z0-9_]+\b", resolve, chunk)
        for kind, chunk in _rust_chunks(normalized)
    )


def _canonical_const_values(values: set[str], declarations: dict[str, set[str]]) -> set[str]:
    return {_canonical_const_value(value, declarations) for value in values}


def _delimiter_span(text: str, open_index: int) -> int | None:
    """Index of the delimiter closing the one at `open_index`, or None when it never closes.

    String-literal aware for the same reason `_normalize_const_value` is: a brace or bracket inside
    a token literal (`"<|im_start|>"` is safe, but a prompt template need not be) is content, not
    structure.
    """
    openers = {"(": ")", "[": "]", "{": "}"}
    if text[open_index] not in openers:
        return None
    stack = [text[open_index]]
    index = open_index + 1
    while index < len(text):
        char = text[index]
        if char == '"':
            index += 1
            while index < len(text) and text[index] != '"':
                index += 2 if text[index] == "\\" else 1
        elif char in openers:
            stack.append(char)
        elif char in openers.values():
            if char != openers[stack.pop()]:
                return None
            if not stack:
                return index
        index += 1
    return None


def _canonical_leaf(value: str) -> str:
    """Strip what only wraps a canonical value: the kind tag, `&`, and a transparent `Some`.

    `Some` is unwrapped so an optional aggregate is addressed by the fields it actually holds
    (`packing.supports_file`), while `Some(x)` against `None` still compares as one whole value —
    the unwrap only happens when both sides are `Some`, because `None` has nothing to unwrap.
    """
    text = value
    if text.startswith("text:"):
        text = text[len("text:") :]
    text = text.strip()
    while text.startswith("&"):
        text = text[1:].strip()
    if text.startswith("Some(") and _delimiter_span(text, 4) == len(text) - 1:
        return _canonical_leaf(text[5:-1])
    return text


def _slice_element_label(element: str) -> str | None:
    """Label a slice element by its leading string field (`purpose`, `role`), else None.

    Keying an element by its own name is what lets a backend that ships one MORE of something be
    reported as that one extra element rather than as a wholesale divergence of the list.
    """
    text = _canonical_leaf(element)
    open_index = text.find("{")
    if not text.endswith("}") or open_index < 0:
        return None
    if _delimiter_span(text, open_index) != len(text) - 1:
        return None
    first = _split_top_level_args(text[open_index + 1 : -1])[0]
    _, separator, rest = first.partition(":")
    rest = rest.strip()
    if not separator or len(rest) < 2 or not rest.startswith('"') or not rest.endswith('"'):
        return None
    return rest[1:-1]


def _aggregate_parts(value: str) -> tuple[str, list[tuple[str, str]]] | None:
    """Split one canonical value into its type head and labeled sub-values, or None for a leaf.

    Struct fields are labeled `.name` and slice elements `[key]`, so a sub-value is addressed by a
    path (`.packing.supports_file`, `.prompt_executions[krea_t2i].length`) that reads the same way
    the Rust does.
    """
    text = _canonical_leaf(value)
    if not text or text[-1] not in "}]":
        return None
    opener = "{" if text.endswith("}") else "["
    open_index = text.find(opener)
    if open_index < 0 or _delimiter_span(text, open_index) != len(text) - 1:
        return None
    head = text[:open_index]
    items = [item for item in _split_top_level_args(text[open_index + 1 : -1]) if item.strip()]
    if not items:
        return head, []
    if opener == "{":
        fields: list[tuple[str, str]] = []
        for item in items:
            name, separator, rest = item.partition(":")
            if not separator or not name.strip().isidentifier():
                return None
            fields.append(("." + name.strip(), rest))
        return head, fields
    labels = [_slice_element_label(item) for item in items]
    if any(label is None for label in labels) or len(set(labels)) != len(labels):
        labels = [str(index) for index in range(len(items))]
    return head, [(f"[{label}]", item) for label, item in zip(labels, items)]


def _value_divergences(left: str, right: str, path: str = "", depth: int = 0) -> list[str]:
    """Every sub-value path at which two canonical values differ, coarsest-first.

    Two values that will not decompose the same way — a leaf, a different type head, reordered
    elements — yield the one path that contains them, so a divergence is always reported SOMEWHERE.
    Nothing here can return an empty list for values that differ, which is what makes it safe to
    treat "no unexempted path" as agreement.
    """
    if left == right:
        return []
    if depth > 8:
        return [path or "."]
    left_split = _aggregate_parts(left)
    right_split = _aggregate_parts(right)
    if left_split is None or right_split is None:
        return [path or "."]
    left_head, left_parts = left_split
    right_head, right_parts = right_split
    if left_head != right_head:
        return [path or "."]
    left_map = dict(left_parts)
    right_map = dict(right_parts)
    if len(left_map) != len(left_parts) or len(right_map) != len(right_parts):
        return [path or "."]
    shared = [label for label in left_map if label in right_map]
    if shared != [label for label in right_map if label in left_map]:
        # Same members in a different order. For a slice that is a real difference and there is no
        # smaller path that carries it.
        return [path or "."]
    divergences: list[str] = []
    for label, value in left_parts:
        if label in right_map:
            divergences.extend(
                _value_divergences(value, right_map[label], path + label, depth + 1)
            )
        else:
            divergences.append(path + label)
    divergences.extend(path + label for label, _ in right_parts if label not in left_map)
    return divergences


def _narrowed_divergence_violations(
    family: str,
    constant: str,
    candle_relative: str,
    mlx_relative: str,
    left: set[str],
    right: set[str],
    exempted_paths: dict[str, str],
) -> list[str]:
    """Hold a sub-field exemption to exactly the sub-fields it names."""
    if len(left) != 1 or len(right) != 1:
        return [
            f"{family}: `{constant}` carries a sub-field divergence exemption but is declared with "
            f"{len(left)}/{len(right)} different values across {candle_relative}/{mlx_relative}, so "
            "the gate cannot tell which pair the exemption addresses — exempt the whole constant "
            "with a written reason instead"
        ]
    diverging = _value_divergences(next(iter(left)), next(iter(right)))
    violations = [
        f"{family}: `{constant}{path}` diverges and is not one of the sub-fields its exemption "
        f"names ({', '.join(sorted(exempted_paths))}): {candle_relative} and {mlx_relative} "
        "disagree about it. Fix the backend that is wrong, or record this sub-field too"
        for path in diverging
        if path not in exempted_paths
    ]
    violations.extend(
        f"{family}: `{constant}{path}` carries a sub-field divergence exemption but the two "
        "backends now agree about it — delete that path from the exemption"
        for path in sorted(set(exempted_paths) - set(diverging))
    )
    return violations


def check_cross_backend_geometry(metadata: dict, root: Path) -> None:
    """Fail when a family's two backends declare different geometry, in shipped code or in fixtures.

    Four things are checked, and none is redundant.

    1. **Every dual-backend family is compared.** The families come from `cargo metadata`, so the
       gate's coverage is the workspace's contents rather than a list someone remembered to update.

    2. **The two crates agree on every constant they both declare.** Every ``const`` carrying a
       visibility modifier — ``pub``, ``pub(crate)``, ``pub(super)``, ``pub(in path)`` — at any
       indentation, module-level or associated on a type, anywhere under each crate's ``src/``, is
       compared by value. So `usize`/`i32` and `f64`/`f32` spellings of the same number pass while a
       genuine difference reds. A bare private ``const`` is *not* read; ``RUST_PUB_CONST``'s comment
       says why, and what that leaves for sc-19696. A divergence is either fixed or carries a
       written reason in ``CROSS_BACKEND_GEOMETRY_EXEMPTIONS``; a reason that no longer describes a
       divergence is a failure wherever the gate can resolve both sides to a common form, so the
       exemptions cannot outlive their subject where they are checkable at all — the table's own
       comment names the three entries that are not.

       An *aggregate* constant — the encoder, tokenizer and prompt-execution contracts, each one
       `const` holding twenty-odd behavior-bearing fields — is exempted per sub-field instead, in
       ``CROSS_BACKEND_GEOMETRY_FIELD_EXEMPTIONS``, so that recording one known difference does not
       stop comparing the rest of the aggregate. Every path that diverges must be listed, and every
       listed path must still diverge.

       A family that shares *no* constant name is compared against nothing while every other clause
       passes, so that is a failure too, unless it is listed in
       ``CROSS_BACKEND_GEOMETRY_NO_SHARED_CONSTANTS`` with a reason — and listing a family that does
       share one is itself a failure.

    3. **Both agree with the reference.** Agreement alone would have been satisfiable by copying the
       wrong value across, which is precisely the move this gate must not reward: the defect it was
       written for (sc-19419) was a wrong value pinned *as if correct* by a test whose doc comment
       claimed cross-backend coverage it could not have. So `CROSS_BACKEND_GEOMETRY_REFERENCE` holds
       values read out of the released checkpoint through diffusers, and both sides are held to them.

    4. **The test fixtures' geometry agrees too** (sc-19496). Every family in
       ``CROSS_BACKEND_FIXTURE_FAMILIES`` — `anima`, `bernini`, `krea`, `ltx`, `minimax-h3`, `sana` —
       commits
       fixture files byte-identically on both sides and loads them through geometry each lane
       hand-types, so a drift in either config leaves both lanes internally consistent and both parity
       suites green while the two backends compare a tensor dumped at one shape against a model built
       at another. Those numbers are declared as ``SHARED_FIXTURE_*`` constants and compared here,
       name set and values both; declaring none at all is a failure. The membership premise is
       re-derived per run by ``_shares_a_fixture_file`` rather than trusted, so an entry cannot outlive
       the sharing that justifies it. This does **not** claim every same-named fixture file matches —
       ``sana`` commits two genuinely different `dcae_encode_golden.safetensors` — nor does it reach
       geometry a golden records in its own ``__metadata__`` and both lanes read back, which needs no
       comparison because there is only one copy of the number.

    Fail-closed throughout. A missing crate, a family that yields zero shared constants, or a
    required constant that will not resolve to a number is a failure, not a skip — a gate that
    quietly stops comparing is worse than no gate, because it still reports green. What it does
    *not* do is prove the two backends compute the same thing from these numbers; that is what the
    `cross_backend.rs` fixture parity tests are for.
    """
    violations: list[str] = []
    families = _dual_backend_families(metadata)
    known = {family for family, _, _ in families}

    for family in sorted(CROSS_BACKEND_GEOMETRY_EXEMPT_FAMILIES):
        if family not in known:
            violations.append(
                f"`{family}` is exempted from cross-backend comparison but is not a dual-backend "
                "family any more — drop the exemption"
            )
    for family in sorted(CROSS_BACKEND_GEOMETRY_REFERENCE):
        if family not in known:
            violations.append(
                f"`{family}` is pinned against a reference but is not a dual-backend family any "
                "more — drop the reference block"
            )
    for family in sorted(CROSS_BACKEND_FIXTURE_FAMILIES):
        if family not in known:
            violations.append(
                f"`{family}` is required to declare `{CROSS_BACKEND_FIXTURE_PREFIX}*` fixture "
                "geometry but is not a dual-backend family any more — drop the entry"
            )
    # A reference pin is a strictly stronger claim than shared fixtures, so a pinned family that is
    # not also required to declare its fixture geometry would silently narrow this gate's reach.
    for family in sorted(set(CROSS_BACKEND_GEOMETRY_REFERENCE) - set(CROSS_BACKEND_FIXTURE_FAMILIES)):
        violations.append(
            f"`{family}` is pinned against the diffusers reference but is missing from "
            "CROSS_BACKEND_FIXTURE_FAMILIES, so its fixture geometry is not required to be declared "
            "at all — add it with the reason its two crates share fixture bytes"
        )
    for family, constant in sorted(
        set(CROSS_BACKEND_GEOMETRY_EXEMPTIONS) | set(CROSS_BACKEND_GEOMETRY_FIELD_EXEMPTIONS)
    ):
        if family not in known:
            violations.append(
                f"`{family}`: `{constant}` carries a divergence exemption but `{family}` is not a "
                "dual-backend family any more — drop the exemption"
            )
    for family, constant in sorted(
        set(CROSS_BACKEND_GEOMETRY_EXEMPTIONS) & set(CROSS_BACKEND_GEOMETRY_FIELD_EXEMPTIONS)
    ):
        violations.append(
            f"`{family}`: `{constant}` carries both a whole-constant and a sub-field divergence "
            "exemption. The whole-constant one wins and the sub-field paths would never be "
            "checked — keep exactly one"
        )
    for (family, constant), paths in sorted(CROSS_BACKEND_GEOMETRY_FIELD_EXEMPTIONS.items()):
        if not paths or any(not reason.strip() for reason in paths.values()):
            violations.append(
                f"`{family}`: `{constant}` carries a sub-field divergence exemption with a path "
                "that has no written reason — a divergence with no reason is a defect"
            )
    for family in sorted(CROSS_BACKEND_GEOMETRY_NO_SHARED_CONSTANTS):
        if family not in known:
            violations.append(
                f"`{family}` is recorded as sharing no constant between its backends but is not a "
                "dual-backend family any more — drop the entry"
            )

    shared_exemptions: set[tuple[str, str]] = set()

    for family, candle_crate, mlx_crate in families:
        if family in CROSS_BACKEND_GEOMETRY_EXEMPT_FAMILIES:
            continue
        candle_relative = candle_crate.relative_to(root).as_posix()
        mlx_relative = mlx_crate.relative_to(root).as_posix()

        sides: dict[str, dict[str, set[str]]] = {}
        for relative, crate in ((candle_relative, candle_crate), (mlx_relative, mlx_crate)):
            if not (crate / "src" / "lib.rs").is_file():
                fail(
                    f"cross-backend geometry: {relative}/src/lib.rs is missing, so the gate cannot "
                    "read the crate it was told to compare."
                )
            sides[relative] = _crate_pub_consts(crate, "src")

        candle = sides[candle_relative]
        mlx = sides[mlx_relative]

        # Clause 2's floor. Without it a family whose crates share no constant name runs the loop
        # below zero times, reaches the reference and fixture clauses with nothing to say (both are
        # keyed on families this gate pins), and lands in the OK line having compared nothing —
        # which is what `joycaption` and `sam3` did, and what any future regression in
        # `RUST_PUB_CONST` would silently do to every non-reference family at once.
        shared = set(candle) & set(mlx)
        recorded_as_empty = family in CROSS_BACKEND_GEOMETRY_NO_SHARED_CONSTANTS
        if not shared and not recorded_as_empty:
            violations.append(
                f"{family}: {candle_relative} and {mlx_relative} share no constant name, so the "
                "gate compared nothing for this family. Either the two backends really do declare "
                "no common constant — record that in "
                "`CROSS_BACKEND_GEOMETRY_NO_SHARED_CONSTANTS` with the reason — or the parser has "
                "stopped reaching declarations it used to read"
            )
        if shared and recorded_as_empty:
            violations.append(
                f"{family}: is recorded as sharing no constant between its backends, but "
                f"{len(shared)} are now declared on both sides ({', '.join(sorted(shared)[:5])}"
                f"{', …' if len(shared) > 5 else ''}) — delete the entry so they are compared"
            )

        for constant in sorted(shared):
            left = _canonical_const_values(candle[constant], candle)
            right = _canonical_const_values(mlx[constant], mlx)
            exempted = (family, constant) in CROSS_BACKEND_GEOMETRY_EXEMPTIONS
            narrowed = CROSS_BACKEND_GEOMETRY_FIELD_EXEMPTIONS.get((family, constant))
            if exempted or narrowed is not None:
                # Seen, whether or not it still diverges — so the "no longer declared on both
                # sides" sweep below cannot also fire and report the opposite of what is true.
                shared_exemptions.add((family, constant))
            if left == right:
                if exempted or narrowed is not None:
                    violations.append(
                        f"{family}: `{constant}` carries a divergence exemption but the two "
                        "backends now agree about it — delete the exemption"
                    )
                continue
            if exempted:
                continue
            if narrowed is not None:
                violations.extend(
                    _narrowed_divergence_violations(
                        family,
                        constant,
                        candle_relative,
                        mlx_relative,
                        left,
                        right,
                        narrowed,
                    )
                )
                continue
            violations.append(
                f"{family}: `{constant}` diverges: {candle_relative} says "
                f"{sorted(candle[constant])}, {mlx_relative} says {sorted(mlx[constant])}"
            )

        for constant, expected in sorted(CROSS_BACKEND_GEOMETRY_REFERENCE.get(family, {}).items()):
            for relative, declarations in ((candle_relative, candle), (mlx_relative, mlx)):
                if constant not in declarations:
                    violations.append(
                        f"{relative}: `{constant}` is pinned against the diffusers reference but is "
                        "not declared anywhere under src/"
                    )
                    continue
                if len(declarations[constant]) != 1:
                    violations.append(
                        f"{relative}: `{constant}` is declared with {len(declarations[constant])} "
                        "different values, so the gate cannot tell which one the reference pins"
                    )
                    continue
                value = next(iter(declarations[constant]))
                numbers = _const_numbers(
                    _fold_const_arithmetic(
                        RUST_PATH_QUALIFIER.sub("", value), _single_valued(declarations)
                    ),
                    _single_valued(declarations),
                )
                if numbers is None:
                    violations.append(
                        f"{relative}: `{constant}` = {value!r} does not resolve to numbers, so the "
                        "gate cannot hold it to the reference"
                    )
                    continue
                if len(numbers) != len(expected) or (
                    _relative_max_abs_diff(numbers, expected) > CROSS_BACKEND_GEOMETRY_TOLERANCE
                ):
                    violations.append(
                        f"{relative}: `{constant}` = {value!r} resolves to {numbers}, but the "
                        f"released checkpoint read through diffusers says {expected}"
                    )

        candle_fixtures = _crate_pub_consts(
            candle_crate, "tests", prefix=CROSS_BACKEND_FIXTURE_PREFIX
        )
        mlx_fixtures = _crate_pub_consts(mlx_crate, "tests", prefix=CROSS_BACKEND_FIXTURE_PREFIX)
        if family in CROSS_BACKEND_FIXTURE_FAMILIES:
            if not _shares_a_fixture_file(candle_crate, mlx_crate):
                violations.append(
                    f"{family}: is required to declare `{CROSS_BACKEND_FIXTURE_PREFIX}*` fixture "
                    f"geometry because {candle_relative} and {mlx_relative} commit the same fixture "
                    "bytes, but no same-named file under their tests/fixtures/ is byte-identical any "
                    f"more — the reason recorded in CROSS_BACKEND_FIXTURE_FAMILIES ("
                    f"{CROSS_BACKEND_FIXTURE_FAMILIES[family]}) no longer describes them"
                )
            elif not (candle_fixtures and mlx_fixtures):
                violations.append(
                    f"{family}: its two crates commit byte-identical fixtures, but "
                    f"{candle_relative if not candle_fixtures else mlx_relative} declares no "
                    f"`{CROSS_BACKEND_FIXTURE_PREFIX}*` constants under tests/. The fixture geometry "
                    "is hand-typed on both sides and nothing else compares it"
                )
        for constant in sorted(set(candle_fixtures) - set(mlx_fixtures)):
            violations.append(
                f"{family}: `{constant}` is declared in {candle_relative}/tests but not in "
                f"{mlx_relative}/tests"
            )
        for constant in sorted(set(mlx_fixtures) - set(candle_fixtures)):
            violations.append(
                f"{family}: `{constant}` is declared in {mlx_relative}/tests but not in "
                f"{candle_relative}/tests"
            )
        for constant in sorted(set(candle_fixtures) & set(mlx_fixtures)):
            left = _canonical_const_values(candle_fixtures[constant], candle_fixtures)
            right = _canonical_const_values(mlx_fixtures[constant], mlx_fixtures)
            if left != right:
                violations.append(
                    f"{family}: fixture geometry `{constant}` diverges: {candle_relative}/tests "
                    f"says {sorted(candle_fixtures[constant])}, {mlx_relative}/tests says "
                    f"{sorted(mlx_fixtures[constant])}"
                )

    for family, constant in sorted(
        (set(CROSS_BACKEND_GEOMETRY_EXEMPTIONS) | set(CROSS_BACKEND_GEOMETRY_FIELD_EXEMPTIONS))
        - shared_exemptions
    ):
        if family in known and family not in CROSS_BACKEND_GEOMETRY_EXEMPT_FAMILIES:
            violations.append(
                f"{family}: `{constant}` carries a divergence exemption but is no longer declared "
                "on both sides — delete the exemption"
            )

    if violations:
        joined = "\n  ".join(violations)
        fail(
            "a model family's two backends disagree about geometry, or disagree with the released "
            "checkpoint, or carry an exemption that no longer describes anything. These crates "
            "cannot import each other (mlx-gen is macOS-only), so no per-crate test can see this "
            "and a stale value survives as long as its own crate keeps asserting it (sc-19419). Fix "
            "the backend that is wrong against the reference — never by copying the other backend's "
            "number — or record why the two differ:\n  " + joined
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="require Cargo to validate entirely from its local cache",
    )
    args = parser.parse_args()

    try:
        check_filesystem()
        metadata = cargo_metadata(args.offline)
        check_graph(metadata)
        check_network_clients(metadata)
        check_rust_sources(ROOT)
        check_pid_decode_route_adoption(metadata, ROOT)
        check_snapshot_path_derivation(ROOT)
        check_test_temp_dir_guards(ROOT)
        check_cross_backend_geometry(metadata, ROOT)
    except (AssertionError, json.JSONDecodeError) as error:
        print(f"workspace gate: FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "workspace gate: OK "
        f"({EXPECTED_MEMBER_COUNT} path members, one lockfile, explicit registries, pinned backends, "
        "intentional tokenizer split, no network clients, no HF-cache references, "
        "no $HOME-derived snapshot paths, no unguarded test temp roots, "
        "no cross-backend geometry drift)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
