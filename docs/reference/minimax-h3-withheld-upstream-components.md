# MiniMax-H3 — the withheld upstream components (record)

- **Story:** sc-17162 (epic sc-17137 — MiniMax-H3 joint audio+video generation)
- **Status:** Accepted, engine side. The catalog-copy items it defers to are named in
  [What this repository does not decide](#what-this-repository-does-not-decide).
- **Applies to:** `candle-gen-minimax-h3` and `mlx-gen-minimax-h3`.
- **Upstream re-check:** 2026-08-15 (epic sc-17137). None of the components below had been
  published. `huggingface/diffusers#14371` touched the MiniMax-H3 pipeline but was a bit-exact
  refactor: no lattice, geometry or attention behaviour changed.

MiniMax ships the H3 weights without three components that its own hosted Hailuo product runs, and
declares a fourth surface — the `<d>` dialogue markers — in a config file with nothing behind it.
This document records what each one costs this port, what the port does today instead, and what
would change if upstream publishes.

It is a record of consequences, **not a plan to implement any of them**. Nothing here is scheduled.

## 1. `H3-Context-IR` — hosted prompt understanding

**Withheld.** It sits in front of the model in MiniMax's own stack, so prompt adherence from the open
weights differs from the hosted product regardless of how faithful the port is.

**What the port does.** Nothing: neither crate rewrites, refines or expands a prompt. The
conditioning path tokenizes the caller's text and stops — see
`text_encoder/tokenizer.rs`, whose module docs record that the official conditioner builds every
presentation as `tokenizer(text, add_special_tokens=False)` with no chat template
(`APPLIES_CHAT_TEMPLATE`).

**The decision.** The engine does not inject prompt rewriting, and must not: a silent rewrite would
make the rendered prompt differ from the one the caller wrote, with no seam where a user could see
or disable it. If prompt refinement is ever offered for this family it belongs above the engine, as
a visible control on an already-refined prompt string.

**If upstream publishes it.** It would be a separate component with its own weights, its own licence
read and its own manifest entry — not a change to these crates' conditioning path.

## 2. `H3-Regenerate-2K` — in-context 2K upsampling

**Withheld.** "Up to 2K" is a property of the hosted product, not of these weights.

**Upstream commitment.** MiniMax staff publicly committed, in discussion #39 on `MiniMaxAI/MiniMax-H3`
(undated), to open-source `H3-Regenerate-2K` "once this set of technologies becomes stable" — no date
given, nothing published as of the 2026-08-17 re-check.

**What the port does.** It *enforces* a canvas envelope rather than merely advertising one, in two
separate places that are not the same constraint. The per-edge bound is `Capabilities::max_size`,
checked at the shared capability floor in `gen-core/src/generator.rs`. The area bound is
`CANVAS_MAX_PIXELS` in each crate's `pipeline.rs`, checked as a *product* by `resolve_geometry` —
whose own comment records why the two are distinct: a square inside the per-edge cap can still be
far over the area the model generates at. Either check refuses; an over-size request is never
silently refitted. No number is quoted here, because sc-17152 is moving the per-edge ceiling — read
the constants.

**The decision.** The advertised ceiling is the one the weights generate at. The engine exposes no
2K path and claims none. Whether the product offers a *post-hoc upscale* (SeedVR2 / real-esrgan /
aura-sr are already separate providers) is a catalog-side choice about a separate model — it would
be an upscale of a render made inside the envelope above, and must be labelled as one rather than as
native 2K.

**If upstream publishes it.** `check_cross_backend_geometry` in `scripts/check-workspace.py`
compares every visibility-carrying `const` under each crate's `src/` by value, so a bound that lives
in a named constant cannot be raised on one backend and forgotten on the other. A bound written as a
bare literal in a struct initializer is outside that comparison.

## 3. Sparse-attention inference

**Withheld.** The model is *trained* with sparse attention; the inference implementation is not
released.

**What the port does.** Dense attention over one packed text + audio + video sequence. The cost this
imposes is already worked out and recorded in `mlx-gen-minimax-h3/src/cost.rs`, whose module docs
give the two separate quantities — `DitSequenceCost::widest_materialized_elements` (what the forward
actually writes) and `DitSequenceCost::dense_score_elements` (what a materializing attention *would*
build) — and warn against conflating them.

**The consequence.** The duration envelope is the lattice, not a policy: `LEGAL_FRAME_COUNTS` holds
the fourteen `17n + 5` counts from 124 to 345 frames, and `MIN_DURATION_SECONDS` /
`MAX_DURATION_SECONDS` are *derived* from it (5.1667 s to 14.375 s at 24 fps) rather than declared
beside it. A request for the reference's advertised flat 15.0 s has no lattice point and is refused.

**If upstream publishes it.** The bound worth revisiting is `MAX_WRITABLE_ELEMS` and the frame counts
`largest_writable_frame_count` reports, not the lattice: `17n + 5` comes from the VAE's
`clip_length`/`tokens_chunk_size`, which sparse attention does not touch.

## 4. `<d>` and the six other declared-but-untrained special tokens

**Not withheld so much as empty.** `MINIMAX_ADDED_SPECIALS` — `<d>`, `</d>`, `<|cutoff|>`,
`<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>`, `<|caption_end|>` — are declared only as
strings in `tokenizer_config.json`'s `additional_special_tokens`. The model card's
`<d>[English] …</d>` prompt syntax implies the model renders marked dialogue as speech; the open
weights contain no trained rows for those ids.

**What the port does.** It tokenizes them *correctly* and claims nothing about their meaning. Both
`text_encoder/tokenizer.rs` module docs carry the evidence (embedding rows 151669–151675 are
statistically indistinguishable from the untrained padding tail) and state that the card's syntax is
consumed by the withheld `H3-Context-IR`, not by these weights. Registering the specials is still
load-bearing for a different reason recorded there: a bare `tokenizer.json` BPE-splits `<d>` into
two ids, so a policy chosen later is implemented against correct ids either way.

**The decision.** Pass them through unchanged. Stripping or rewriting a user's prompt inside the
engine is the same silent-rewrite objection as §1, and warning on them is a UI concern, not an
engine one.

## What this repository does not decide

Two sc-17162 acceptance items are catalog copy and product framing, and live in the SceneWorks
repository rather than here:

- the manifest `ui.description` and prompt-guide wording for this family — it must not imply 2K or
  hosted-grade prompt handling, and must not advertise `<d>` as a way to direct spoken dialogue;
- whether a 2K *upscale* path is offered alongside this model, and whether prompt refinement is
  offered as a visible, opt-in control.

The engine facts those decisions need are the four sections above: an enforced canvas envelope —
per-edge and area, both refusing rather than refitting — no prompt rewriting, a lattice-derived
5.1667–14.375 s envelope, and seven tokenizable but semantically inert markers.
