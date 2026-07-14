# Mochi 1 — quantization tier strategy (decision)

- **Story:** sc-11984 (epic A1 — Mochi 1 native port foundations)
- **Status:** Accepted
- **Applies to:** the native Mochi 1 (`genmo/mochi-1-preview`, Apache-2.0) provider crates
  (`mlx-gen-mochi` / `candle-gen-mochi`, provisioned by stories A2–A6).

## Decision

Ship Mochi as **pre-quantized, self-contained per-tier artifacts**, exactly as LTX and Wan do:

- Three tiers — **`q4`**, **`q8`**, **`bf16`** — each a **self-contained directory** carrying its
  own `split_model.json` (weights already at the tier's precision; no dense→tier conversion at load).
- The distribution manifest lists the three tiers as **separate `variant` downloads**.
- **`q4` is the default** variant (a video model — this follows the video q4-first default).
- The other tiers are **fetched on demand** when the user selects them; a client only ever downloads
  the one tier it runs.
- `convert.rs` (story **A6**) is the producer: it emits the `q4/`, `q8/`, and `bf16/` directories,
  each with its own `split_model.json`, from the upstream snapshot.

## Explicitly rejected

**Do not** ship a single dense bundle plus **on-the-fly quantization** at load time. That would force
every client to download full-precision weights (Mochi is a ~10B DiT + T5-XXL ≈ 11B text encoder +
VAE — `size_class = "very-large"`), then pay a conversion cost on every cold load, and it makes the
selected precision a runtime side effect rather than an explicit, downloadable artifact.

## Rationale

- **Download only the tier you pick.** Pre-quantized per-tier dirs mean a q4 user never pulls the
  bf16 shards. For a very-large video model this is the dominant cost, so it dominates the decision.
- **Precision is a deliberate, self-describing artifact**, not an on-the-fly transform — the tier a
  client runs is exactly the tier it downloaded, and the quant tier stays a user-visible creative
  choice (never silently switched under the user).
- **Consistency with the shipped video quant matrices.** This mirrors the LTX tier layout and the
  Wan quant matrices (sc-9941 TI2V-5B, sc-9942 T2V-A14B, sc-9943 I2V-A14B): each tier is a
  standalone directory with its own `split_model.json`, surfaced as a distinct manifest variant.
- **All models get all three tiers.** Mochi ships q4/q8/bf16 like the rest of the catalog; the
  default is q4, with q8/bf16 available on demand.

## Consumed by / references

- Pinned upstream weights + CI snapshot pin:
  [`release/real-weight-models.toml`](../../release/real-weight-models.toml) (`key = "mochi-1-preview"`,
  env `MOCHI_SNAPSHOT`, revision pinned for the parity oracle and downstream stories).
- Parity oracle that gates the tiers' correctness:
  [`crates/media/mlx-gen/tools/dump_mochi_golden.py`](../../crates/media/mlx-gen/tools/dump_mochi_golden.py)
  and the real-weights golden convention in
  [`tools/golden/README.md`](../../crates/media/mlx-gen/tools/golden/README.md).
- Tier producer: `convert.rs` (story A6) — emits the `q4/`, `q8/`, `bf16/` directories.
