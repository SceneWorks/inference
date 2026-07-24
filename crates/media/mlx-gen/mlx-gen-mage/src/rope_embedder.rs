//! `MageFlowEmbedRope` — the 3-axis multimodal RoPE ("msrope") — **owned by sc-14040**.
//!
//! Port of `_vendor/mage_flow/models/modules/mage_layers.py:105-210`. The pinned facts (all
//! verified in `_vendor/MAGE_FLOW_GAPS.md` GAP 3, several of which correct the epic description):
//!
//! - **`theta = 10000` is hardcoded in code**, not read from the config's `"theta"` key
//!   ([`crate::config::ROPE_THETA`]); the axis split is `axes_dim = [16, 56, 56]` =
//!   `(frame, height, width)`, half-dims `[8, 28, 28]`, asserted to sum to `head_dim`.
//! - There is **no `axes_lens` field**. A fixed [`crate::config::ROPE_TABLE_LEN`]-entry positive
//!   table plus a mirrored negative table (`index.flip(0) * -1 - 1`) is precomputed per axis.
//! - `scale_rope = true`: height/width are **centred** —
//!   `cat(neg[-(L - L/2):], pos[:L/2])`, i.e. indices `-(L - L/2) … L/2 - 1` (`:194-203`).
//!   The frame axis is *not* centred: it is the segment's index in `img_shapes` (`:171`, `:192`).
//! - Coordinates come from **`img_shapes`, never `img_ids`**. `img_ids` is computed by the
//!   reference pipeline but never reaches the model — it is vestigial. **Do not port it.**
//!
//! **Trap — `batch_cfg` shifts the unconditional branch's frame index.** `_build_pack_ctx`
//! concatenates the segment list (`pipeline.py:167`), so the duplicated uncond half rotates at
//! frame index **1**, not 0. The fused CFG path is therefore *not* numerically identical to two
//! separate forwards, contrary to the reference's own docstring at `pipeline.py:136-140`
//! (measured: cond half exact identity, uncond half differs by max_abs 0.9589 confined to the
//! frame lanes). A port must replicate the shift or deliberately target the unfused trajectory and
//! pin that choice in its parity test. `tools/verify_mage_flow_golden.py` asserts this.
