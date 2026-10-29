//! Adapter key normalization onto the Wan and VACE module spellings.

use super::*;

/// Normalize a LoRA module path to the Wan checkpoint's naming (the reference
/// `_normalize_wan_lora_key`): strip a known prefix, then the `convert_wan` renames
/// `ffn.0/.2 → ffn.fc1/.fc2`, `text_embedding.0/.2 → text_embedding_0/_1`,
/// `time_embedding.0/.2 → time_embedding_0/_1`, `time_projection.1 → time_projection`,
/// `patch_embedding → patch_embedding_proj`. attn `q/k/v/o` pass through. Both the `.X.` infix and
/// the bare `…X` suffix forms are handled, as the reference does (a LoRA module stem ends at the
/// module, so the suffix forms fire).
///
/// Public because the Krea Realtime provider (`mlx-gen-krea-realtime`), whose DiT reuses this crate's
/// [`WanTransformer`](crate::WanTransformer) verbatim, normalizes Wan-family LoRA keys through it in its
/// own [`AdaptableHost`] before delegating to the inner Wan host
/// (sc-15015) — the same reference-faithful renames the in-crate LoRA install path uses.
pub fn normalize_wan_key(key: &str) -> String {
    let stripped = PREFIXES
        .iter()
        .find_map(|p| key.strip_prefix(p))
        .unwrap_or(key);
    let mut t = stripped.to_string();

    // ffn.0 → ffn.fc1, ffn.2 → ffn.fc2
    t = t
        .replace(".ffn.0.", ".ffn.fc1.")
        .replace(".ffn.2.", ".ffn.fc2.");
    if let Some(h) = t.strip_suffix(".ffn.0") {
        t = format!("{h}.ffn.fc1");
    }
    if let Some(h) = t.strip_suffix(".ffn.2") {
        t = format!("{h}.ffn.fc2");
    }

    // text_embedding.0/.2 → text_embedding_0/_1
    t = t
        .replace("text_embedding.0.", "text_embedding_0.")
        .replace("text_embedding.2.", "text_embedding_1.");
    if let Some(h) = t.strip_suffix("text_embedding.0") {
        t = format!("{h}text_embedding_0");
    }
    if let Some(h) = t.strip_suffix("text_embedding.2") {
        t = format!("{h}text_embedding_1");
    }

    // time_embedding.0/.2 → time_embedding_0/_1
    t = t
        .replace("time_embedding.0.", "time_embedding_0.")
        .replace("time_embedding.2.", "time_embedding_1.");
    if let Some(h) = t.strip_suffix("time_embedding.0") {
        t = format!("{h}time_embedding_0");
    }
    if let Some(h) = t.strip_suffix("time_embedding.2") {
        t = format!("{h}time_embedding_1");
    }

    // time_projection.1 → time_projection
    t = t.replace("time_projection.1.", "time_projection.");
    if let Some(h) = t.strip_suffix("time_projection.1") {
        t = format!("{h}time_projection");
    }

    // patch_embedding → patch_embedding_proj
    if t.contains("patch_embedding") && !t.contains("patch_embedding_proj") {
        t = t.replace("patch_embedding", "patch_embedding_proj");
    }
    t
}

/// Normalize a LoRA module path to the **diffusers** Wan-VACE checkpoint's naming (sc-3439). Unlike
/// [`normalize_wan_key`] (which targets the *native* converted Wan layout), the VACE transformer
/// ([`crate::vace::WanVaceTransformer`]) reads the diffusers tensor names directly, so the target is
/// already diffusers. This mirrors diffusers' own LoRA loader
/// (`_convert_non_diffusers_wan_lora_to_diffusers`): strip a known prefix, rename the **native** Wan
/// module spellings (which third-party trainers like musubi-tuner / diffusion-pipe emit) to their
/// diffusers equivalents, and pass an **already-diffusers** key (incl. every `vace_blocks.*` module,
/// which the diffusers converter does not touch and which only ever ships diffusers-named) through
/// unchanged. Renames (both the `.X.` infix and the bare `…X` suffix forms, as a LoRA module stem
/// ends at the module):
/// - `self_attn.{q,k,v,o}` → `attn1.{to_q,to_k,to_v,to_out.0}`
/// - `cross_attn.{q,k,v,o}` → `attn2.{to_q,to_k,to_v,to_out.0}`
/// - `ffn.0`/`ffn.2` → `ffn.net.0.proj`/`ffn.net.2`
/// - VACE block `before_proj`/`after_proj` → `proj_in`/`proj_out`
/// - `time_projection.1` → `condition_embedder.time_proj`; `head.head` → `proj_out`
/// - `text_embedding.0/.2` → `condition_embedder.text_embedder.linear_1/2`
/// - `time_embedding.0/.2` → `condition_embedder.time_embedder.linear_1/2`
///
/// The i2v `k_img`/`v_img` cross-attn factors (diffusers `add_k_proj`/`add_v_proj`) are intentionally
/// not mapped — the VACE host has no such modules, so they resolve to no weight and are surfaced
/// (skipped), never silently mis-folded.
pub(crate) fn normalize_vace_key(key: &str) -> String {
    let stripped = PREFIXES
        .iter()
        .find_map(|p| key.strip_prefix(p))
        .unwrap_or(key);
    let mut t = stripped.to_string();

    // VACE block hint projections: native before_proj/after_proj → diffusers proj_in/proj_out.
    t = t
        .replace(".before_proj.", ".proj_in.")
        .replace(".after_proj.", ".proj_out.");
    if let Some(h) = t.strip_suffix(".before_proj") {
        t = format!("{h}.proj_in");
    }
    if let Some(h) = t.strip_suffix(".after_proj") {
        t = format!("{h}.proj_out");
    }

    // Self-/cross-attn projections → attn1/attn2.{to_q,to_k,to_v,to_out.0}.
    for (src, dst) in [("self_attn", "attn1"), ("cross_attn", "attn2")] {
        for (n, d) in [
            ("q", "to_q"),
            ("k", "to_k"),
            ("v", "to_v"),
            ("o", "to_out.0"),
        ] {
            t = t.replace(&format!(".{src}.{n}."), &format!(".{dst}.{d}."));
            if let Some(h) = t.strip_suffix(&format!(".{src}.{n}")) {
                t = format!("{h}.{dst}.{d}");
            }
        }
    }

    // ffn.0/.2 → ffn.net.0.proj / ffn.net.2.
    t = t
        .replace(".ffn.0.", ".ffn.net.0.proj.")
        .replace(".ffn.2.", ".ffn.net.2.");
    if let Some(h) = t.strip_suffix(".ffn.0") {
        t = format!("{h}.ffn.net.0.proj");
    }
    if let Some(h) = t.strip_suffix(".ffn.2") {
        t = format!("{h}.ffn.net.2");
    }

    // Global modules (the diffusers converter's "Remaining" branch).
    t = t.replace("time_projection.1.", "condition_embedder.time_proj.");
    if let Some(h) = t.strip_suffix("time_projection.1") {
        t = format!("{h}condition_embedder.time_proj");
    }
    t = t.replace("head.head.", "proj_out.");
    if let Some(h) = t.strip_suffix("head.head") {
        t = format!("{h}proj_out");
    }
    t = t
        .replace(
            "text_embedding.0.",
            "condition_embedder.text_embedder.linear_1.",
        )
        .replace(
            "text_embedding.2.",
            "condition_embedder.text_embedder.linear_2.",
        );
    if let Some(h) = t.strip_suffix("text_embedding.0") {
        t = format!("{h}condition_embedder.text_embedder.linear_1");
    }
    if let Some(h) = t.strip_suffix("text_embedding.2") {
        t = format!("{h}condition_embedder.text_embedder.linear_2");
    }
    t = t
        .replace(
            "time_embedding.0.",
            "condition_embedder.time_embedder.linear_1.",
        )
        .replace(
            "time_embedding.2.",
            "condition_embedder.time_embedder.linear_2.",
        );
    if let Some(h) = t.strip_suffix("time_embedding.0") {
        t = format!("{h}condition_embedder.time_embedder.linear_1");
    }
    if let Some(h) = t.strip_suffix("time_embedding.2") {
        t = format!("{h}condition_embedder.time_embedder.linear_2");
    }
    t
}
