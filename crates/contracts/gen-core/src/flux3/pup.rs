//! Prompt-upsampler (PUP) constants the request surface needs. The upsampler itself lives in
//! `gen_core_flux3::pup` (sc-27316).

/// `pup.SAMPLING["max_tokens"]` — the decode budget.
pub const MAX_TOKENS: u32 = 3584;
/// `re.fullmatch(r"ref_image_\d+|target", origin)`. Upstream's `\d` is Unicode-aware; this
/// accepts ASCII digits only, a difference no real answer exercises.
pub fn is_row_origin(s: &str) -> bool {
    s == "target"
        || s.strip_prefix("ref_image_")
            .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}
