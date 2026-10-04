//! Request-scoped memory admission for local LLM providers.

use crate::{Error, RequestResourceExhausted, Result};

/// Deterministic operational override used by CI and by discrete-device launchers.
pub const AVAILABLE_MEMORY_OVERRIDE: &str = "SCENEWORKS_LLM_AVAILABLE_MEMORY_BYTES";

/// Loaded decoder geometry used to price KV and eager-attention workspaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LlmMemoryGeometry {
    pub query_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub layers: u64,
    /// Scalar width of the K/V cache, decoder activations, and logits: the decoder's compute
    /// dtype (2 for a BF16 decoder, 4 for an F32 one).
    pub element_bytes: u64,
    /// Scalar width of the attention scores, additive mask, and softmax weights the attention
    /// workspace term prices. A backend whose score path upcasts (MLX's eager SDPA computes scores
    /// in F32 for a BF16 decoder) declares that width here rather than widening every other term.
    pub score_element_bytes: u64,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub vocab_size: u64,
    pub recurrent_bytes: u64,
}

impl LlmMemoryGeometry {
    /// The K/V cache geometry these terms price.
    pub fn kv_shape(&self) -> crate::KvCacheShape {
        crate::KvCacheShape {
            layers: self.layers,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            element_bytes: self.element_bytes,
        }
    }
}

/// Conservative checked estimate for request-owned native memory.
pub fn estimate_request_bytes(
    prompt_tokens: usize,
    max_new_tokens: u32,
    geometry: LlmMemoryGeometry,
    vision_workspace_bytes: u64,
    mtp_width: u32,
) -> Option<u64> {
    let prompt = u64::try_from(prompt_tokens).ok()?;
    let total = prompt.checked_add(u64::from(max_new_tokens))?;
    // Eager prefill materializes scores, the additive mask, and softmax weights.
    let attention = prompt
        .checked_mul(prompt)?
        .checked_mul(geometry.query_heads)?
        .checked_mul(geometry.score_element_bytes)?
        .checked_mul(3)?;
    // K and V caches for every layer through the requested terminal position.
    let kv = total
        .checked_mul(geometry.layers)?
        .checked_mul(geometry.kv_heads)?
        .checked_mul(geometry.head_dim)?
        .checked_mul(geometry.element_bytes)?
        .checked_mul(2)?;
    // Include predictor cache, draft verification positions, and clone/replay rollback state.
    let mtp = if mtp_width > 0 {
        kv.checked_mul(2)?.checked_add(
            u64::from(mtp_width)
                .checked_mul(geometry.hidden_size.checked_add(geometry.vocab_size)?)?
                .checked_mul(geometry.element_bytes)?,
        )?
    } else {
        0
    };
    // Projection/MLP intermediates and logits are materialized during eager prefill.
    let activations = prompt
        .checked_mul(
            geometry
                .intermediate_size
                .checked_mul(3)?
                .checked_add(geometry.hidden_size.checked_mul(8)?)?
                .checked_add(geometry.vocab_size)?,
        )?
        .checked_mul(geometry.element_bytes)?;
    attention
        .checked_add(kv)?
        .checked_add(mtp)?
        .checked_add(vision_workspace_bytes)?
        .checked_add(activations)?
        .checked_add(
            geometry
                .recurrent_bytes
                .checked_mul(if mtp_width > 0 { 3 } else { 1 })?,
        )
}

/// Conservative checked estimate for a request whose attention implementation bounds the number
/// of simultaneously materialized query rows and whose prefill projects only the final hidden row
/// to vocabulary logits.
///
/// `max_attention_query_tokens` must be the same non-zero tile bound enforced by the backend's
/// attention runtime. KV, speculative rollback, media, decoder activation, and recurrent-state
/// costs remain fully priced; only the two prompt-scaled tensors proven absent from that runtime
/// path differ from [`estimate_request_bytes`].
///
/// With `mtp_width > 0` the recurrent-state term is charged three times — the live state plus the
/// copies a clone/replay MTP loop holds while it verifies and restores. A backend whose
/// `geometry.recurrent_bytes` already prices every rollback copy its cache holds uses
/// [`estimate_chunked_request_bytes_with_recurrent_copies`] instead.
pub fn estimate_chunked_request_bytes(
    prompt_tokens: usize,
    max_new_tokens: u32,
    geometry: LlmMemoryGeometry,
    vision_workspace_bytes: u64,
    mtp_width: u32,
    max_attention_query_tokens: usize,
) -> Option<u64> {
    estimate_chunked_request_bytes_with_recurrent_copies(
        prompt_tokens,
        max_new_tokens,
        geometry,
        vision_workspace_bytes,
        mtp_width,
        max_attention_query_tokens,
        if mtp_width > 0 { 3 } else { 1 },
    )
}

/// [`estimate_chunked_request_bytes`] with the recurrent-state multiplier chosen by the caller:
/// `geometry.recurrent_bytes` is charged exactly `recurrent_copies` times, whatever `mtp_width`
/// is. Every other term is identical.
///
/// Pass `1` when `geometry.recurrent_bytes` is already the whole recurrent footprint of the
/// request's cache — e.g. a cache that rolls back by selecting a slot of a preallocated
/// per-token checkpoint ring priced into the geometry, and never clones. A clone/replay MTP loop
/// keeps the `3` [`estimate_chunked_request_bytes`] applies. `recurrent_copies == 0` is refused
/// (`None`): it would drop the recurrent state from admission altogether.
pub fn estimate_chunked_request_bytes_with_recurrent_copies(
    prompt_tokens: usize,
    max_new_tokens: u32,
    geometry: LlmMemoryGeometry,
    vision_workspace_bytes: u64,
    mtp_width: u32,
    max_attention_query_tokens: usize,
    recurrent_copies: u64,
) -> Option<u64> {
    if max_attention_query_tokens == 0 {
        return None;
    }
    let prompt = u64::try_from(prompt_tokens).ok()?;
    let attention_rows = prompt.min(u64::try_from(max_attention_query_tokens).ok()?);
    let attention = prompt
        .checked_mul(attention_rows)?
        .checked_mul(geometry.query_heads)?
        .checked_mul(geometry.score_element_bytes)?
        .checked_mul(3)?;
    estimate_tiled_request_bytes_with_recurrent_copies(
        prompt_tokens,
        max_new_tokens,
        geometry,
        vision_workspace_bytes,
        mtp_width,
        attention,
        recurrent_copies,
    )
}

/// [`estimate_chunked_request_bytes_with_recurrent_copies`] with the prompt-scaled attention
/// workspace supplied by the backend instead of derived from a row bound: for an attention runtime
/// whose per-tile transient is not a per-head score/mask/softmax set (e.g. a fused kernel that
/// keeps scores on chip and only slices an explicit mask per tile). Every other term is identical;
/// `recurrent_copies == 0` is refused (`None`).
pub fn estimate_tiled_request_bytes_with_recurrent_copies(
    prompt_tokens: usize,
    max_new_tokens: u32,
    geometry: LlmMemoryGeometry,
    vision_workspace_bytes: u64,
    mtp_width: u32,
    attention_workspace_bytes: u64,
    recurrent_copies: u64,
) -> Option<u64> {
    let prompt = u64::try_from(prompt_tokens).ok()?;
    let total = prompt.checked_add(u64::from(max_new_tokens))?;
    let kv = geometry.kv_shape().dense_bytes(total)?;
    estimate_tiled_request_bytes_with_kv_bytes(
        prompt_tokens,
        geometry,
        vision_workspace_bytes,
        mtp_width,
        attention_workspace_bytes,
        recurrent_copies,
        kv,
    )
}

/// [`estimate_tiled_request_bytes_with_recurrent_copies`] with the request's K/V cache bytes
/// supplied by the backend instead of priced dense: for a request the backend serves from a
/// compressed cache ([`crate::compressed_kv_cache_bytes`]). `kv_cache_bytes` must cover every
/// position through the requested terminal one; every other term is identical, and
/// `recurrent_copies == 0` is refused (`None`).
pub fn estimate_tiled_request_bytes_with_kv_bytes(
    prompt_tokens: usize,
    geometry: LlmMemoryGeometry,
    vision_workspace_bytes: u64,
    mtp_width: u32,
    attention_workspace_bytes: u64,
    recurrent_copies: u64,
    kv_cache_bytes: u64,
) -> Option<u64> {
    if recurrent_copies == 0 {
        return None;
    }
    let attention = attention_workspace_bytes;
    let prompt = u64::try_from(prompt_tokens).ok()?;
    let kv = kv_cache_bytes;
    let mtp = if mtp_width > 0 {
        kv.checked_mul(2)?.checked_add(
            u64::from(mtp_width)
                .checked_mul(geometry.hidden_size.checked_add(geometry.vocab_size)?)?
                .checked_mul(geometry.element_bytes)?,
        )?
    } else {
        0
    };
    let activations = tiled_prefill_activation_bytes(prompt, geometry)?;
    attention
        .checked_add(kv)?
        .checked_add(mtp)?
        .checked_add(vision_workspace_bytes)?
        .checked_add(activations)?
        .checked_add(geometry.recurrent_bytes.checked_mul(recurrent_copies)?)
}

/// Decoder activations a tiled prefill of `prompt_tokens` holds at once: one decoder layer's live
/// projections, MLP tensors, and residuals across the whole prompt, plus one row of vocabulary
/// logits (the backend narrows the final hidden state before applying lm_head). This is the
/// activation term of [`estimate_tiled_request_bytes_with_recurrent_copies`], exposed so a caller
/// pricing a prompt's prefill outside a request admission prices it identically.
pub fn tiled_prefill_activation_bytes(
    prompt_tokens: u64,
    geometry: LlmMemoryGeometry,
) -> Option<u64> {
    prompt_tokens
        .checked_mul(
            geometry
                .intermediate_size
                .checked_mul(3)?
                .checked_add(geometry.hidden_size.checked_mul(8)?)?,
        )?
        .checked_mul(geometry.element_bytes)?
        .checked_add(geometry.vocab_size.checked_mul(geometry.element_bytes)?)
}

/// Reject an estimated request before native tensor allocation.
pub fn admit_request_memory(required: u64, available: u64) -> Result<()> {
    if required > available {
        return Err(Error::InvalidRequest(format!(
            "request requires an estimated {required} bytes of native workspace but only {available} bytes are available; reduce prompt/media length or max_new_tokens"
        )));
    }
    Ok(())
}

/// Reject an architecturally valid generation request with typed preallocation evidence.
pub fn admit_request_memory_with_geometry(
    prompt_tokens: usize,
    max_new_tokens: u32,
    max_context_tokens: usize,
    required_bytes: u64,
    available_bytes: u64,
) -> Result<()> {
    if required_bytes > available_bytes {
        return Err(Error::RequestResourceExhausted(RequestResourceExhausted {
            prompt_tokens,
            max_new_tokens,
            max_context_tokens,
            required_bytes,
            available_bytes,
        }));
    }
    Ok(())
}

/// An operational budget caps measured availability; it never substitutes for capacity.
pub fn operational_memory_override() -> Result<Option<u64>> {
    match std::env::var(AVAILABLE_MEMORY_OVERRIDE) {
        Ok(value) => parse_memory_budget(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(Error::InvalidRequest(
            "memory budget is not valid Unicode".into(),
        )),
    }
}

fn parse_memory_budget(value: Option<&str>) -> Result<Option<u64>> {
    value
        .map(|value| {
            value.parse::<u64>().map_err(|_| {
                Error::InvalidRequest(format!(
                    "{AVAILABLE_MEMORY_OVERRIDE} must be an unsigned byte count"
                ))
            })
        })
        .transpose()
}

/// Fail closed when capacity is unknown, and cap a valid budget by current capacity.
pub fn effective_memory_budget(capacity: Option<u64>, budget: Option<u64>) -> Result<u64> {
    let capacity = capacity.ok_or_else(|| {
        Error::InvalidRequest(
            "request admission could not read current device/host available memory".into(),
        )
    })?;
    Ok(budget.map_or(capacity, |budget| capacity.min(budget)))
}

/// Fresh host/unified-memory availability snapshot. This is never used as CUDA capacity.
pub fn available_host_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kib = text.lines().find_map(|line| {
            line.strip_prefix("MemAvailable:")?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
        })?;
        return kib.checked_mul(1024);
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("vm_stat").output().ok()?;
        let text = String::from_utf8(output.stdout).ok()?;
        let page_size = text
            .lines()
            .next()?
            .split("page size of ")
            .nth(1)?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()?;
        let pages = text
            .lines()
            .skip(1)
            .filter_map(|line| {
                let (name, value) = line.split_once(':')?;
                matches!(name, "Pages free" | "Pages inactive" | "Pages speculative")
                    .then(|| value.trim().trim_end_matches('.').parse::<u64>().ok())?
            })
            .sum::<u64>();
        return pages.checked_mul(page_size);
    }
    #[cfg(target_os = "windows")]
    {
        let output = std::process::Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
            ])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        return String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()?
            .checked_mul(1024);
    }
    #[allow(unreachable_code)]
    None
}

/// Sum only checkpoint payload files without reading weights into memory.
pub fn checkpoint_payload_bytes(source: &std::path::Path) -> Result<u64> {
    let io_error = |error: std::io::Error| Error::Load(format!("checkpoint admission: {error}"));
    if source.is_file() {
        return source.metadata().map(|m| m.len()).map_err(io_error);
    }
    let mut total = 0u64;
    for entry in std::fs::read_dir(source).map_err(io_error)? {
        let path = entry.map_err(io_error)?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            total = total
                .checked_add(path.metadata().map_err(io_error)?.len())
                .ok_or_else(|| Error::Load("checkpoint byte count overflow".into()))?;
        }
    }
    Ok(total)
}

/// Largest checkpoint payload file loaded into a host buffer at one time.
///
/// Candle's safetensors loader processes a sharded directory sequentially: `std::fs::read` owns one
/// shard buffer, its tensors are copied to the target device, and that buffer is dropped before the
/// next shard. A single-file checkpoint has that file's full size as its staging requirement.
pub fn checkpoint_staging_bytes(source: &std::path::Path) -> Result<u64> {
    let io_error = |error: std::io::Error| Error::Load(format!("checkpoint admission: {error}"));
    if source.is_file() {
        return source.metadata().map(|m| m.len()).map_err(io_error);
    }
    let mut largest = 0u64;
    for entry in std::fs::read_dir(source).map_err(io_error)? {
        let path = entry.map_err(io_error)?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            largest = largest.max(path.metadata().map_err(io_error)?.len());
        }
    }
    Ok(largest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_is_checked_and_prices_quadratic_prefill() {
        let geometry = LlmMemoryGeometry {
            query_heads: 24,
            kv_heads: 4,
            head_dim: 128,
            layers: 40,
            element_bytes: 4,
            score_element_bytes: 4,
            hidden_size: 5120,
            intermediate_size: 17408,
            vocab_size: 248320,
            recurrent_bytes: 1024,
        };
        let short = estimate_request_bytes(1024, 16, geometry, 0, 0).unwrap();
        let long = estimate_request_bytes(131_072, 16, geometry, 0, 0).unwrap();
        assert!(long > short * 1_000);
        assert!(
            admit_request_memory(long, 102_171_148_288).is_err(),
            "architecturally valid long context must still fail closed when current capacity is insufficient"
        );
        assert!(estimate_request_bytes(usize::MAX, u32::MAX, geometry, 0, 3).is_none());
    }

    /// The tiled estimate is the chunked one with the attention term replaced by the caller's bytes:
    /// equal at the chunked term, and it moves one-for-one with the supplied workspace.
    #[test]
    fn tiled_estimate_is_the_chunked_estimate_with_a_supplied_attention_term() {
        let g = LlmMemoryGeometry {
            query_heads: 24,
            kv_heads: 8,
            head_dim: 128,
            layers: 28,
            element_bytes: 2,
            score_element_bytes: 4,
            hidden_size: 3072,
            intermediate_size: 8192,
            vocab_size: 128_256,
            recurrent_bytes: 0,
        };
        let (prompt, rows) = (4096u64, 8u64);
        let chunked_term = prompt * rows * g.query_heads * g.score_element_bytes * 3;
        let chunked =
            estimate_chunked_request_bytes_with_recurrent_copies(4096, 16, g, 0, 0, 8, 1).unwrap();
        let tiled = |attention| {
            estimate_tiled_request_bytes_with_recurrent_copies(4096, 16, g, 0, 0, attention, 1)
                .unwrap()
        };
        assert_eq!(tiled(chunked_term), chunked);
        assert_eq!(tiled(chunked_term + 12_345), chunked + 12_345);
        assert_eq!(
            estimate_tiled_request_bytes_with_recurrent_copies(4096, 16, g, 0, 0, 0, 0),
            None
        );
    }

    /// The supplied-KV estimator differs from the dense one only in the K/V term: dense pricing is
    /// `(prompt + max_new) · layers · kv_heads · head_dim · element_bytes · 2`.
    #[test]
    fn supplied_kv_bytes_replace_only_the_dense_kv_term() {
        let g = LlmMemoryGeometry {
            query_heads: 24,
            kv_heads: 8,
            head_dim: 128,
            layers: 28,
            element_bytes: 2,
            score_element_bytes: 4,
            hidden_size: 3072,
            intermediate_size: 8192,
            vocab_size: 128_256,
            recurrent_bytes: 0,
        };
        let dense_kv = (4096 + 16) * 28 * 8 * 128 * 2 * 2;
        assert_eq!(g.kv_shape().dense_bytes(4096 + 16), Some(dense_kv));
        let dense = estimate_tiled_request_bytes_with_recurrent_copies(4096, 16, g, 0, 0, 77, 1);
        let supplied =
            |kv| estimate_tiled_request_bytes_with_kv_bytes(4096, g, 0, 0, 77, 1, kv).unwrap();
        assert_eq!(dense, Some(supplied(dense_kv)));
        assert_eq!(supplied(dense_kv) - supplied(1_000), dense_kv - 1_000);
        assert_eq!(
            estimate_tiled_request_bytes_with_kv_bytes(4096, g, 0, 0, 77, 0, 1),
            None
        );
    }

    #[test]
    fn chunked_estimate_prices_runtime_tile_and_last_row_logits() {
        let geometry = LlmMemoryGeometry {
            query_heads: 40,
            kv_heads: 4,
            head_dim: 128,
            layers: 64,
            element_bytes: 4,
            score_element_bytes: 4,
            hidden_size: 5120,
            intermediate_size: 17_408,
            vocab_size: 248_320,
            recurrent_bytes: 0,
        };
        let chunked = estimate_chunked_request_bytes(29_600, 128, geometry, 0, 0, 8).unwrap();
        let eager = estimate_request_bytes(29_600, 128, geometry, 0, 0).unwrap();
        assert_eq!(chunked, 18_940_659_712);
        assert!(chunked < 36_000_000_000);
        assert!(eager > 400_000_000_000);
        assert!(estimate_chunked_request_bytes(128, 16, geometry, 0, 0, 0).is_none());
        assert!(estimate_chunked_request_bytes(usize::MAX, u32::MAX, geometry, 0, 3, 8).is_none());
    }

    /// sc-20671: the compute width prices K/V, activations, logits and MTP state; only the
    /// attention-score term takes the score width. A BF16 decoder with F32 eager scores must not
    /// be charged a 4-byte K/V cache, and must still be charged 4-byte scores.
    #[test]
    fn compute_and_score_widths_price_separate_terms() {
        let geometry = LlmMemoryGeometry {
            query_heads: 8,
            kv_heads: 2,
            head_dim: 64,
            layers: 4,
            element_bytes: 2,
            score_element_bytes: 4,
            hidden_size: 256,
            intermediate_size: 512,
            vocab_size: 1024,
            recurrent_bytes: 4096,
        };
        let (prompt, new, rows, mtp) = (300_u64, 20_u64, 8_u64, 3_u64);
        let g = geometry;
        let kv = (prompt + new) * g.layers * g.kv_heads * g.head_dim * g.element_bytes * 2;
        let mtp_bytes = kv * 2 + mtp * (g.hidden_size + g.vocab_size) * g.element_bytes;
        let mlp = g.intermediate_size * 3 + g.hidden_size * 8;
        let eager = prompt * prompt * g.query_heads * g.score_element_bytes * 3
            + kv
            + mtp_bytes
            + prompt * (mlp + g.vocab_size) * g.element_bytes
            + g.recurrent_bytes * 3;
        assert_eq!(
            estimate_request_bytes(300, 20, geometry, 0, 3).unwrap(),
            eager
        );
        let chunked = prompt * rows * g.query_heads * g.score_element_bytes * 3
            + kv
            + mtp_bytes
            + prompt * mlp * g.element_bytes
            + g.vocab_size * g.element_bytes
            + g.recurrent_bytes * 3;
        assert_eq!(
            estimate_chunked_request_bytes(300, 20, geometry, 0, 3, 8).unwrap(),
            chunked
        );
    }

    #[test]
    fn recurrent_copies_scale_only_the_recurrent_term() {
        let recurrent = 7_000_003u64;
        let geometry = LlmMemoryGeometry {
            query_heads: 24,
            kv_heads: 4,
            head_dim: 128,
            layers: 64,
            element_bytes: 4,
            score_element_bytes: 4,
            hidden_size: 5120,
            intermediate_size: 17_408,
            vocab_size: 248_320,
            recurrent_bytes: recurrent,
        };
        let with = |mtp_width, copies| {
            estimate_chunked_request_bytes_with_recurrent_copies(
                1_000, 64, geometry, 0, mtp_width, 8, copies,
            )
        };
        let once = with(5, 1).unwrap();
        let thrice = with(5, 3).unwrap();
        assert_eq!(
            thrice - once,
            2 * recurrent,
            "only the recurrent term moves with the multiplier"
        );
        // The legacy entry point is the x3 clone/replay pricing with MTP and x1 without.
        assert_eq!(
            estimate_chunked_request_bytes(1_000, 64, geometry, 0, 5, 8),
            Some(thrice)
        );
        assert_eq!(
            estimate_chunked_request_bytes(1_000, 64, geometry, 0, 0, 8),
            with(0, 1)
        );
        // With x1 the MTP request differs from the MTP-off one only by the non-recurrent MTP
        // terms: the recurrent term is charged once either way.
        let no_recurrent = LlmMemoryGeometry {
            recurrent_bytes: 0,
            ..geometry
        };
        let mtp_terms = estimate_chunked_request_bytes(1_000, 64, no_recurrent, 0, 5, 8).unwrap()
            - estimate_chunked_request_bytes(1_000, 64, no_recurrent, 0, 0, 8).unwrap();
        assert_eq!(once - with(0, 1).unwrap(), mtp_terms);
        // Zero copies would drop the recurrent state from admission: refused.
        assert_eq!(with(5, 0), None);
        assert_eq!(with(5, u64::MAX / 2), None, "the multiplication is checked");
    }

    #[test]
    fn budgets_cannot_inflate_capacity_or_hide_unknown_capacity() {
        assert_eq!(effective_memory_budget(Some(100), Some(1000)).unwrap(), 100);
        assert_eq!(effective_memory_budget(Some(100), Some(1)).unwrap(), 1);
        assert!(effective_memory_budget(None, Some(1000)).is_err());
        assert!(parse_memory_budget(Some("not-bytes")).is_err());
        assert!(parse_memory_budget(Some("-1")).is_err());
        assert_eq!(parse_memory_budget(Some("0")).unwrap(), Some(0));
    }

    #[test]
    fn rejection_is_actionable_and_names_estimate() {
        let error = admit_request_memory(200, 100).unwrap_err().to_string();
        assert!(error.contains("estimated 200 bytes"));
        assert!(error.contains("only 100 bytes are available"));
        assert!(error.contains("reduce prompt/media length or max_new_tokens"));
    }

    #[test]
    fn request_rejection_retains_exact_geometry_and_compatible_display() {
        let error = admit_request_memory_with_geometry(80, 16, 128, 200, 100).unwrap_err();
        assert_eq!(
            error.to_string(),
            admit_request_memory(200, 100).unwrap_err().to_string()
        );
        let Error::RequestResourceExhausted(evidence) = error else {
            panic!("request admission must remain typed");
        };
        assert_eq!(
            evidence,
            RequestResourceExhausted {
                prompt_tokens: 80,
                max_new_tokens: 16,
                max_context_tokens: 128,
                required_bytes: 200,
                available_bytes: 100,
            }
        );
    }

    #[test]
    fn checkpoint_counts_total_residency_and_peak_shard_staging_separately() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-1.safetensors"), [0u8; 7]).unwrap();
        std::fs::write(dir.path().join("model-2.safetensors"), [0u8; 11]).unwrap();
        std::fs::write(dir.path().join("config.json"), [0u8; 23]).unwrap();

        assert_eq!(checkpoint_payload_bytes(dir.path()).unwrap(), 18);
        assert_eq!(checkpoint_staging_bytes(dir.path()).unwrap(), 11);
    }
}
