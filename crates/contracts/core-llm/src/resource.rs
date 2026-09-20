//! Request-scoped memory admission for local LLM providers.

use crate::{Error, Result};

/// Deterministic operational override used by CI and by discrete-device launchers.
pub const AVAILABLE_MEMORY_OVERRIDE: &str = "SCENEWORKS_LLM_AVAILABLE_MEMORY_BYTES";

/// Loaded decoder geometry used to price KV and eager-attention workspaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LlmMemoryGeometry {
    pub query_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub layers: u64,
    pub element_bytes: u64,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub vocab_size: u64,
    pub recurrent_bytes: u64,
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
        .checked_mul(geometry.element_bytes)?
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
pub fn estimate_chunked_request_bytes(
    prompt_tokens: usize,
    max_new_tokens: u32,
    geometry: LlmMemoryGeometry,
    vision_workspace_bytes: u64,
    mtp_width: u32,
    max_attention_query_tokens: usize,
) -> Option<u64> {
    if max_attention_query_tokens == 0 {
        return None;
    }
    let prompt = u64::try_from(prompt_tokens).ok()?;
    let total = prompt.checked_add(u64::from(max_new_tokens))?;
    let attention_rows = prompt.min(u64::try_from(max_attention_query_tokens).ok()?);
    let attention = prompt
        .checked_mul(attention_rows)?
        .checked_mul(geometry.query_heads)?
        .checked_mul(geometry.element_bytes)?
        .checked_mul(3)?;
    let kv = total
        .checked_mul(geometry.layers)?
        .checked_mul(geometry.kv_heads)?
        .checked_mul(geometry.head_dim)?
        .checked_mul(geometry.element_bytes)?
        .checked_mul(2)?;
    let mtp = if mtp_width > 0 {
        kv.checked_mul(2)?.checked_add(
            u64::from(mtp_width)
                .checked_mul(geometry.hidden_size.checked_add(geometry.vocab_size)?)?
                .checked_mul(geometry.element_bytes)?,
        )?
    } else {
        0
    };
    // One decoder layer's live projections, MLP tensors, and residuals. The vocabulary projection
    // is one row because the backend narrows the final hidden state before applying lm_head.
    let activations = prompt
        .checked_mul(
            geometry
                .intermediate_size
                .checked_mul(3)?
                .checked_add(geometry.hidden_size.checked_mul(8)?)?,
        )?
        .checked_mul(geometry.element_bytes)?
        .checked_add(geometry.vocab_size.checked_mul(geometry.element_bytes)?)?;
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

/// Reject an estimated request before native tensor allocation.
pub fn admit_request_memory(required: u64, available: u64) -> Result<()> {
    if required > available {
        return Err(Error::InvalidRequest(format!(
            "request requires an estimated {required} bytes of native workspace but only {available} bytes are available; reduce prompt/media length or max_new_tokens"
        )));
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

    #[test]
    fn chunked_estimate_prices_runtime_tile_and_last_row_logits() {
        let geometry = LlmMemoryGeometry {
            query_heads: 40,
            kv_heads: 4,
            head_dim: 128,
            layers: 64,
            element_bytes: 4,
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
    fn checkpoint_counts_total_residency_and_peak_shard_staging_separately() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-1.safetensors"), [0u8; 7]).unwrap();
        std::fs::write(dir.path().join("model-2.safetensors"), [0u8; 11]).unwrap();
        std::fs::write(dir.path().join("config.json"), [0u8; 23]).unwrap();

        assert_eq!(checkpoint_payload_bytes(dir.path()).unwrap(), 18);
        assert_eq!(checkpoint_staging_bytes(dir.path()).unwrap(), 11);
    }
}
