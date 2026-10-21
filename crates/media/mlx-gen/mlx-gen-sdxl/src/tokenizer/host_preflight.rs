//! Requested Rust layouts for the fresh pre-key CLIP count ONLY. No window flats, MLX Arrays,
//! allocator usable size, RSS, physical lifetime, or shared runtime-cache credit is included.
//! Native facts require the separately reviewed, source-bound regex layout helper and its checks.
use mlx_gen::gen_core::PinnedWeightsFile;
use mlx_gen::{Error, Result};
use std::alloc::Layout;
use std::io::Read;

pub(super) struct RegexSourceFacts {
    constructor_new_peak: Option<u64>,
    first_search_new_peak: Option<u64>,
}
impl RegexSourceFacts {
    pub(super) fn constructor(&self) -> Result<u64> {
        self.constructor_new_peak
            .ok_or_else(|| unclosed("fixed CLIP regex constructor requested-layout certificate"))
    }
    pub(super) fn search(&self) -> Result<u64> {
        self.first_search_new_peak
            .ok_or_else(|| unclosed("fixed CLIP first-search requested-layout certificate"))
    }
}
pub(super) fn reviewed_regex_source_facts() -> Result<RegexSourceFacts> {
    // Exact owning-type Layout queries and checked source arithmetic only: no matcher,
    // cache, files or weights are constructed. Source/native checks fail closed.
    let (constructor, search) =
        regex::clip_source_certificate::checked_requested_layouts(super::CLIP_PATTERN)
            .map_err(|_| unclosed("fixed CLIP regex source/layout certificate checks"))?;
    let facts = RegexSourceFacts {
        constructor_new_peak: Some(constructor),
        first_search_new_peak: Some(search),
    };
    facts.constructor()?;
    facts.search()?;
    Ok(facts)
}
fn unclosed(term: &str) -> Error {
    Error::Msg(format!("sdxl pre-key requested layout is unclosed: {term}"))
}
fn overflow() -> Error {
    Error::Msg("sdxl pre-key requested-layout counter overflow".into())
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn array<T>(count: u64) -> Result<u64> {
    let count = usize::try_from(count).map_err(|_| overflow())?;
    Ok(Layout::array::<T>(count).map_err(|_| overflow())?.size() as u64)
}
fn align_up(n: u64, a: u64) -> Result<u64> {
    Ok(add(n, a - 1)? & !(a - 1))
}
fn table<E>(entries: u64) -> Result<u64> {
    if entries == 0 {
        return Ok(0);
    }
    // Exact corrected std/hashbrown source authority currently covers this served target cfg.
    // Other targets remain explicit source obligations, never a guessed group width.
    if !cfg!(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )) {
        return Err(unclosed("hashbrown control-group layout for this target"));
    }
    let buckets = if entries < 4 {
        4
    } else if entries < 8 {
        8
    } else {
        mul(entries, 8)?
            .checked_div(7)
            .ok_or_else(overflow)?
            .checked_next_power_of_two()
            .ok_or_else(overflow)?
    };
    let layout = Layout::new::<E>();
    let alignment = (layout.align() as u64).max(8);
    let control = align_up(mul(buckets, layout.size() as u64)?, alignment)?;
    let requested = add(add(control, buckets)?, 8)?;
    Layout::from_size_align(
        usize::try_from(requested).map_err(|_| overflow())?,
        alignment as usize,
    )
    .map_err(|_| overflow())?;
    Ok(requested)
}
/// All NEW requested storage in the vocab phase, including raw bytes, decoded key storage,
/// reusable JSON scratch and old/new HashMap table overlap. Retained prior source objects are
/// already in the caller's current snapshot and are not added again.
pub(super) fn vocab_new_upper(bytes: u64) -> Result<u64> {
    array::<u8>(bytes)?;
    let entries = bytes.saturating_sub(1) / 5;
    let scratch = if bytes == 0 { 0 } else { mul(bytes, 2)?.max(8) };
    add(
        add(mul(bytes, 2)?, scratch)?,
        mul(table::<(String, i32)>(entries)?, 2)?,
    )
}
pub(super) fn merges_new_upper(bytes: u64) -> Result<u64> {
    array::<u8>(bytes)?;
    let lines = add(bytes, 1)?;
    // On growth old capacity < needed and new <= max(2*needed, min4).
    // Three times the final length covers old+new, not just final retained capacity.
    let line_capacity = mul(lines, 3)?.max(4);
    let ranks = bytes.min((super::MERGES_END - 1) as u64);
    add(
        add(mul(bytes, 2)?, array::<&str>(line_capacity)?)?,
        mul(table::<((String, String), usize)>(ranks)?, 2)?,
    )
}
fn id_backing(count: u64) -> Result<u64> {
    array::<i32>(mul(count, 2)?.max(4))
}
fn id_growth_peak(count: u64) -> Result<u64> {
    if count <= 4 {
        array::<i32>(4)
    } else {
        array::<i32>(mul(count, 3)?)
    }
}
fn lowered_len(text: &str) -> Result<u64> {
    // str's contextual final sigma changes the codepoint, not this UTF8 byte count. No String
    // normalization is substituted for the tokenizer's actual str::to_lowercase operation.
    text.chars()
        .flat_map(char::to_lowercase)
        .try_fold(0, |sum, c| add(sum, c.len_utf8() as u64))
}
fn lowered_backing(raw: u64, lowered: u64) -> Result<u64> {
    if raw == 0 {
        return Ok(0);
    }
    let capacity = raw.max(mul(lowered, 2)?).max(8);
    array::<u8>(capacity)?;
    Ok(capacity)
}
fn row_new_upper(raw: u64, lowered: u64) -> Result<u64> {
    let ids = id_growth_peak(add(lowered, 2)?)?;
    if lowered == 0 {
        return id_backing(2);
    }
    // v6 retained backing plus explicit old/new source-request overlap during expansion.
    // Contracting/equal output never needs to grow the initial exact raw-B reservation.
    let retained_lower = lowered_backing(raw, lowered)?;
    let lower_capacity = if lowered <= raw {
        raw
    } else {
        add(lowered, retained_lower)?
    };
    let strings = add(array::<String>(lowered)?, add(mul(lowered, 12)?, 8)?)?;
    let bpe = add(
        add(mul(strings, 2)?, mul(add(mul(lowered, 2)?, 4)?, 2)?)?,
        add(
            mul(lowered, 2)?.max(8),
            array::<char>(mul(lowered, 2)?.max(4))?,
        )?,
    )?;
    add(add(lower_capacity, lowered)?, add(ids, bpe)?)
}
pub(super) fn prompt_new_upper(
    prompt: &str,
    negative: Option<&str>,
    regex: &RegexSourceFacts,
) -> Result<u64> {
    let positive = lowered_len(prompt)?;
    let positive_ids = id_backing(add(positive, 2)?)?;
    let positive_work = row_new_upper(prompt.len() as u64, positive)?;
    let (work, retained_ids, outer) = if let Some(negative) = negative {
        let lowered = lowered_len(negative)?;
        let negative_ids = id_backing(add(lowered, 2)?)?;
        (
            positive_work.max(add(
                positive_ids,
                row_new_upper(negative.len() as u64, lowered)?,
            )?),
            add(positive_ids, negative_ids)?,
            array::<Vec<i32>>(5)?,
        )
    } else {
        (positive_work, positive_ids, array::<Vec<i32>>(1)?)
    };
    // ChunkPlan probes tokenize("") AFTER both ID rows. No W-aligned row/flat buffer is built
    // by common_chunks. One sequential search cache is shared across positive/negative/probe.
    let probe_peak = add(retained_ids, id_backing(2)?)?;
    add(add(work.max(probe_peak), outer)?, regex.search()?)
}

/// Enforce the captured B before heap growth, rather than discovering a growth race only in the
/// pin's post-read check. B is the asset's own pinned size, not an arbitrary file/prompt cap.
pub(super) fn read_pinned_utf8(
    pin: &PinnedWeightsFile,
    label: &str,
    next: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<String> {
    mlx_gen::ios_pricing::with_core_source_capacity(next, |core_next| {
        pin.read_unchanged_with_capacity(core_next, |path, _same_next| -> Result<String> {
            let mut file = std::fs::File::open(path)
                .map_err(|e| Error::Msg(format!("sdxl tokenizer: read {label}: {e}")))?;
            let bound = usize::try_from(pin.target_fingerprint().size).map_err(|_| overflow())?;
            read_limited_utf8(&mut file, bound, label)
        })
    })
}
fn read_limited_utf8(read: &mut impl Read, bound: usize, label: &str) -> Result<String> {
    Layout::array::<u8>(bound).map_err(|_| overflow())?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(bound)
        .map_err(|_| Error::Msg(format!("sdxl tokenizer: read {label}: reservation failed")))?;
    let mut chunk = [0u8; 8192];
    loop {
        let remaining = bound - bytes.len();
        let count = if remaining == 0 {
            // Probe on the stack: the captured-size heap reservation never grows to B+1.
            let mut extra = [0u8; 1];
            match read.read(&mut extra) {
                Ok(0) => break,
                Ok(_) => {
                    return Err(Error::Msg(format!(
                        "sdxl tokenizer: read {label}: pinned source grew beyond captured size"
                    )))
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(Error::Msg(format!("sdxl tokenizer: read {label}: {e}"))),
            }
        } else {
            let limit = remaining.min(chunk.len());
            match read.read(&mut chunk[..limit]) {
                Ok(count) => count,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(Error::Msg(format!("sdxl tokenizer: read {label}: {e}"))),
            }
        };
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    String::from_utf8(bytes).map_err(|_| {
        Error::Msg(format!(
            "sdxl tokenizer: read {label}: stream did not contain valid UTF-8"
        ))
    })
}

/// A missing member is diagnosed only when its original lazy read is reached. A newly appeared
/// path is a changed source, never permission to read an unbounded object outside the inventory.
pub(super) fn missing_member(label: &str) -> Error {
    Error::Msg(format!(
        "sdxl tokenizer: read {label}: member absent from retained source inventory"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lowercase_count_layout_and_unclosed_regex_are_not_zero_host() {
        let text = "K".repeat(16);
        assert_eq!(text.len(), 48);
        assert_eq!(lowered_len(&text).unwrap(), 16);
        assert_eq!(lowered_backing(48, 16).unwrap(), 48);
        assert_eq!(lowered_backing(0, 0).unwrap(), 0);
        assert!(lowered_backing(1, u64::MAX).is_err());
        assert!(id_backing(u64::MAX).is_err());
        let unknown = RegexSourceFacts {
            constructor_new_peak: None,
            first_search_new_peak: None,
        };
        assert!(unknown
            .constructor()
            .err()
            .expect("explicit unknown source term")
            .to_string()
            .contains("unclosed"));
        assert!(unknown.search().is_err());
        // Explicit finite arithmetic fixture, NOT an actual regex certificate or admission value.
        let fixture = RegexSourceFacts {
            constructor_new_peak: Some(64),
            first_search_new_peak: Some(128),
        };
        assert!(prompt_new_upper("", None, &fixture).unwrap() >= 16 + 128);
        assert!(
            prompt_new_upper("abc", Some("negative"), &fixture).unwrap()
                > prompt_new_upper("abc", None, &fixture).unwrap()
        );
    }
    #[test]
    fn serial_positive_negative_and_probe_charge_one_search_high_water_host() {
        // Explicit finite composition fixtures, NOT native regex source facts.
        let base = RegexSourceFacts {
            constructor_new_peak: Some(64),
            first_search_new_peak: Some(0),
        };
        let cache = RegexSourceFacts {
            constructor_new_peak: Some(64),
            first_search_new_peak: Some(97),
        };
        for (positive, negative) in [
            ("", None),
            ("abc", None),
            ("abc", Some("")),
            ("abc", Some("real optional CFG negative")),
            ("Kİ", Some("σ")),
        ] {
            assert_eq!(
                prompt_new_upper(positive, negative, &cache).unwrap()
                    - prompt_new_upper(positive, negative, &base).unwrap(),
                97
            );
        }
    }
    #[test]
    fn checked_native_fact_hook_keeps_both_terms_or_an_explicit_failure_host() {
        match reviewed_regex_source_facts() {
            Ok(facts) => {
                assert!(facts.constructor().unwrap() > 1_754_183);
                assert!(facts.search().unwrap() > 20_018_036);
            }
            Err(error) => assert!(error.to_string().contains("unclosed")),
        }
    }
    #[test]
    fn bounded_reader_enforces_captured_size_without_heap_overflow_probe_host() {
        use std::io::Cursor;
        let text = read_limited_utf8(&mut Cursor::new(b"abc"), 3, "vocab.json").unwrap();
        assert_eq!(text, "abc");
        assert_eq!(
            text.capacity(),
            3,
            "exact requested Vec reservation, not malloc usable-size proof"
        );
        assert!(
            read_limited_utf8(&mut Cursor::new(b"abcd"), 3, "vocab.json")
                .err()
                .unwrap()
                .to_string()
                .contains("grew beyond captured size")
        );
        assert_eq!(
            read_limited_utf8(&mut Cursor::new(b""), 0, "merges.txt").unwrap(),
            ""
        );
        assert!(read_limited_utf8(&mut Cursor::new(b"x"), 0, "merges.txt").is_err());
        assert!(
            read_limited_utf8(&mut Cursor::new(&[255u8][..]), 1, "vocab.json")
                .err()
                .unwrap()
                .to_string()
                .contains("valid UTF-8")
        );
    }
}

#[cfg(test)]
pub(super) fn finite_test_fixture() -> RegexSourceFacts {
    // Arbitrary finite arithmetic fixture ONLY; not native regex certificate values.
    RegexSourceFacts {
        constructor_new_peak: Some(64),
        first_search_new_peak: Some(128),
    }
}

#[cfg(test)]
pub(super) fn native_pin_reader_test_fixture(
    pin: &PinnedWeightsFile,
    label: &str,
    _next: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<String> {
    // ONLY exercises tokenizer/error/window semantics under labeled finite arithmetic fixtures.
    // This is the original native metadata guard, NOT a guarded filesystem source certificate.
    pin.read_unchanged(|path| {
        let mut file = std::fs::File::open(path)?;
        let bound = usize::try_from(pin.target_fingerprint().size).map_err(|_| overflow())?;
        read_limited_utf8(&mut file, bound, label)
    })
}
