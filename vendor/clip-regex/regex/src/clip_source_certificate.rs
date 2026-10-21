//! Fixed CLIP requested-layout certificate, not allocator usable size or RSS.
//! Pure type queries + checked source arithmetic: no Regex construction, warmup,
//! asset reads, cache creation, invented private repr(Rust) layouts or measurements.
//! Source derivation: whole-operation-proof-v8 SHA256
//! 4e254aaeb8c28a1ed30bf99f33657cb3a252880e7ce4299f61b30311100a0b84.
use core::alloc::Layout;
use regex_automata::{
    hybrid::LazyStateID,
    nfa::thompson,
    util::primitives::{SmallIndex, StateID},
};
use regex_syntax::{ast, hir};
use std::{
    cell::Cell,
    collections::HashMap,
    string::String,
    sync::{atomic::AtomicUsize, Arc},
    vec::Vec,
};

// Literal SHA256: a785edc83cd9dd0e08b74f1d267950fde9fd291099886c888cd15dc200454afb.
// Full raw equality below is stronger than trusting a caller-supplied digest.
const PATTERN: &str =
    r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+";
const SOURCE: &str =
    "clip-regex-source:v1:4e254aaeb8c28a1ed30bf99f33657cb3a252880e7ce4299f61b30311100a0b84";
const K: u64 = 2_097_152;

/// A failed check never becomes a zero or a partial certificate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CertificateError {
    /// The raw tokenizer pattern is not the proven literal.
    LiteralMismatch,
    /// One actual linked crate version differs.
    DependencyMismatch,
    /// Build-time source/compiler provenance is not established.
    SourceMismatch,
    /// The allocation-law target binding does not apply.
    UnsupportedTarget,
    /// Logging/instrumentation would add unproven allocations.
    UnsupportedFeatures,
    /// Checked arithmetic or native conversion failed.
    Overflow,
    /// A concrete layout or fixed-size source invariant failed.
    InvalidLayout,
    /// The nonrecursive hybrid initialization predicate failed.
    InitPredicate,
}
impl core::fmt::Display for CertificateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}
type R<T> = Result<T, CertificateError>;
fn add(a: u64, b: u64) -> R<u64> {
    a.checked_add(b).ok_or(CertificateError::Overflow)
}
fn mul(a: u64, b: u64) -> R<u64> {
    a.checked_mul(b).ok_or(CertificateError::Overflow)
}
fn native(n: u64) -> R<usize> {
    usize::try_from(n).map_err(|_| CertificateError::Overflow)
}
fn sum(terms: &[(u64, u64)]) -> R<u64> {
    terms.iter().try_fold(0, |s, &(c, v)| add(s, mul(c, v)?))
}
fn aligned(n: u64, a: u64) -> R<u64> {
    if a == 0 || !a.is_power_of_two() {
        return Err(CertificateError::InvalidLayout);
    }
    Ok(add(n, a - 1)? & !(a - 1))
}
fn request(layout: Layout, count: u64) -> R<()> {
    // Validate an upper SINGLE request, never treat the old+new SUM as an array.
    let bytes = mul(layout.size() as u64, count)?;
    Layout::from_size_align(native(bytes)?, layout.align())
        .map_err(|_| CertificateError::InvalidLayout)?;
    Ok(())
}
fn largest<const N: usize>(layouts: [Layout; N], request_max: u64) -> R<Layout> {
    let mut largest = Layout::new::<()>();
    for l in layouts {
        request(l, request_max)?;
        if l.size() > largest.size() {
            largest = l;
        }
    }
    Ok(largest)
}
fn arc_payload(layout: Layout) -> R<u64> {
    // Exact repr(C) strong/weak AtomicUsize header from the authenticated std source.
    // NOT sizeof(Arc<T>), and NOT a guessed mirrored private ArcInner<T>.
    let (extended, _) = Layout::new::<[AtomicUsize; 2]>()
        .extend(layout)
        .map_err(|_| CertificateError::InvalidLayout)?;
    Ok(extended.pad_to_align().size() as u64)
}
fn tls_upper() -> R<u64> {
    // Source-derived OS TLS Value<T> upper; native TLS can require zero.
    let l = Layout::new::<Cell<(u64, u64)>>();
    let a = (l.align() as u64).max(16);
    let value = aligned(add(aligned(l.size() as u64, 16)?, 16)?, a)?;
    Layout::from_size_align(native(value)?, native(a)?)
        .map_err(|_| CertificateError::InvalidLayout)?;
    add(32, value)
}
fn identity(pattern: &str, versions: [&str; 3], sources: [&str; 3]) -> R<()> {
    // Full raw literal equality is stronger than trusting a caller-supplied digest.
    if pattern != PATTERN {
        return Err(CertificateError::LiteralMismatch);
    }
    if versions != ["1.13.0", "0.4.15", "0.8.11"] {
        return Err(CertificateError::DependencyMismatch);
    }
    if sources.iter().any(|&s| s != SOURCE) {
        return Err(CertificateError::SourceMismatch);
    }
    Ok(())
}
fn init_model(i: u64, z: u64) -> R<()> {
    let j = sum(&[(1, 237_931), (6461, i), (10, z)])?;
    if j >= K {
        return Err(CertificateError::InitPredicate);
    }
    // Noncircular representational-fence model; NOT an allocation charge.
    let m0 = sum(&[
        (1, 4_435_661_620_096),
        (68_854_223_672, i),
        (268_437_504, z),
    ])?;
    native(m0)?;
    Ok(())
}
// Entries are (coefficient, real payload layout, Arc payload?, single-request bound).
// These fixed stack arrays are internal calculation inputs, never inventory/root DTOs.
type Term = (u64, Layout, bool, u64);
fn polynomial(constant: u64, tau: u64, terms: &[Term]) -> R<u64> {
    let mut total = add(constant, tau)?;
    for &(coefficient, layout, is_arc, max_request) in terms {
        if layout.size() < 2 {
            return Err(CertificateError::InvalidLayout);
        }
        request(layout, max_request)?;
        let value = if is_arc {
            arc_payload(layout)?
        } else {
            layout.size() as u64
        };
        total = add(total, mul(coefficient, value)?)?;
    }
    Ok(total)
}

/// Both complete NEW requested-layout envelopes, (cold constructor, first search).
/// S is one high-water envelope for arbitrarily many serial iterator/positive/
/// CFG-negative/empty-probe advances; do not multiply it by the number of scans.
/// The live hybrid backing is included alongside fallback, not max'ed away.
/// No numeric/native success is implied by the presence of this source code.
pub fn checked_requested_layouts(pattern: &str) -> R<(u64, u64)> {
    let (sv, ss, sx) = regex_syntax::__clip_regex_private_layouts();
    let (av, asource, features, ra) = regex_automata::__clip_regex_private_layouts();
    identity(
        pattern,
        [env!("CARGO_PKG_VERSION"), av, sv],
        [env!("CLIP_REGEX_SOURCE_VERIFIED"), asource, ss],
    )?;
    if !cfg!(all(
        target_vendor = "apple",
        target_arch = "aarch64",
        target_pointer_width = "64",
        target_feature = "neon",
        target_endian = "little"
    )) {
        return Err(CertificateError::UnsupportedTarget);
    }
    if !features || cfg!(feature = "logging") {
        return Err(CertificateError::UnsupportedFeatures);
    }
    native_facts(sx, ra)
}

fn native_facts(sx: [Layout; 7], ra: [Layout; 17]) -> R<(u64, u64)> {
    if Layout::new::<usize>().size() != 8
        || Layout::new::<usize>().align() != 8
        || Layout::new::<AtomicUsize>().size() != 8
        || Layout::new::<AtomicUsize>().align() != 8
        || Layout::new::<StateID>().size() != 4
        || Layout::new::<SmallIndex>().size() != 4
        || Layout::new::<LazyStateID>().size() < 4
        || ra[15].size() < 8
    {
        return Err(CertificateError::InvalidLayout);
    }
    init_model(
        Layout::new::<LazyStateID>().size() as u64,
        ra[15].size() as u64,
    )?;
    let ast_box = largest(
        [
            Layout::new::<ast::Ast>(),
            Layout::new::<ast::Span>(),
            Layout::new::<ast::SetFlags>(),
            Layout::new::<ast::Literal>(),
            Layout::new::<ast::Assertion>(),
            Layout::new::<ast::ClassUnicode>(),
            Layout::new::<ast::ClassPerl>(),
            Layout::new::<ast::ClassBracketed>(),
            Layout::new::<ast::Repetition>(),
            Layout::new::<ast::Group>(),
            Layout::new::<ast::Alternation>(),
            Layout::new::<ast::Concat>(),
        ],
        1,
    )?;
    let ast_element = largest(
        [
            Layout::new::<ast::Ast>(),
            Layout::new::<ast::ClassSet>(),
            Layout::new::<ast::ClassSetItem>(),
            Layout::new::<ast::FlagsItem>(),
        ],
        364,
    )?;
    // Constant-byte and fixed-width arrays from N(b,e,u), syntax and search.
    request(Layout::new::<u8>(), 13_216)?;
    request(Layout::new::<usize>(), 9_380)?; // two-slot Pike tables, individual <= 2F+2
    request(Layout::new::<u8>(), 524_288)?; // max SINGLE visited-backing request, not old+new sum
    request(Layout::new::<u8>(), 66_096)?; // one encoding scratch request <= 2Dmax
                                           // Exact NEON hashbrown control group: bucket elements then aligned controls.
    let entry = ra[16];
    let table_align = entry.align().max(8);
    let controls = aligned(mul(524_288, entry.size() as u64)?, table_align as u64)?;
    let table_bytes = add(add(controls, 524_288)?, 8)?;
    Layout::from_size_align(native(table_bytes)?, table_align)
        .map_err(|_| CertificateError::InvalidLayout)?;
    let tau = tls_upper()?;
    let constructor = polynomial(
        1754183,
        tau,
        &[
            (364, ast_box, false, 1),                      // s_ast_box_payload_max
            (3640, ast_element, false, 364),               // s_ast_vector_element_max
            (273, sx[0], false, 182),                      // s_parser_group_state
            (273, sx[1], false, 182),                      // s_parser_class_state
            (1092, sx[2], false, 364),                     // s_ast_visitor_frame
            (1092, sx[3], false, 364),                     // s_class_visitor_frame
            (1092, sx[4], false, 728),                     // s_hir_frame
            (3096, Layout::new::<hir::Hir>(), false, 728), // s_hir
            (364, sx[5], false, 1),                        // s_properties_inner
            (
                819898,
                Layout::new::<hir::ClassUnicodeRange>(),
                false,
                15044,
            ), // s_unicode_range
            (2, Layout::new::<hir::ClassBytesRange>(), false, 2), // s_byte_range
            (6000, Layout::new::<hir::literal::Literal>(), false, 500), // s_literal_seq_literal
            (33888, ra[0], false, 13214),                  // s_builder_state
            (33888, Layout::new::<thompson::State>(), false, 13214), // s_nfa_state
            (33888, Layout::new::<(StateID, StateID)>(), false, 13214), // s_sid_pair
            (10, Layout::new::<(SmallIndex, SmallIndex)>(), false, 4), // s_smallindex_pair
            (10, Layout::new::<HashMap<Arc<str>, SmallIndex>>(), false, 4), // s_capture_name_map
            (20, Layout::new::<Vec<Option<Arc<str>>>>(), false, 4), // s_vec_optional_arcstr
            (12, Layout::new::<Option<Arc<str>>>(), false, 4), // s_optional_arcstr
            (4, ra[6], true, 1),                           // arc_group_info_inner
            (2, ra[1], true, 1),                           // arc_nfa_inner
            (58890, Layout::new::<thompson::Transition>(), false, 13150), // s_transition
            (10000, ra[2], false, 10000),                  // s_utf8_compiled_entry
            (12, ra[4], false, 8),                         // s_utf8_node
            (12, sx[6], false, 4),                         // s_scalar_range
            (1000, ra[3], false, 1000),                    // s_utf8_suffix_entry
            (5, Layout::new::<String>(), false, 4),        // s_string
            (5, Layout::new::<hir::Properties>(), false, 4), // s_properties_handle
            (1, ra[7], true, 1),                           // arc_regex_info_inner
            (1, ra[8], true, 1),                           // arc_core_strategy
            (1, ra[9], true, 1),                           // arc_regex_inner
            (1, ra[10], false, 1),                         // s_inner_pool
            (1, ra[11], false, 1),                         // s_creator_environment
            (8, ra[12], false, 8),                         // s_shard_cell
            (42, ra[5], false, 2),                         // s_range_trie_state
        ],
    )?;
    let first_search = polynomial(
        20018036,
        tau,
        &[
            (3145764, Layout::new::<LazyStateID>(), false, 1048576), // s_lazy_state_id
            (1572864, ra[15], false, 524288),                        // s_determinize_state
            (2097152, ra[16], false, 524288),                        // s_state_map_entry
            (2228190, ra[13], false, 1485460),                       // s_backtrack_frame
            (51, ra[14], false, 34),                                 // s_pike_follow_epsilon
            (1, Layout::new::<regex_automata::MatchErrorKind>(), false, 1), // s_match_error_kind
        ],
    )?;
    Ok((constructor, first_search))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn literal_and_dependency_identity_are_exact() {
        let v = ["1.13.0", "0.4.15", "0.8.11"];
        assert_eq!(identity(PATTERN, v, [SOURCE; 3]), Ok(()));
        assert_eq!(
            identity("(?i)<|startoftext|>", v, [SOURCE; 3]),
            Err(CertificateError::LiteralMismatch)
        );
        assert_eq!(
            identity(PATTERN, ["1.13.1", v[1], v[2]], [SOURCE; 3]),
            Err(CertificateError::DependencyMismatch)
        );
        assert_eq!(
            identity(PATTERN, v, [SOURCE, "unverified", SOURCE]),
            Err(CertificateError::SourceMismatch)
        );
    }
    #[test]
    fn overflow_is_not_a_saturating_certificate() {
        assert_eq!(sum(&[(u64::MAX, 2)]), Err(CertificateError::Overflow));
        assert_eq!(
            sum(&[(u64::MAX, 1), (1, 1)]),
            Err(CertificateError::Overflow)
        );
        assert_eq!(init_model(u64::MAX, 8), Err(CertificateError::Overflow));
        assert_eq!(
            polynomial(u64::MAX, 1, &[]),
            Err(CertificateError::Overflow)
        );
    }
    #[test]
    fn init_predicate_is_strict_and_checked_before_any_facts() {
        // Synthetic arithmetic inputs, NOT asserted native enum sizes.
        assert_eq!(init_model(4, 8), Ok(()));
        assert_eq!(init_model(4, K), Err(CertificateError::InitPredicate));
        assert_eq!(init_model(K, 8), Err(CertificateError::InitPredicate));
        let (_, _, sx) = regex_syntax::__clip_regex_private_layouts();
        let (_, _, _, mut ra) = regex_automata::__clip_regex_private_layouts();
        ra[15] = Layout::from_size_align(K as usize, 8).unwrap();
        assert_eq!(native_facts(sx, ra), Err(CertificateError::InitPredicate));
    }
    #[test]
    fn real_type_queries_produce_two_distinct_finite_terms_when_supported() {
        // Hosted-only test; source authoring did not execute these queries.
        let result = checked_requested_layouts(PATTERN);
        let eligible = cfg!(all(
            target_vendor = "apple",
            target_arch = "aarch64",
            target_pointer_width = "64",
            target_feature = "neon",
            target_endian = "little"
        )) && env!("CLIP_REGEX_SOURCE_VERIFIED") == SOURCE
            && !cfg!(feature = "logging")
            && regex_automata::__clip_regex_private_layouts().2;
        if eligible {
            let (cold, search) = result.expect("native J/layouts and both checked formulas");
            assert!(cold > 1_754_183);
            assert!(search > 20_018_036);
            assert_ne!(cold, search);
        } else {
            assert!(result.is_err());
        }
    }
    #[test]
    fn search_keeps_hybrid_and_both_fallback_stores() {
        let (_, _, sx) = regex_syntax::__clip_regex_private_layouts();
        let (_, _, _, ra) = regex_automata::__clip_regex_private_layouts();
        if Layout::new::<usize>().size() == 8 {
            let (_, search) = native_facts(sx, ra).expect("actual private layouts and J");
            let i = Layout::new::<LazyStateID>().size() as u64;
            let z = ra[15].size() as u64;
            let e = ra[16].size() as u64;
            let both_hybrids = sum(&[
                (1, 9_438_018 + 9_568_466),
                (3_145_764, i),
                (1_572_864, z),
                (2_097_152, e),
            ])
            .unwrap();
            let fallback = sum(&[
                (1, 225_104 + 16 + 786_432),
                (2_228_190, ra[13].size() as u64),
                (51, ra[14].size() as u64),
                (
                    1,
                    Layout::new::<regex_automata::MatchErrorKind>().size() as u64,
                ),
            ])
            .unwrap();
            assert_eq!(
                search,
                add(add(both_hybrids, fallback).unwrap(), tls_upper().unwrap()).unwrap()
            );
            assert!(search > both_hybrids && search > fallback);
        }
    }
    #[test]
    fn arc_payload_is_not_the_arc_pointer_layout() {
        let l = Layout::new::<[u64; 8]>();
        assert_eq!(arc_payload(l).unwrap(), 80);
        assert!(arc_payload(l).unwrap() > Layout::new::<Arc<[u64; 8]>>().size() as u64);
        assert_eq!(tls_upper().unwrap(), 64);
    }
}
