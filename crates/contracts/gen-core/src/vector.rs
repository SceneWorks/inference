//! Bounded, inert static SVG sanitization, canonicalization, derived preview rasterization, and
//! typed [`VectorOutput`] construction (`PIPE-12`, `E4`).
//!
//! This module is deliberately **tensor-free** and **zero-external-XML-dependency** so it lives
//! inside `gen-core` and is shared identically by MLX and Candle StarVector routes as well as
//! SceneWorks consumers.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use crate::core_llm::StarVectorFinishReason;
use crate::{Error, Image, Result, VectorFinishReason, VectorOutput, VectorTerminalMetadata};

/// Canonical sanitizer policy/version identifier embedded in [`VectorTerminalMetadata::sanitizer_id`].
pub const VECTOR_SANITIZER_VERSION: &str = "sceneworks-static-svg-v2";
/// Canonical SVG XML namespace enforced on every sanitized root element.
pub const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";
/// Canonical XLink XML namespace required on root `<svg>` whenever `xlink:href` is used.
pub const XLINK_NAMESPACE: &str = "http://www.w3.org/1999/xlink";
/// Maximum accepted raw/canonical SVG document size in UTF-8 bytes (256 KiB).
pub const MAX_SVG_BYTES: usize = 256 * 1024;
/// Maximum allowed XML element nesting depth.
pub const MAX_SVG_DEPTH: usize = 32;
/// Maximum allowed retained/expanded SVG elements per document.
pub const MAX_SVG_ELEMENTS: usize = 2_000;
/// Maximum allowed attributes across the entire SVG document.
pub const MAX_SVG_ATTRIBUTES: usize = 4_096;
/// Maximum byte length of any single attribute value (64 KiB).
pub const MAX_SVG_ATTRIBUTE_VALUE_BYTES: usize = 64 * 1024;
/// Maximum total byte length across all attribute values in one document (192 KiB).
pub const MAX_SVG_ATTRIBUTE_VALUE_BYTES_TOTAL: usize = 192 * 1024;
/// Maximum total `d` path-data bytes across a document (including `<use>` expansion).
pub const MAX_SVG_PATH_DATA_BYTES: usize = 160 * 1024;
/// Maximum total path commands across a document (including `<use>` expansion).
pub const MAX_SVG_PATH_COMMANDS: usize = 16_384;
/// Maximum total path numbers across a document (including `<use>` expansion).
pub const MAX_SVG_PATH_NUMBERS: usize = 65_536;
/// Maximum total CSS bytes across inline styles and `<style>` elements (64 KiB).
pub const MAX_STATIC_SVG_CSS_BYTES: usize = 64 * 1024;
/// Maximum resource definitions (`clipPath`, `linearGradient`, `radialGradient`, `filter`, `symbol`).
pub const MAX_SVG_RESOURCES: usize = 128;
/// Maximum local resource references (`href`, `xlink:href`, `url(#id)`).
pub const MAX_SVG_RESOURCE_REFERENCES: usize = 256;
/// Maximum gradient `<stop>` elements per document.
pub const MAX_SVG_GRADIENT_STOPS: usize = 512;
/// Maximum `<filter>` primitive elements per document.
pub const MAX_SVG_FILTER_PRIMITIVES: usize = 64;
/// Maximum `stdDeviation` blur sigma on `<feGaussianBlur>`.
pub const MAX_SVG_FILTER_BLUR_SIGMA: f64 = 64.0;
/// Maximum preview width/height edge in pixels.
pub const MAX_PREVIEW_DIMENSION: u32 = 2_048;
/// Default derived preview maximum edge in pixels.
pub const DEFAULT_PREVIEW_DIMENSION: u32 = 512;
/// Upper bound on parsed SVG root dimension / viewBox coordinate magnitude.
const MAX_SVG_COORDINATE_MAGNITUDE: f64 = 1_000_000.0;
const XML_DECLARATION: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";

/// Options controlling SVG sanitization and derived preview rasterization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorSanitizeOptions {
    /// Maximum accepted SVG byte length (clamped to [`MAX_SVG_BYTES`]).
    pub max_bytes: usize,
    /// Whether to derive a rasterized RGB preview [`Image`] for [`VectorOutput::preview`].
    pub render_preview: bool,
    /// Maximum preview edge dimension in pixels (clamped to `1..=MAX_PREVIEW_DIMENSION`).
    pub preview_max_dimension: u32,
}

impl Default for VectorSanitizeOptions {
    fn default() -> Self {
        Self {
            max_bytes: MAX_SVG_BYTES,
            render_preview: true,
            preview_max_dimension: DEFAULT_PREVIEW_DIMENSION,
        }
    }
}

/// Sanitized, canonical static SVG document plus extracted viewport and provenance facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalSvg {
    /// Canonical UTF-8 SVG document (starts with `<?xml version="1.0" encoding="UTF-8"?>`).
    pub svg: String,
    /// Canonical integer viewport width (pixels/units, `>= 1`).
    pub width: u32,
    /// Canonical integer viewport height (pixels/units, `>= 1`).
    pub height: u32,
    /// Count of retained static SVG elements (including the root `<svg>`).
    pub element_count: usize,
    /// Lowercase SHA-256 hex digest of `svg.as_bytes()`.
    pub svg_sha256: String,
}

/// Map a native [`StarVectorFinishReason`] onto `(VectorFinishReason, provider_reason_str, truncated)`.
pub const fn map_starvector_finish_reason(
    reason: StarVectorFinishReason,
) -> (VectorFinishReason, &'static str, bool) {
    match reason {
        StarVectorFinishReason::CompleteRoot => (VectorFinishReason::Stop, "complete_root", false),
        StarVectorFinishReason::Eos => (VectorFinishReason::Stop, "eos", false),
        StarVectorFinishReason::TokenLimit => (VectorFinishReason::Length, "token_limit", true),
        StarVectorFinishReason::ByteLimit => (VectorFinishReason::Length, "byte_limit", true),
        StarVectorFinishReason::WallTimeLimit => {
            (VectorFinishReason::Length, "wall_time_limit", true)
        }
        StarVectorFinishReason::Cancelled => (VectorFinishReason::Canceled, "cancelled", true),
    }
}

/// Compute lowercase SHA-256 hex digest of UTF-8/raw bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

/// Sanitize and canonicalize an SVG UTF-8 string using default limits.
pub fn sanitize_svg(input: &str) -> Result<CanonicalSvg> {
    sanitize_svg_with_options(input.as_bytes(), VectorSanitizeOptions::default())
}

/// Sanitize and canonicalize raw UTF-8 SVG bytes using default limits.
pub fn sanitize_svg_bytes(input: &[u8]) -> Result<CanonicalSvg> {
    sanitize_svg_with_options(input, VectorSanitizeOptions::default())
}

/// Sanitize and canonicalize raw UTF-8 SVG bytes with explicit options.
pub fn sanitize_svg_with_options(
    input: &[u8],
    options: VectorSanitizeOptions,
) -> Result<CanonicalSvg> {
    let max_bytes = options.max_bytes.min(MAX_SVG_BYTES);
    if max_bytes == 0 {
        return Err(Error::Msg(
            "vector sanitizer max_bytes must be non-zero".into(),
        ));
    }
    if input.is_empty() {
        return Err(Error::Msg("vector SVG input is empty".into()));
    }
    if input.len() > max_bytes {
        return Err(Error::Msg(format!(
            "vector SVG exceeds the {max_bytes}-byte sanitizer budget ({} bytes)",
            input.len()
        )));
    }
    if input.contains(&0) {
        return Err(Error::Msg("vector SVG contains forbidden NUL bytes".into()));
    }
    let text = std::str::from_utf8(input)
        .map_err(|err| Error::Msg(format!("vector SVG is not valid UTF-8: {err}")))?;
    validate_xml_chars(text, "vector SVG source")?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(Error::Msg("vector SVG input is empty".into()));
    }

    let nodes = parse_and_sanitize_document(trimmed)?;
    let root = nodes
        .first()
        .ok_or_else(|| Error::Msg("vector SVG has no root <svg> element".into()))?;
    if root.name != "svg" || nodes.len() != 1 {
        return Err(Error::Msg(
            "vector SVG must contain exactly one <svg> root element".into(),
        ));
    }

    let (width, height, canonical_root) = canonicalize_root(root)?;
    let element_count = count_elements(&canonical_root);
    if element_count > MAX_SVG_ELEMENTS {
        return Err(Error::Msg(format!(
            "vector SVG exceeds the {MAX_SVG_ELEMENTS}-element budget ({element_count} elements)"
        )));
    }

    let mut out = String::with_capacity(trimmed.len().min(max_bytes) + XML_DECLARATION.len() + 64);
    out.push_str(XML_DECLARATION);
    serialize_node(&canonical_root, &mut out);
    if out.len().saturating_sub(XML_DECLARATION.len()) > max_bytes {
        return Err(Error::Msg(format!(
            "canonical SVG exceeds the {max_bytes}-byte sanitizer budget"
        )));
    }
    let svg_sha256 = sha256_hex(out.as_bytes());
    Ok(CanonicalSvg {
        svg: out,
        width,
        height,
        element_count,
        svg_sha256,
    })
}

/// Build a typed [`VectorOutput`] from a provider's raw output, finish reason, and counters (`PIPE-12`).
///
/// - `CompleteRoot` / `Eos`: sanitizes `raw_svg`, derives an RGB preview when requested, and
///   populates `VectorTerminalMetadata`.
/// - `Cancelled`: returns `Err(Error::Canceled)`.
/// - `TokenLimit` / `ByteLimit` / `WallTimeLimit`: when `allow_non_publishable` is `true`, returns
///   a typed `VectorOutput` with `svg: ""`, `preview: None`, `finish_reason: VectorFinishReason::Length`,
///   `truncated: true`, and populated terminal metadata; otherwise fails closed with `Err(Error::Msg)`.
#[allow(clippy::too_many_arguments)]
pub fn build_vector_output(
    raw_svg: Option<&str>,
    finish_reason: StarVectorFinishReason,
    generated_tokens: u32,
    prompt_tokens: Option<usize>,
    generated_bytes: usize,
    provider_id: &str,
    options: VectorSanitizeOptions,
    allow_non_publishable: bool,
) -> Result<VectorOutput> {
    let (mapped_reason, provider_reason, truncated) = map_starvector_finish_reason(finish_reason);
    match finish_reason {
        StarVectorFinishReason::Cancelled => Err(Error::Canceled),
        StarVectorFinishReason::TokenLimit
        | StarVectorFinishReason::ByteLimit
        | StarVectorFinishReason::WallTimeLimit => {
            if !allow_non_publishable {
                return Err(Error::Msg(format!(
                    "vector provider stopped at {provider_reason}; no partial SVG was published"
                )));
            }
            Ok(VectorOutput {
                svg: String::new(),
                preview: None,
                finish_reason: mapped_reason,
                generated_tokens: generated_tokens as usize,
                prompt_tokens,
                truncated,
                terminal: VectorTerminalMetadata {
                    sanitizer_id: Some(VECTOR_SANITIZER_VERSION.to_owned()),
                    viewport: None,
                    svg_sha256: None,
                    element_count: None,
                    provider_finish_reason: Some(provider_reason.to_owned()),
                    generated_bytes: Some(generated_bytes),
                    provider_id: Some(provider_id.to_owned()),
                    kv_cache: None,
                },
            })
        }
        StarVectorFinishReason::CompleteRoot | StarVectorFinishReason::Eos => {
            let source = raw_svg.ok_or_else(|| {
                Error::Msg("vector provider completed without a publishable SVG".into())
            })?;
            let canonical = sanitize_svg_with_options(source.as_bytes(), options)?;
            let preview = if options.render_preview {
                Some(render_svg_preview(
                    &canonical,
                    options.preview_max_dimension,
                )?)
            } else {
                None
            };
            Ok(VectorOutput {
                svg: canonical.svg,
                preview,
                finish_reason: mapped_reason,
                generated_tokens: generated_tokens as usize,
                prompt_tokens,
                truncated,
                terminal: VectorTerminalMetadata {
                    sanitizer_id: Some(VECTOR_SANITIZER_VERSION.to_owned()),
                    viewport: Some([canonical.width, canonical.height]),
                    svg_sha256: Some(canonical.svg_sha256),
                    element_count: Some(canonical.element_count),
                    provider_finish_reason: Some(provider_reason.to_owned()),
                    generated_bytes: Some(generated_bytes),
                    provider_id: Some(provider_id.to_owned()),
                    kv_cache: None,
                },
            })
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Bounded Static SVG Parser & Sanitizer
// -------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct SvgNode {
    name: String,
    attrs: Vec<(String, String)>,
    children: Vec<SvgChild>,
}

#[derive(Clone, Debug)]
enum SvgChild {
    Element(SvgNode),
    Text(String),
}

fn is_valid_xml_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x09 | 0x0A | 0x0D | 0x20..=0xD7FF | 0xE000..=0xFFFD | 0x10000..=0x10FFFF
    )
}

fn validate_xml_chars(text: &str, label: &str) -> Result<()> {
    if let Some(bad) = text.chars().find(|&c| !is_valid_xml_char(c)) {
        return Err(Error::Msg(format!(
            "{label} contains forbidden XML 1.0 character U+{:04X}",
            bad as u32
        )));
    }
    Ok(())
}

fn is_allowed_static_element(name: &str, hidden: bool) -> bool {
    matches!(
        name,
        "svg"
            | "g"
            | "defs"
            | "symbol"
            | "use"
            | "path"
            | "rect"
            | "circle"
            | "ellipse"
            | "line"
            | "polyline"
            | "polygon"
            | "style"
            | "title"
            | "desc"
            | "metadata"
            | "linearGradient"
            | "radialGradient"
            | "stop"
            | "clipPath"
            | "filter"
            | "feGaussianBlur"
            | "feOffset"
            | "feMerge"
            | "feMergeNode"
    ) || (hidden && matches!(name, "text" | "tspan"))
}

fn is_forbidden_attribute_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("on")
        || matches!(
            lower.as_str(),
            "src" | "cursor" | "action" | "formaction" | "srcdoc"
        )
}

fn is_allowed_static_attribute(element: &str, key: &str, hidden: bool) -> bool {
    let common = matches!(key, "id" | "class" | "style");
    let presentation = matches!(
        key,
        "display"
            | "visibility"
            | "color"
            | "color-interpolation"
            | "color-interpolation-filters"
            | "color-rendering"
            | "fill"
            | "fill-opacity"
            | "fill-rule"
            | "clip-rule"
            | "clip-path"
            | "filter"
            | "opacity"
            | "paint-order"
            | "shape-rendering"
            | "stroke"
            | "stroke-opacity"
            | "stroke-width"
            | "stroke-linecap"
            | "stroke-linejoin"
            | "stroke-miterlimit"
            | "stroke-dasharray"
            | "stroke-dashoffset"
            | "transform"
            | "vector-effect"
    );
    common
        || (presentation
            && matches!(
                element,
                "svg"
                    | "g"
                    | "symbol"
                    | "use"
                    | "path"
                    | "rect"
                    | "circle"
                    | "ellipse"
                    | "line"
                    | "polyline"
                    | "polygon"
                    | "text"
                    | "tspan"
            ))
        || match element {
            "svg" => matches!(
                key,
                "xmlns"
                    | "xmlns:xlink"
                    | "version"
                    | "baseProfile"
                    | "width"
                    | "height"
                    | "viewBox"
                    | "preserveAspectRatio"
                    | "overflow"
                    | "x"
                    | "y"
                    | "xml:space"
                    | "enable-background"
            ),
            "defs" | "g" | "feMerge" => false,
            "symbol" => matches!(
                key,
                "viewBox" | "preserveAspectRatio" | "overflow" | "x" | "y" | "width" | "height"
            ),
            "use" => matches!(key, "href" | "xlink:href" | "x" | "y" | "width" | "height"),
            "path" => matches!(key, "d" | "pathLength"),
            "rect" => matches!(key, "x" | "y" | "width" | "height" | "rx" | "ry"),
            "circle" => matches!(key, "cx" | "cy" | "r" | "pathLength"),
            "ellipse" => matches!(key, "cx" | "cy" | "rx" | "ry" | "pathLength"),
            "line" => matches!(key, "x1" | "x2" | "y1" | "y2" | "pathLength"),
            "polyline" | "polygon" => matches!(key, "points" | "pathLength"),
            "style" => key == "type",
            "title" | "desc" | "metadata" => key == "id",
            "linearGradient" => matches!(
                key,
                "x1" | "y1"
                    | "x2"
                    | "y2"
                    | "gradientUnits"
                    | "gradientTransform"
                    | "spreadMethod"
                    | "href"
                    | "xlink:href"
            ),
            "radialGradient" => matches!(
                key,
                "cx" | "cy"
                    | "r"
                    | "fx"
                    | "fy"
                    | "fr"
                    | "gradientUnits"
                    | "gradientTransform"
                    | "spreadMethod"
                    | "href"
                    | "xlink:href"
            ),
            "stop" => matches!(key, "offset" | "stop-color" | "stop-opacity"),
            "clipPath" => matches!(key, "clipPathUnits"),
            "filter" => matches!(
                key,
                "x" | "y" | "width" | "height" | "filterUnits" | "primitiveUnits"
            ),
            "feGaussianBlur" => matches!(key, "in" | "result" | "stdDeviation"),
            "feOffset" => matches!(key, "in" | "result" | "dx" | "dy"),
            "feMergeNode" => key == "in",
            "text" | "tspan" if hidden => matches!(
                key,
                "x" | "y"
                    | "dx"
                    | "dy"
                    | "font"
                    | "font-family"
                    | "font-size"
                    | "font-style"
                    | "font-weight"
                    | "text-anchor"
                    | "xml:space"
            ),
            _ => false,
        }
}

fn is_allowed_css_property(name: &str) -> bool {
    matches!(
        name,
        "fill"
            | "fill-rule"
            | "clip-rule"
            | "fill-opacity"
            | "stroke"
            | "stroke-opacity"
            | "stroke-width"
            | "stroke-linecap"
            | "stroke-linejoin"
            | "stroke-miterlimit"
            | "stroke-dasharray"
            | "stroke-dashoffset"
            | "opacity"
            | "color"
            | "color-interpolation"
            | "color-interpolation-filters"
            | "color-rendering"
            | "display"
            | "visibility"
            | "shape-rendering"
            | "paint-order"
            | "vector-effect"
            | "clip-path"
            | "filter"
            | "stop-color"
            | "stop-opacity"
            | "flood-color"
            | "flood-opacity"
            | "enable-background"
    )
}

fn validate_static_svg_id(value: &str, label: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || value
            .chars()
            .any(|ch| ch.is_control() || ch.is_whitespace() || ch == '#' || ch == '\\')
    {
        return Err(Error::Msg(format!(
            "vector SVG {label} is not a bounded local identifier"
        )));
    }
    Ok(())
}

fn local_href_target(value: &str) -> Result<&str> {
    let trimmed = value.trim();
    let target = trimmed
        .strip_prefix('#')
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            Error::Msg(format!(
                "vector SVG href must be an exact local fragment, got {value:?}"
            ))
        })?;
    validate_static_svg_id(target, "local href reference")?;
    Ok(target)
}

fn local_url_target(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    let inner = trimmed.strip_prefix("url(")?.strip_suffix(')')?.trim();
    let unquoted = if inner.len() >= 2
        && ((inner.starts_with('\'') && inner.ends_with('\''))
            || (inner.starts_with('"') && inner.ends_with('"')))
    {
        &inner[1..inner.len() - 1]
    } else {
        inner
    };
    if unquoted.contains('(') || unquoted.contains(')') || unquoted.contains('\\') {
        return None;
    }
    let target = unquoted.strip_prefix('#')?;
    if validate_static_svg_id(target, "url resource reference").is_ok() {
        Some(target)
    } else {
        None
    }
}

fn contains_external_or_active_tokens(value: &str) -> bool {
    if value.contains('\\') || value.contains("/*") || value.contains("*/") {
        return true;
    }
    let compact: String = value
        .chars()
        .filter(|c| !c.is_ascii_control() && !c.is_ascii_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();
    compact.contains("javascript:")
        || compact.contains("vbscript:")
        || compact.contains("data:")
        || compact.contains("file:")
        || compact.contains("http:")
        || compact.contains("https:")
        || compact.contains("ftp:")
        || compact.contains("ws:")
        || compact.contains("wss:")
        || compact.contains("//")
        || compact.contains("expression(")
        || compact.contains("behavior:")
        || compact.contains("binding:")
        || compact.contains("-moz-binding")
        || compact.contains("image-set(")
        || compact.contains("@import")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaticReferenceKind {
    Paint,
    ClipPath,
    Filter,
}

#[derive(Debug)]
struct TypedReference {
    source_node: usize,
    target: String,
    kind: StaticReferenceKind,
}

#[derive(Default)]
struct SanitizerCounters {
    total_attrs: usize,
    total_attr_value_bytes: usize,
    css_bytes: usize,
    resources: usize,
    gradient_stops: usize,
    filter_primitives: usize,
    reference_count: usize,
    path_data_bytes: usize,
    path_commands: usize,
    path_numbers: usize,
}

fn record_reference_count(counters: &mut SanitizerCounters) -> Result<()> {
    counters.reference_count = counters
        .reference_count
        .checked_add(1)
        .ok_or_else(|| Error::Msg("vector SVG reference count overflow".into()))?;
    if counters.reference_count > MAX_SVG_RESOURCE_REFERENCES {
        return Err(Error::Msg(format!(
            "vector SVG exceeds the {MAX_SVG_RESOURCE_REFERENCES}-reference budget"
        )));
    }
    Ok(())
}

fn validate_css_value_functions(property: &str, value: &str) -> Result<Option<String>> {
    if contains_external_or_active_tokens(value) || value.contains('@') {
        return Err(Error::Msg(format!(
            "vector SVG CSS property '{property}' contains forbidden external, escape, or active content"
        )));
    }
    let lower = value.to_ascii_lowercase();
    if lower.contains('(') || lower.contains(')') {
        // Strict CSS value function allowlist: only exact local url(#id) on resource properties
        // or static rgb()/rgba()/hsl()/hsla() color functions.
        if let Some(target) = local_url_target(value) {
            if matches!(property, "fill" | "stroke" | "clip-path" | "filter") {
                return Ok(Some(target.to_owned()));
            }
            return Err(Error::Msg(format!(
                "vector SVG CSS property '{property}' does not permit url(...) references"
            )));
        }
        let mut rest = lower.as_str();
        while let Some(open) = rest.find('(') {
            let prefix = rest[..open].trim_end();
            let fn_name = prefix
                .rsplit(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
                .next()
                .unwrap_or("");
            if !matches!(fn_name, "rgb" | "rgba" | "hsl" | "hsla") {
                return Err(Error::Msg(format!(
                    "vector SVG CSS function '{fn_name}(' is forbidden in '{property}'"
                )));
            }
            let after_open = &rest[open + 1..];
            let Some(close) = after_open.find(')') else {
                return Err(Error::Msg(format!(
                    "vector SVG CSS property '{property}' has an unterminated function"
                )));
            };
            let args = &after_open[..close];
            if args.contains('(')
                || !args.chars().all(|c| {
                    c.is_ascii_digit()
                        || c.is_ascii_whitespace()
                        || matches!(c, ',' | '.' | '%' | '/' | '+' | '-')
                })
            {
                return Err(Error::Msg(format!(
                    "vector SVG CSS color function in '{property}' has invalid arguments"
                )));
            }
            rest = &after_open[close + 1..];
        }
        if rest.contains(')') {
            return Err(Error::Msg(format!(
                "vector SVG CSS property '{property}' has an unmatched ')'"
            )));
        }
    }
    Ok(None)
}

fn validate_css_declarations(
    css: &str,
    source_node: usize,
    counters: &mut SanitizerCounters,
    typed_refs: &mut Vec<TypedReference>,
) -> Result<String> {
    if css.contains('\\')
        || css.contains("/*")
        || css.contains("*/")
        || css.contains('@')
        || css.contains('{')
        || css.contains('}')
    {
        return Err(Error::Msg(
            "vector SVG CSS contains forbidden escapes, comments, at-rules, or blocks".into(),
        ));
    }
    let mut kept = Vec::new();
    for raw_decl in css.split(';') {
        let decl = raw_decl.trim();
        if decl.is_empty() {
            continue;
        }
        let Some((raw_prop, raw_val)) = decl.split_once(':') else {
            return Err(Error::Msg(
                "vector SVG CSS declaration is missing ':'".into(),
            ));
        };
        let prop = raw_prop.trim();
        let val = raw_val.trim();
        if prop.is_empty() || val.is_empty() || prop.starts_with("--") || val.contains(':') {
            return Err(Error::Msg(format!(
                "vector SVG CSS declaration '{decl}' is malformed or disallowed"
            )));
        }
        let prop_lower = prop.to_ascii_lowercase();
        if !is_allowed_css_property(&prop_lower) {
            return Err(Error::Msg(format!(
                "vector SVG CSS property '{prop}' is not permitted by the static allowlist"
            )));
        }
        if prop_lower == "vector-effect" && !matches!(val, "none" | "non-scaling-stroke") {
            return Err(Error::Msg(
                "vector SVG CSS vector-effect value is unsupported".into(),
            ));
        }
        if let Some(target) = validate_css_value_functions(&prop_lower, val)? {
            let kind = match prop_lower.as_str() {
                "fill" | "stroke" => StaticReferenceKind::Paint,
                "clip-path" => StaticReferenceKind::ClipPath,
                "filter" => StaticReferenceKind::Filter,
                _ => unreachable!("guarded in validate_css_value_functions"),
            };
            record_reference_count(counters)?;
            typed_refs.push(TypedReference {
                source_node,
                target,
                kind,
            });
        }
        kept.push(format!("{prop_lower}:{val}"));
    }
    if kept.is_empty() && !css.trim().is_empty() {
        return Err(Error::Msg(
            "vector SVG CSS contains no valid declarations".into(),
        ));
    }
    Ok(kept.join(";"))
}

fn validate_inline_style(
    style: &str,
    source_node: usize,
    counters: &mut SanitizerCounters,
    typed_refs: &mut Vec<TypedReference>,
) -> Result<String> {
    counters.css_bytes = counters
        .css_bytes
        .checked_add(style.len())
        .ok_or_else(|| Error::Msg("vector SVG CSS byte count overflow".into()))?;
    if counters.css_bytes > MAX_STATIC_SVG_CSS_BYTES {
        return Err(Error::Msg(format!(
            "vector SVG CSS exceeds the {MAX_STATIC_SVG_CSS_BYTES}-byte budget"
        )));
    }
    validate_css_declarations(style, source_node, counters, typed_refs)
}

fn validate_stylesheet(
    css: &str,
    source_node: usize,
    counters: &mut SanitizerCounters,
    typed_refs: &mut Vec<TypedReference>,
) -> Result<String> {
    counters.css_bytes = counters
        .css_bytes
        .checked_add(css.len())
        .ok_or_else(|| Error::Msg("vector SVG stylesheet byte count overflow".into()))?;
    if counters.css_bytes > MAX_STATIC_SVG_CSS_BYTES {
        return Err(Error::Msg(format!(
            "vector SVG stylesheets exceed the {MAX_STATIC_SVG_CSS_BYTES}-byte budget"
        )));
    }
    if css.contains('\\') || css.contains("/*") || css.contains("*/") || css.contains('@') {
        return Err(Error::Msg(
            "vector SVG stylesheet contains forbidden escapes, comments, or @-rules".into(),
        ));
    }
    let mut kept = Vec::new();
    let mut rest = css.trim();
    while !rest.is_empty() {
        let Some(open) = rest.find('{') else {
            return Err(Error::Msg(
                "vector SVG stylesheet contains malformed rule without '{'".into(),
            ));
        };
        let selector = rest[..open].trim();
        if selector.is_empty()
            || selector.contains('}')
            || selector.contains(';')
            || selector.contains('(')
            || selector.contains(')')
            || !selector.chars().all(|c| {
                c.is_ascii_alphanumeric()
                    || c.is_ascii_whitespace()
                    || matches!(c, '.' | '#' | ',' | '>' | '+' | '~' | '*' | '-' | '_' | ':')
            })
        {
            return Err(Error::Msg(format!(
                "vector SVG stylesheet selector '{selector}' is not permitted"
            )));
        }
        let after_open = &rest[open + 1..];
        let Some(close) = after_open.find('}') else {
            return Err(Error::Msg(
                "vector SVG stylesheet has an unterminated rule block".into(),
            ));
        };
        let body = &after_open[..close];
        if body.contains('{') {
            return Err(Error::Msg(
                "vector SVG stylesheet does not permit nested blocks".into(),
            ));
        }
        let clean_body = validate_css_declarations(body, source_node, counters, typed_refs)?;
        if !clean_body.is_empty() {
            kept.push(format!("{selector}{{{clean_body}}}"));
        }
        rest = after_open[close + 1..].trim();
    }
    Ok(kept.join("\n"))
}

fn parse_number_list(raw: &str, label: &str) -> Result<Vec<f64>> {
    let mut nums = Vec::new();
    for token in raw
        .split(|c: char| c.is_ascii_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
    {
        let val: f64 = token.parse().map_err(|_| {
            Error::Msg(format!(
                "vector SVG {label} contains invalid number {token:?}"
            ))
        })?;
        if !val.is_finite() || val.abs() > MAX_SVG_COORDINATE_MAGNITUDE {
            return Err(Error::Msg(format!(
                "vector SVG {label} number {token:?} is out of bounds"
            )));
        }
        nums.push(val);
    }
    if nums.is_empty() {
        return Err(Error::Msg(format!(
            "vector SVG {label} must contain at least one number"
        )));
    }
    Ok(nums)
}

fn scan_path_data_budget(d: &str) -> Result<(usize, usize)> {
    let mut commands = 0usize;
    let mut numbers = 0usize;
    let bytes = d.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() || b == b',' {
            i += 1;
            continue;
        }
        if b.is_ascii_alphabetic() {
            if !matches!(
                b,
                b'M' | b'm'
                    | b'L'
                    | b'l'
                    | b'H'
                    | b'h'
                    | b'V'
                    | b'v'
                    | b'C'
                    | b'c'
                    | b'S'
                    | b's'
                    | b'Q'
                    | b'q'
                    | b'T'
                    | b't'
                    | b'A'
                    | b'a'
                    | b'Z'
                    | b'z'
            ) {
                return Err(Error::Msg(format!(
                    "vector SVG path contains invalid command '{}'",
                    b as char
                )));
            }
            commands += 1;
            i += 1;
            continue;
        }
        let start = i;
        if b == b'+' || b == b'-' {
            i += 1;
        }
        let mut saw_digit = false;
        let mut saw_dot = false;
        while i < bytes.len() {
            let c = bytes[i];
            if c.is_ascii_digit() {
                saw_digit = true;
                i += 1;
            } else if c == b'.' && !saw_dot {
                saw_dot = true;
                i += 1;
            } else if (c == b'e' || c == b'E') && saw_digit {
                i += 1;
                if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                    i += 1;
                }
            } else {
                break;
            }
        }
        if !saw_digit || i == start {
            return Err(Error::Msg("vector SVG path data is malformed".into()));
        }
        let val: f64 = d[start..i]
            .parse()
            .map_err(|_| Error::Msg("vector SVG path data contains invalid number".into()))?;
        if !val.is_finite() || val.abs() > MAX_SVG_COORDINATE_MAGNITUDE {
            return Err(Error::Msg(
                "vector SVG path data number exceeds coordinate bounds".into(),
            ));
        }
        numbers += 1;
    }
    Ok((commands, numbers))
}

struct XmlCursor<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> XmlCursor<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn remaining(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn skip_ws(&mut self) {
        let bytes = self.src.as_bytes();
        while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.src.len()
    }
}

fn decode_xml_entities(input: &str) -> Result<String> {
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    let bytes = input.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'&' {
            let Some(semi) = input[i + 1..].find(';') else {
                return Err(Error::Msg(
                    "vector SVG contains an unterminated XML entity reference".into(),
                ));
            };
            let entity = &input[i + 1..i + 1 + semi];
            let decoded = match entity {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                num if num.starts_with("#x") || num.starts_with("#X") => {
                    let hex = &num[2..];
                    if hex.is_empty() {
                        None
                    } else {
                        u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
                    }
                }
                num if num.starts_with('#') => {
                    let dec = &num[1..];
                    if dec.is_empty() {
                        None
                    } else {
                        dec.parse::<u32>().ok().and_then(char::from_u32)
                    }
                }
                _ => {
                    return Err(Error::Msg(format!(
                        "vector SVG contains unsupported XML entity '&{entity};'"
                    )));
                }
            };
            let ch = decoded.ok_or_else(|| {
                Error::Msg(format!(
                    "vector SVG contains invalid XML character reference '&{entity};'"
                ))
            })?;
            if !is_valid_xml_char(ch) {
                return Err(Error::Msg(format!(
                    "vector SVG entity '&{entity};' decodes to forbidden XML 1.0 character U+{:04X}",
                    ch as u32
                )));
            }
            out.push(ch);
            i += semi + 2;
            continue;
        }
        let ch = input[i..].chars().next().expect("non-empty slice");
        if !is_valid_xml_char(ch) {
            return Err(Error::Msg(format!(
                "vector SVG contains forbidden XML 1.0 character U+{:04X}",
                ch as u32
            )));
        }
        out.push(ch);
        i += ch.len_utf8();
    }
    Ok(out)
}

fn escape_xml_text(input: &str, out: &mut String) {
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            other => out.push(other),
        }
    }
}

fn escape_xml_attr(input: &str, out: &mut String) {
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            other => out.push(other),
        }
    }
}

fn parse_tag_name(cursor: &mut XmlCursor<'_>) -> Result<String> {
    let start = cursor.pos;
    let bytes = cursor.src.as_bytes();
    if cursor.pos >= bytes.len()
        || !(bytes[cursor.pos].is_ascii_alphabetic() || bytes[cursor.pos] == b'_')
    {
        return Err(Error::Msg("vector SVG has an invalid XML tag name".into()));
    }
    while cursor.pos < bytes.len()
        && (bytes[cursor.pos].is_ascii_alphanumeric()
            || matches!(bytes[cursor.pos], b':' | b'_' | b'-' | b'.'))
    {
        cursor.pos += 1;
    }
    Ok(cursor.src[start..cursor.pos].to_owned())
}

fn parse_xml_declaration(cursor: &mut XmlCursor<'_>) -> Result<()> {
    let rem = cursor.remaining();
    if !rem.starts_with("<?") {
        return Ok(());
    }
    // Only allow a single standard `<?xml ...?>` declaration where `xml` is immediately followed
    // by ASCII whitespace. Reject `<?xml-stylesheet ...?>`, `<?xml-model ...?>`, and all other PIs.
    let bytes = rem.as_bytes();
    if !(rem.starts_with("<?xml") && bytes.get(5).is_some_and(u8::is_ascii_whitespace)) {
        return Err(Error::Msg(
            "vector SVG processing instructions are forbidden (only standard <?xml ...?> declaration is permitted)"
                .into(),
        ));
    }
    let Some(end) = rem.find("?>") else {
        return Err(Error::Msg(
            "vector SVG has an unterminated XML declaration".into(),
        ));
    };
    let inner = rem[5..end].trim();
    let mut inner_cursor = XmlCursor::new(inner);
    let mut seen = BTreeSet::new();
    let mut saw_version = false;
    while !inner_cursor.is_eof() {
        inner_cursor.skip_ws();
        if inner_cursor.is_eof() {
            break;
        }
        let name = parse_tag_name(&mut inner_cursor)?;
        inner_cursor.skip_ws();
        if !inner_cursor.remaining().starts_with('=') {
            return Err(Error::Msg(
                "vector SVG XML declaration pseudo-attribute is missing '='".into(),
            ));
        }
        inner_cursor.pos += 1;
        inner_cursor.skip_ws();
        let quote = inner_cursor
            .remaining()
            .as_bytes()
            .first()
            .copied()
            .filter(|&b| b == b'"' || b == b'\'')
            .ok_or_else(|| {
                Error::Msg("vector SVG XML declaration pseudo-attribute must be quoted".into())
            })?;
        inner_cursor.pos += 1;
        let val_start = inner_cursor.pos;
        let ibytes = inner_cursor.src.as_bytes();
        while inner_cursor.pos < ibytes.len() && ibytes[inner_cursor.pos] != quote {
            inner_cursor.pos += 1;
        }
        if inner_cursor.pos >= ibytes.len() {
            return Err(Error::Msg(
                "vector SVG XML declaration has an unterminated quote".into(),
            ));
        }
        let value = &inner_cursor.src[val_start..inner_cursor.pos];
        inner_cursor.pos += 1;
        if !seen.insert(name.to_ascii_lowercase()) {
            return Err(Error::Msg(
                "vector SVG XML declaration has duplicate pseudo-attributes".into(),
            ));
        }
        match name.as_str() {
            "version" if value == "1.0" => saw_version = true,
            "encoding" if value.eq_ignore_ascii_case("utf-8") => {}
            "standalone" if matches!(value, "yes" | "no") => {}
            _ => {
                return Err(Error::Msg(format!(
                    "vector SVG XML declaration attribute '{name}={value:?}' is not permitted"
                )));
            }
        }
        // Require whitespace before next pseudo-attribute if not at end.
        if !inner_cursor.is_eof()
            && !inner_cursor
                .remaining()
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_whitespace)
        {
            return Err(Error::Msg(
                "vector SVG XML declaration attributes must be whitespace-separated".into(),
            ));
        }
    }
    if !saw_version {
        return Err(Error::Msg(
            "vector SVG XML declaration must specify version=\"1.0\"".into(),
        ));
    }
    cursor.pos += end + 2;
    cursor.skip_ws();
    Ok(())
}

fn parse_attributes(
    cursor: &mut XmlCursor<'_>,
    counters: &mut SanitizerCounters,
) -> Result<(Vec<(String, String)>, bool)> {
    let mut attrs = Vec::new();
    let mut seen = BTreeSet::new();
    loop {
        cursor.skip_ws();
        let rem = cursor.remaining();
        if rem.starts_with("/>") {
            cursor.pos += 2;
            return Ok((attrs, true));
        }
        if rem.starts_with('>') {
            cursor.pos += 1;
            return Ok((attrs, false));
        }
        if rem.is_empty() {
            return Err(Error::Msg(
                "vector SVG has an unterminated XML start tag".into(),
            ));
        }
        let name = parse_tag_name(cursor)?;
        cursor.skip_ws();
        if !cursor.remaining().starts_with('=') {
            return Err(Error::Msg(format!(
                "vector SVG attribute '{name}' is missing '='"
            )));
        }
        cursor.pos += 1;
        cursor.skip_ws();
        let quote = cursor
            .remaining()
            .as_bytes()
            .first()
            .copied()
            .filter(|&b| b == b'"' || b == b'\'')
            .ok_or_else(|| {
                Error::Msg(format!(
                    "vector SVG attribute '{name}' value must be quoted"
                ))
            })?;
        cursor.pos += 1;
        let val_start = cursor.pos;
        let bytes = cursor.src.as_bytes();
        while cursor.pos < bytes.len() && bytes[cursor.pos] != quote {
            cursor.pos += 1;
        }
        if cursor.pos >= bytes.len() {
            return Err(Error::Msg(format!(
                "vector SVG attribute '{name}' has an unterminated quote"
            )));
        }
        let raw_value = &cursor.src[val_start..cursor.pos];
        cursor.pos += 1;
        if raw_value.len() > MAX_SVG_ATTRIBUTE_VALUE_BYTES {
            return Err(Error::Msg(format!(
                "vector SVG attribute '{name}' exceeds {MAX_SVG_ATTRIBUTE_VALUE_BYTES}-byte cap"
            )));
        }
        counters.total_attrs = counters
            .total_attrs
            .checked_add(1)
            .ok_or_else(|| Error::Msg("vector SVG attribute count overflow".into()))?;
        if counters.total_attrs > MAX_SVG_ATTRIBUTES {
            return Err(Error::Msg(format!(
                "vector SVG exceeds the {MAX_SVG_ATTRIBUTES}-attribute budget"
            )));
        }
        if !seen.insert(name.to_ascii_lowercase()) {
            return Err(Error::Msg(format!(
                "vector SVG contains duplicate attribute '{name}'"
            )));
        }
        let decoded = decode_xml_entities(raw_value)?;
        if decoded.len() > MAX_SVG_ATTRIBUTE_VALUE_BYTES {
            return Err(Error::Msg(format!(
                "vector SVG attribute '{name}' decoded value exceeds {MAX_SVG_ATTRIBUTE_VALUE_BYTES}-byte cap"
            )));
        }
        counters.total_attr_value_bytes = counters
            .total_attr_value_bytes
            .checked_add(decoded.len())
            .ok_or_else(|| Error::Msg("vector SVG attribute byte count overflow".into()))?;
        if counters.total_attr_value_bytes > MAX_SVG_ATTRIBUTE_VALUE_BYTES_TOTAL {
            return Err(Error::Msg(format!(
                "vector SVG exceeds the {MAX_SVG_ATTRIBUTE_VALUE_BYTES_TOTAL}-byte total attribute budget"
            )));
        }
        attrs.push((name, decoded));
    }
}

#[derive(Debug)]
struct GraphNode {
    name: String,
    children: Vec<usize>,
    href_target: Option<String>,
    path_data_bytes: usize,
    path_commands: usize,
    path_numbers: usize,
}

#[derive(Clone, Copy, Debug)]
struct ExpansionCost {
    elements: usize,
    max_depth: usize,
    path_data_bytes: usize,
    path_commands: usize,
    path_numbers: usize,
}

impl ExpansionCost {
    fn nested(mut self) -> Result<Self> {
        self.max_depth = self
            .max_depth
            .checked_add(1)
            .ok_or_else(|| Error::Msg("vector SVG expanded depth overflow".into()))?;
        if self.max_depth > MAX_SVG_DEPTH {
            return Err(Error::Msg(format!(
                "vector SVG <use> expansion exceeds the {MAX_SVG_DEPTH}-level nesting depth budget"
            )));
        }
        Ok(self)
    }

    fn checked_add(self, other: Self) -> Result<Self> {
        let elements = self
            .elements
            .checked_add(other.elements)
            .ok_or_else(|| Error::Msg("vector SVG expansion count overflow".into()))?;
        let path_data_bytes = self
            .path_data_bytes
            .checked_add(other.path_data_bytes)
            .ok_or_else(|| Error::Msg("vector SVG expanded path bytes overflow".into()))?;
        let path_commands = self
            .path_commands
            .checked_add(other.path_commands)
            .ok_or_else(|| Error::Msg("vector SVG expanded path command count overflow".into()))?;
        let path_numbers = self
            .path_numbers
            .checked_add(other.path_numbers)
            .ok_or_else(|| Error::Msg("vector SVG expanded path number count overflow".into()))?;
        if elements > MAX_SVG_ELEMENTS {
            return Err(Error::Msg(format!(
                "vector SVG <use> expansion exceeds the {MAX_SVG_ELEMENTS}-element budget"
            )));
        }
        if path_data_bytes > MAX_SVG_PATH_DATA_BYTES
            || path_commands > MAX_SVG_PATH_COMMANDS
            || path_numbers > MAX_SVG_PATH_NUMBERS
        {
            return Err(Error::Msg(
                "vector SVG <use> expansion exceeds the path-work budget".into(),
            ));
        }
        Ok(Self {
            elements,
            max_depth: self.max_depth.max(other.max_depth),
            path_data_bytes,
            path_commands,
            path_numbers,
        })
    }
}

fn compute_expansion_cost(
    node: usize,
    nodes: &[GraphNode],
    ids: &BTreeMap<String, usize>,
    visiting: &mut BTreeSet<usize>,
    memo: &mut BTreeMap<usize, ExpansionCost>,
) -> Result<ExpansionCost> {
    if let Some(cost) = memo.get(&node) {
        return Ok(*cost);
    }
    if !visiting.insert(node) {
        return Err(Error::Msg(
            "vector SVG local <use>/resource graph contains a cycle".into(),
        ));
    }
    let mut cost = ExpansionCost {
        elements: 1,
        max_depth: 1,
        path_data_bytes: nodes[node].path_data_bytes,
        path_commands: nodes[node].path_commands,
        path_numbers: nodes[node].path_numbers,
    };
    for &child in &nodes[node].children {
        cost =
            cost.checked_add(compute_expansion_cost(child, nodes, ids, visiting, memo)?.nested()?)?;
    }
    if let Some(target) = &nodes[node].href_target {
        let target_node = *ids.get(target).ok_or_else(|| {
            Error::Msg(format!(
                "vector SVG local reference '#{target}' is unresolved"
            ))
        })?;
        if nodes[node].name == "use" {
            if !matches!(
                nodes[target_node].name.as_str(),
                "svg"
                    | "g"
                    | "symbol"
                    | "path"
                    | "rect"
                    | "circle"
                    | "ellipse"
                    | "line"
                    | "polyline"
                    | "polygon"
            ) {
                return Err(Error::Msg(format!(
                    "vector SVG <use> target '#{target}' (<{}>) is not permitted",
                    nodes[target_node].name
                )));
            }
        } else if !matches!(
            nodes[node].name.as_str(),
            "linearGradient" | "radialGradient"
        ) || !matches!(
            nodes[target_node].name.as_str(),
            "linearGradient" | "radialGradient"
        ) {
            return Err(Error::Msg(format!(
                "vector SVG href target '#{target}' has an unsupported resource type"
            )));
        }
        cost = cost.checked_add(
            compute_expansion_cost(target_node, nodes, ids, visiting, memo)?.nested()?,
        )?;
    }
    visiting.remove(&node);
    memo.insert(node, cost);
    Ok(cost)
}

fn visit_reference_graph(
    node: usize,
    edges: &[Vec<usize>],
    visiting: &mut BTreeSet<usize>,
    complete: &mut BTreeSet<usize>,
) -> Result<()> {
    if complete.contains(&node) {
        return Ok(());
    }
    if !visiting.insert(node) {
        return Err(Error::Msg(
            "vector SVG resource reference graph contains a cycle".into(),
        ));
    }
    for &target in &edges[node] {
        visit_reference_graph(target, edges, visiting, complete)?;
    }
    visiting.remove(&node);
    complete.insert(node);
    Ok(())
}

fn validate_filter_primitive(
    name: &str,
    attrs: &[(String, String)],
    filter_id: Option<&str>,
    parent_name: Option<&str>,
    filter_results: &mut BTreeMap<String, BTreeSet<String>>,
    counters: &mut SanitizerCounters,
) -> Result<()> {
    if !matches!(
        name,
        "feGaussianBlur" | "feOffset" | "feMerge" | "feMergeNode"
    ) {
        return Ok(());
    }
    let filter_key = filter_id.ok_or_else(|| {
        Error::Msg(format!(
            "vector SVG filter primitive <{name}> must appear inside a <filter>"
        ))
    })?;
    if name == "feMergeNode" && parent_name != Some("feMerge") {
        return Err(Error::Msg(
            "vector SVG <feMergeNode> must appear directly inside <feMerge>".into(),
        ));
    }
    counters.filter_primitives = counters
        .filter_primitives
        .checked_add(1)
        .ok_or_else(|| Error::Msg("vector SVG filter primitive count overflow".into()))?;
    if counters.filter_primitives > MAX_SVG_FILTER_PRIMITIVES {
        return Err(Error::Msg(format!(
            "vector SVG exceeds the {MAX_SVG_FILTER_PRIMITIVES}-filter-primitive budget"
        )));
    }
    let results = filter_results.entry(filter_key.to_owned()).or_default();
    if let Some((_, input)) = attrs.iter().find(|(k, _)| k == "in") {
        if !matches!(input.as_str(), "SourceGraphic" | "SourceAlpha") && !results.contains(input) {
            return Err(Error::Msg(format!(
                "vector SVG <{name}> input '{input}' must reference SourceGraphic, SourceAlpha, or an earlier filter result"
            )));
        }
    }
    if let Some((_, result)) = attrs.iter().find(|(k, _)| k == "result") {
        if !matches!(name, "feGaussianBlur" | "feOffset") {
            return Err(Error::Msg(format!(
                "vector SVG <{name}> cannot name a filter result"
            )));
        }
        validate_static_svg_id(result, "filter result")?;
        if !results.insert(result.clone()) {
            return Err(Error::Msg(format!(
                "vector SVG filter result '{result}' is duplicated"
            )));
        }
    }
    Ok(())
}

struct ParseFrame {
    node_idx: usize,
    node: SvgNode,
    hidden: bool,
    filter_key: Option<String>,
    style_text: String,
}

struct ValidatedAttributes {
    attrs: Vec<(String, String)>,
    element_id: Option<String>,
    href_target: Option<String>,
    path_bytes: usize,
    path_cmds: usize,
    path_nums: usize,
}

#[allow(clippy::too_many_arguments)]
fn validate_element_attributes(
    element: &str,
    raw_attrs: Vec<(String, String)>,
    is_root: bool,
    node_idx: usize,
    hidden: &mut bool,
    root_xlink_bound: &mut bool,
    xlink_href_seen: &mut bool,
    counters: &mut SanitizerCounters,
    typed_refs: &mut Vec<TypedReference>,
) -> Result<ValidatedAttributes> {
    let mut clean_attrs = Vec::with_capacity(raw_attrs.len());
    let mut element_id = None;
    let mut href_target = None;
    let mut node_path_bytes = 0usize;
    let mut node_path_cmds = 0usize;
    let mut node_path_nums = 0usize;

    // First check root namespace bindings and hidden flag before validating remaining attributes.
    if is_root {
        for (key, value) in &raw_attrs {
            if key == "xmlns:xlink" {
                if value != XLINK_NAMESPACE {
                    return Err(Error::Msg(format!(
                        "vector SVG root xmlns:xlink must equal {XLINK_NAMESPACE:?}"
                    )));
                }
                *root_xlink_bound = true;
            }
        }
    }
    for (key, value) in &raw_attrs {
        if key == "display" && value.trim() == "none" {
            *hidden = true;
        }
    }

    for (key, value) in raw_attrs {
        if is_forbidden_attribute_name(&key) {
            return Err(Error::Msg(format!(
                "vector SVG active attribute '{key}' is forbidden"
            )));
        }
        if key == "xmlns" {
            if !is_root || element != "svg" || value != SVG_NAMESPACE {
                return Err(Error::Msg(format!(
                    "vector SVG xmlns attribute is only permitted on root <svg> with {SVG_NAMESPACE:?}"
                )));
            }
            clean_attrs.push((key, value));
            continue;
        }
        if key == "xmlns:xlink" {
            if !is_root || element != "svg" || value != XLINK_NAMESPACE {
                return Err(Error::Msg(format!(
                    "vector SVG xmlns:xlink is only permitted on root <svg> with {XLINK_NAMESPACE:?}"
                )));
            }
            clean_attrs.push((key, value));
            continue;
        }
        if key.starts_with("xmlns:")
            || (key.contains(':') && !matches!(key.as_str(), "xml:space" | "xlink:href"))
        {
            return Err(Error::Msg(format!(
                "vector SVG namespaced attribute '{key}' is forbidden"
            )));
        }
        if !is_allowed_static_attribute(element, &key, *hidden) {
            return Err(Error::Msg(format!(
                "vector SVG attribute '{key}' is not permitted on <{element}>"
            )));
        }
        if contains_external_or_active_tokens(&value) {
            return Err(Error::Msg(format!(
                "vector SVG attribute '{key}' contains forbidden external, escape, or active content"
            )));
        }

        match key.as_str() {
            "id" => {
                validate_static_svg_id(&value, "element id")?;
                element_id = Some(value.clone());
                clean_attrs.push((key, value));
            }
            "display" => {
                let v = value.trim();
                if !matches!(
                    v,
                    "inline" | "block" | "none" | "inherit" | "initial" | "unset"
                ) {
                    return Err(Error::Msg(format!(
                        "vector SVG display value {value:?} is unsupported"
                    )));
                }
                if v == "none" {
                    *hidden = true;
                }
                clean_attrs.push((key, v.to_owned()));
            }
            "visibility" => {
                let v = value.trim();
                if !matches!(
                    v,
                    "visible" | "hidden" | "collapse" | "inherit" | "initial" | "unset"
                ) {
                    return Err(Error::Msg(format!(
                        "vector SVG visibility value {value:?} is unsupported"
                    )));
                }
                clean_attrs.push((key, v.to_owned()));
            }
            "vector-effect" => {
                let v = value.trim();
                if !matches!(v, "none" | "non-scaling-stroke") {
                    return Err(Error::Msg(
                        "vector SVG vector-effect value is unsupported".into(),
                    ));
                }
                clean_attrs.push((key, v.to_owned()));
            }
            "xml:space" => {
                if !matches!(value.as_str(), "default" | "preserve") {
                    return Err(Error::Msg(
                        "vector SVG xml:space value must be 'default' or 'preserve'".into(),
                    ));
                }
                clean_attrs.push((key, value));
            }
            "version" => {
                if !is_root || !matches!(value.as_str(), "1.0" | "1.1" | "2.0") {
                    return Err(Error::Msg(
                        "vector SVG version attribute is unsupported".into(),
                    ));
                }
                clean_attrs.push((key, value));
            }
            "baseProfile" => {
                if !is_root || value != "full" {
                    return Err(Error::Msg(
                        "vector SVG baseProfile must be exact root 'full'".into(),
                    ));
                }
                clean_attrs.push((key, value));
            }
            "type" if element == "style" => {
                if value != "text/css" {
                    return Err(Error::Msg(
                        "vector SVG <style> type attribute must be 'text/css'".into(),
                    ));
                }
                clean_attrs.push((key, value));
            }
            "style" => {
                let cleaned = validate_inline_style(&value, node_idx, counters, typed_refs)?;
                if cleaned
                    .split(';')
                    .filter_map(|d| d.split_once(':'))
                    .any(|(p, v)| p == "display" && v.eq_ignore_ascii_case("none"))
                {
                    *hidden = true;
                }
                if !cleaned.is_empty() {
                    clean_attrs.push(("style".to_owned(), cleaned));
                }
            }
            "href" | "xlink:href" => {
                if !matches!(element, "use" | "linearGradient" | "radialGradient") {
                    return Err(Error::Msg(format!(
                        "vector SVG attribute '{key}' is not allowed on <{element}>"
                    )));
                }
                if key == "xlink:href" {
                    *xlink_href_seen = true;
                    if !*root_xlink_bound {
                        return Err(Error::Msg(
                            "vector SVG xlink:href requires root xmlns:xlink=\"http://www.w3.org/1999/xlink\""
                                .into(),
                        ));
                    }
                }
                if href_target.is_some() {
                    return Err(Error::Msg(format!(
                        "vector SVG <{element}> has ambiguous href and xlink:href attributes"
                    )));
                }
                let target = local_href_target(&value)?;
                record_reference_count(counters)?;
                href_target = Some(target.to_owned());
                clean_attrs.push((key, format!("#{target}")));
            }
            "fill" | "stroke" | "clip-path" | "filter" => {
                if let Some(target) = validate_css_value_functions(&key, &value)? {
                    let kind = match key.as_str() {
                        "fill" | "stroke" => StaticReferenceKind::Paint,
                        "clip-path" => StaticReferenceKind::ClipPath,
                        "filter" => StaticReferenceKind::Filter,
                        _ => unreachable!(),
                    };
                    record_reference_count(counters)?;
                    typed_refs.push(TypedReference {
                        source_node: node_idx,
                        target,
                        kind,
                    });
                }
                clean_attrs.push((key, value));
            }
            "d" if element == "path" => {
                counters.path_data_bytes = counters
                    .path_data_bytes
                    .checked_add(value.len())
                    .ok_or_else(|| Error::Msg("vector SVG path bytes overflow".into()))?;
                if counters.path_data_bytes > MAX_SVG_PATH_DATA_BYTES {
                    return Err(Error::Msg(
                        "vector SVG exceeds the path-data byte budget".into(),
                    ));
                }
                let (cmds, nums) = scan_path_data_budget(&value)?;
                counters.path_commands = counters
                    .path_commands
                    .checked_add(cmds)
                    .ok_or_else(|| Error::Msg("vector SVG path command count overflow".into()))?;
                counters.path_numbers = counters
                    .path_numbers
                    .checked_add(nums)
                    .ok_or_else(|| Error::Msg("vector SVG path number count overflow".into()))?;
                if counters.path_commands > MAX_SVG_PATH_COMMANDS {
                    return Err(Error::Msg(
                        "vector SVG exceeds the path-command budget".into(),
                    ));
                }
                if counters.path_numbers > MAX_SVG_PATH_NUMBERS {
                    return Err(Error::Msg(
                        "vector SVG exceeds the path-number budget".into(),
                    ));
                }
                node_path_bytes = value.len();
                node_path_cmds = cmds;
                node_path_nums = nums;
                clean_attrs.push((key, value));
            }
            "stdDeviation" if element == "feGaussianBlur" => {
                let vals = parse_number_list(&value, "filter stdDeviation")?;
                if vals.len() > 2
                    || vals
                        .iter()
                        .any(|v| !(0.0..=MAX_SVG_FILTER_BLUR_SIGMA).contains(v))
                {
                    return Err(Error::Msg(format!(
                        "vector SVG <feGaussianBlur> stdDeviation must be 1..=2 values in 0..={MAX_SVG_FILTER_BLUR_SIGMA}"
                    )));
                }
                clean_attrs.push((key, value));
            }
            "dx" | "dy" if element == "feOffset" => {
                let vals = parse_number_list(&value, &key)?;
                if vals.len() != 1 {
                    return Err(Error::Msg(format!(
                        "vector SVG <feOffset> {key} must be a single finite number"
                    )));
                }
                clean_attrs.push((key, value));
            }
            "width" | "height" if element == "filter" => {
                let trimmed = value.trim();
                if let Some(pct) = trimmed.strip_suffix('%') {
                    let vals = parse_number_list(pct, &key)?;
                    if vals.len() != 1 || !(0.0..=200.0).contains(&vals[0]) {
                        return Err(Error::Msg(format!(
                            "vector SVG <filter> {key} percentage must be in 0%..=200%"
                        )));
                    }
                } else {
                    let vals = parse_number_list(trimmed, &key)?;
                    if vals.len() != 1
                        || vals[0] <= 0.0
                        || vals[0] > f64::from(MAX_PREVIEW_DIMENSION)
                    {
                        return Err(Error::Msg(format!(
                            "vector SVG <filter> {key} exceeds bounded dimension budget"
                        )));
                    }
                }
                clean_attrs.push((key, value));
            }
            "offset" if element == "stop" => {
                let trimmed = value.trim();
                let (num_str, max_v) = if let Some(pct) = trimmed.strip_suffix('%') {
                    (pct, 100.0)
                } else {
                    (trimmed, 1.0)
                };
                let vals = parse_number_list(num_str, "gradient stop offset")?;
                if vals.len() != 1 || !(0.0..=max_v).contains(&vals[0]) {
                    return Err(Error::Msg(
                        "vector SVG gradient stop offset is outside its range".into(),
                    ));
                }
                clean_attrs.push((key, value));
            }
            "stop-opacity" if element == "stop" => {
                let vals = parse_number_list(&value, "stop-opacity")?;
                if vals.len() != 1 || !(0.0..=1.0).contains(&vals[0]) {
                    return Err(Error::Msg(
                        "vector SVG stop-opacity must be between 0 and 1".into(),
                    ));
                }
                clean_attrs.push((key, value));
            }
            _ => {
                let compact_lower: String = value
                    .chars()
                    .filter(|c| !c.is_ascii_whitespace())
                    .collect::<String>()
                    .to_ascii_lowercase();
                if compact_lower.contains("url(") {
                    return Err(Error::Msg(format!(
                        "vector SVG attribute '{key}' does not permit url(...) values"
                    )));
                }
                clean_attrs.push((key, value));
            }
        }
    }

    if element == "use" && href_target.is_none() {
        return Err(Error::Msg(
            "vector SVG <use> requires a local href or xlink:href fragment".into(),
        ));
    }

    Ok(ValidatedAttributes {
        attrs: clean_attrs,
        element_id,
        href_target,
        path_bytes: node_path_bytes,
        path_cmds: node_path_cmds,
        path_nums: node_path_nums,
    })
}

fn parse_and_sanitize_document(input: &str) -> Result<Vec<SvgNode>> {
    let mut cursor = XmlCursor::new(input);
    cursor.skip_ws();
    parse_xml_declaration(&mut cursor)?;

    let mut roots: Vec<SvgNode> = Vec::new();
    let mut stack: Vec<ParseFrame> = Vec::new();
    let mut graph_nodes: Vec<GraphNode> = Vec::new();
    let mut ids: BTreeMap<String, usize> = BTreeMap::new();
    let mut typed_refs: Vec<TypedReference> = Vec::new();
    let mut filter_results: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut counters = SanitizerCounters::default();
    let mut root_idx: Option<usize> = None;
    let mut root_xlink_bound = false;
    let mut xlink_href_seen = false;

    while !cursor.is_eof() {
        let rem = cursor.remaining();
        if !rem.starts_with('<') {
            let next_lt = rem.find('<').unwrap_or(rem.len());
            let text_raw = &rem[..next_lt];
            cursor.pos += next_lt;
            let decoded = decode_xml_entities(text_raw)?;
            let Some(parent) = stack.last_mut() else {
                if !decoded.trim().is_empty() {
                    return Err(Error::Msg(
                        "vector SVG contains content outside the <svg> root".into(),
                    ));
                }
                continue;
            };
            if parent.node.name == "style" {
                parent.style_text.push_str(&decoded);
            } else if matches!(parent.node.name.as_str(), "title" | "desc" | "metadata")
                || parent.hidden
            {
                if !decoded.trim().is_empty()
                    && matches!(parent.node.name.as_str(), "title" | "desc")
                {
                    parent.node.children.push(SvgChild::Text(decoded));
                }
            } else if !decoded.trim().is_empty() {
                return Err(Error::Msg(
                    "vector SVG visible text content is not permitted by the static sanitizer"
                        .into(),
                ));
            }
            continue;
        }

        // Comment
        if let Some(after_comment) = rem.strip_prefix("<!--") {
            let Some(end) = after_comment.find("-->") else {
                return Err(Error::Msg(
                    "vector SVG contains an unterminated XML comment".into(),
                ));
            };
            cursor.pos += 4 + end + 3;
            continue;
        }

        // CDATA — permitted only inside <style> (matching sceneworks-static-svg-v2)
        if let Some(after_cdata) = rem.strip_prefix("<![CDATA[") {
            let Some(end) = after_cdata.find("]]>") else {
                return Err(Error::Msg(
                    "vector SVG contains an unterminated CDATA section".into(),
                ));
            };
            let cdata = &after_cdata[..end];
            validate_xml_chars(cdata, "vector SVG CDATA")?;
            cursor.pos += 9 + end + 3;
            let Some(parent) = stack.last_mut() else {
                return Err(Error::Msg(
                    "vector SVG contains CDATA outside a <style> element".into(),
                ));
            };
            if parent.node.name != "style" {
                return Err(Error::Msg(
                    "vector SVG contains CDATA outside a <style> element".into(),
                ));
            }
            parent.style_text.push_str(cdata);
            continue;
        }

        // Declarations (DOCTYPE / ENTITY / etc.) and any subsequent processing instructions are forbidden.
        if rem.starts_with("<!") || rem.starts_with("<?") {
            return Err(Error::Msg(
                "vector SVG declarations and processing instructions are forbidden".into(),
            ));
        }

        // Closing tag
        if rem.starts_with("</") {
            cursor.pos += 2;
            let name = parse_tag_name(&mut cursor)?;
            cursor.skip_ws();
            if !cursor.remaining().starts_with('>') {
                return Err(Error::Msg(format!(
                    "vector SVG closing tag </{name}> is malformed"
                )));
            }
            cursor.pos += 1;
            let Some(mut frame) = stack.pop() else {
                return Err(Error::Msg(format!(
                    "vector SVG has unexpected closing tag </{name}>"
                )));
            };
            if frame.node.name != name {
                return Err(Error::Msg(format!(
                    "vector SVG closes </{name}> while <{}> is open",
                    frame.node.name
                )));
            }
            if frame.node.name == "style" {
                let clean_css = validate_stylesheet(
                    &frame.style_text,
                    frame.node_idx,
                    &mut counters,
                    &mut typed_refs,
                )?;
                if !clean_css.is_empty() {
                    frame.node.children.push(SvgChild::Text(clean_css));
                }
            }
            if let Some(parent) = stack.last_mut() {
                parent.node.children.push(SvgChild::Element(frame.node));
            } else {
                roots.push(frame.node);
            }
            continue;
        }

        // Start or self-closing tag
        cursor.pos += 1;
        let name = parse_tag_name(&mut cursor)?;
        let (raw_attrs, self_closing) = parse_attributes(&mut cursor, &mut counters)?;

        if stack.len() >= MAX_SVG_DEPTH {
            return Err(Error::Msg(format!(
                "vector SVG exceeds the {MAX_SVG_DEPTH}-level nesting depth budget"
            )));
        }

        let is_root = stack.is_empty();
        if is_root && (root_idx.is_some() || name != "svg") {
            return Err(Error::Msg(
                "vector SVG must contain exactly one <svg> root element".into(),
            ));
        }

        let parent_hidden = stack.last().is_some_and(|f| f.hidden);
        let parent_name = stack.last().map(|f| f.node.name.as_str());
        let mut hidden = parent_hidden;
        if !is_allowed_static_element(&name, hidden) {
            return Err(Error::Msg(format!(
                "vector SVG element <{name}> is active or unsupported by the static SVG allowlist"
            )));
        }

        if graph_nodes.len() >= MAX_SVG_ELEMENTS {
            return Err(Error::Msg(format!(
                "vector SVG exceeds the {MAX_SVG_ELEMENTS}-element budget"
            )));
        }

        let node_idx = graph_nodes.len();
        let ValidatedAttributes {
            attrs: clean_attrs,
            element_id,
            href_target,
            path_bytes,
            path_cmds,
            path_nums,
        } = validate_element_attributes(
            &name,
            raw_attrs,
            is_root,
            node_idx,
            &mut hidden,
            &mut root_xlink_bound,
            &mut xlink_href_seen,
            &mut counters,
            &mut typed_refs,
        )?;

        if is_root && hidden {
            return Err(Error::Msg(
                "vector SVG root display:none is unsupported".into(),
            ));
        }

        if name == "style" && !matches!(parent_name, Some("svg" | "defs")) {
            return Err(Error::Msg(
                "vector SVG <style> element must be directly under <svg> or <defs>".into(),
            ));
        }

        if matches!(
            name.as_str(),
            "clipPath" | "linearGradient" | "radialGradient" | "filter" | "symbol"
        ) {
            counters.resources = counters
                .resources
                .checked_add(1)
                .ok_or_else(|| Error::Msg("vector SVG resource count overflow".into()))?;
            if counters.resources > MAX_SVG_RESOURCES {
                return Err(Error::Msg(format!(
                    "vector SVG exceeds the {MAX_SVG_RESOURCES}-resource budget"
                )));
            }
        }

        if name == "stop" {
            counters.gradient_stops = counters
                .gradient_stops
                .checked_add(1)
                .ok_or_else(|| Error::Msg("vector SVG gradient stop count overflow".into()))?;
            if counters.gradient_stops > MAX_SVG_GRADIENT_STOPS {
                return Err(Error::Msg(format!(
                    "vector SVG exceeds the {MAX_SVG_GRADIENT_STOPS}-gradient-stop budget"
                )));
            }
        }

        let filter_key = if name == "filter" {
            Some(
                element_id
                    .clone()
                    .unwrap_or_else(|| format!("__anon_filter_{node_idx}")),
            )
        } else {
            stack.last().and_then(|f| f.filter_key.clone())
        };
        validate_filter_primitive(
            &name,
            &clean_attrs,
            filter_key.as_deref(),
            parent_name,
            &mut filter_results,
            &mut counters,
        )?;

        if let Some(id) = element_id {
            if ids.insert(id.clone(), node_idx).is_some() {
                return Err(Error::Msg(format!(
                    "vector SVG element id '{id}' is duplicated"
                )));
            }
        }

        if let Some(parent) = stack.last() {
            graph_nodes[parent.node_idx].children.push(node_idx);
        } else {
            root_idx = Some(node_idx);
        }
        graph_nodes.push(GraphNode {
            name: name.clone(),
            children: Vec::new(),
            href_target,
            path_data_bytes: path_bytes,
            path_commands: path_cmds,
            path_numbers: path_nums,
        });

        let node = SvgNode {
            name: name.clone(),
            attrs: clean_attrs,
            children: Vec::new(),
        };

        if self_closing {
            if let Some(parent) = stack.last_mut() {
                parent.node.children.push(SvgChild::Element(node));
            } else {
                roots.push(node);
            }
        } else {
            stack.push(ParseFrame {
                node_idx,
                node,
                hidden,
                filter_key,
                style_text: String::new(),
            });
        }
    }

    if let Some(unclosed) = stack.last() {
        return Err(Error::Msg(format!(
            "vector SVG is truncated: unclosed <{}> element",
            unclosed.node.name
        )));
    }
    let root = root_idx.ok_or_else(|| Error::Msg("vector SVG has no root <svg> element".into()))?;
    if xlink_href_seen && !root_xlink_bound {
        return Err(Error::Msg(
            "vector SVG xlink:href requires root xmlns:xlink=\"http://www.w3.org/1999/xlink\""
                .into(),
        ));
    }
    if graph_nodes
        .iter()
        .any(|node| node.name == "use" && !node.children.is_empty())
    {
        return Err(Error::Msg("vector SVG <use> element must be empty".into()));
    }

    // Build combined reference graph (containment + href + typed CSS/attribute url(#id)),
    // verify every target resolves to the right resource type, reject cycles, and enforce
    // bounded <use> expansion cost.
    let mut ref_edges: Vec<Vec<usize>> = graph_nodes.iter().map(|n| n.children.clone()).collect();
    for (src, gnode) in graph_nodes.iter().enumerate() {
        if let Some(target) = &gnode.href_target {
            let target_idx = *ids.get(target).ok_or_else(|| {
                Error::Msg(format!(
                    "vector SVG local reference '#{target}' is unresolved"
                ))
            })?;
            ref_edges[src].push(target_idx);
        }
    }
    for tref in &typed_refs {
        let target_idx = *ids.get(&tref.target).ok_or_else(|| {
            Error::Msg(format!(
                "vector SVG resource reference '#{}' is unresolved",
                tref.target
            ))
        })?;
        let target_name = graph_nodes[target_idx].name.as_str();
        let valid_type = match tref.kind {
            StaticReferenceKind::Paint => {
                matches!(target_name, "linearGradient" | "radialGradient")
            }
            StaticReferenceKind::ClipPath => target_name == "clipPath",
            StaticReferenceKind::Filter => target_name == "filter",
        };
        if !valid_type {
            return Err(Error::Msg(format!(
                "vector SVG resource reference '#{}' points to wrong element type <{target_name}>",
                tref.target
            )));
        }
        ref_edges[tref.source_node].push(target_idx);
    }

    visit_reference_graph(root, &ref_edges, &mut BTreeSet::new(), &mut BTreeSet::new())?;
    compute_expansion_cost(
        root,
        &graph_nodes,
        &ids,
        &mut BTreeSet::new(),
        &mut BTreeMap::new(),
    )?;

    Ok(roots)
}

fn parse_positive_dimension(raw: &str) -> Option<f64> {
    let trimmed = raw.trim().strip_suffix("px").unwrap_or(raw.trim()).trim();
    let value: f64 = trimmed.parse().ok()?;
    if value.is_finite() && value > 0.0 && value <= MAX_SVG_COORDINATE_MAGNITUDE {
        Some(value)
    } else {
        None
    }
}

fn parse_viewbox(raw: &str) -> Option<[f64; 4]> {
    let nums: Vec<f64> = raw
        .split(|c: char| c.is_ascii_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f64>().ok())
        .collect::<Option<Vec<_>>>()?;
    if nums.len() != 4 {
        return None;
    }
    let [x, y, w, h] = [nums[0], nums[1], nums[2], nums[3]];
    if x.is_finite()
        && y.is_finite()
        && w.is_finite()
        && h.is_finite()
        && x.abs() <= MAX_SVG_COORDINATE_MAGNITUDE
        && y.abs() <= MAX_SVG_COORDINATE_MAGNITUDE
        && w > 0.0
        && h > 0.0
        && w <= MAX_SVG_COORDINATE_MAGNITUDE
        && h <= MAX_SVG_COORDINATE_MAGNITUDE
    {
        Some([x, y, w, h])
    } else {
        None
    }
}

fn canonicalize_root(root: &SvgNode) -> Result<(u32, u32, SvgNode)> {
    let attr = |key: &str| {
        root.attrs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    };
    let viewbox = match attr("viewBox") {
        Some(raw) => Some(parse_viewbox(raw).ok_or_else(|| {
            Error::Msg(format!("vector SVG has invalid viewBox attribute: {raw:?}"))
        })?),
        None => None,
    };
    let width_f = match attr("width") {
        Some(raw) => Some(parse_positive_dimension(raw).ok_or_else(|| {
            Error::Msg(format!("vector SVG has invalid width attribute: {raw:?}"))
        })?),
        None => None,
    };
    let height_f = match attr("height") {
        Some(raw) => Some(parse_positive_dimension(raw).ok_or_else(|| {
            Error::Msg(format!("vector SVG has invalid height attribute: {raw:?}"))
        })?),
        None => None,
    };

    let (w_f, h_f) = match (width_f, height_f, viewbox) {
        (Some(w), Some(h), _) => (w, h),
        (None, None, Some([_, _, vw, vh])) => (vw, vh),
        (Some(w), None, Some([_, _, vw, vh])) => (w, (w * vh / vw).max(1.0)),
        (None, Some(h), Some([_, _, vw, vh])) => ((h * vw / vh).max(1.0), h),
        _ => {
            return Err(Error::Msg(
                "vector SVG root must declare positive width/height or a valid viewBox".into(),
            ));
        }
    };
    let width = (w_f.round() as u32).max(1);
    let height = (h_f.round() as u32).max(1);

    let mut attrs = Vec::with_capacity(root.attrs.len() + 3);
    attrs.push(("xmlns".to_owned(), SVG_NAMESPACE.to_owned()));
    attrs.push(("width".to_owned(), width.to_string()));
    attrs.push(("height".to_owned(), height.to_string()));
    if let Some([vx, vy, vw, vh]) = viewbox {
        attrs.push(("viewBox".to_owned(), format!("{vx} {vy} {vw} {vh}")));
    } else {
        attrs.push(("viewBox".to_owned(), format!("0 0 {width} {height}")));
    }
    for (k, v) in &root.attrs {
        if matches!(k.as_str(), "xmlns" | "width" | "height" | "viewBox") {
            continue;
        }
        attrs.push((k.clone(), v.clone()));
    }

    Ok((
        width,
        height,
        SvgNode {
            name: root.name.clone(),
            attrs,
            children: root.children.clone(),
        },
    ))
}

fn count_elements(node: &SvgNode) -> usize {
    1 + node
        .children
        .iter()
        .map(|child| match child {
            SvgChild::Element(el) => count_elements(el),
            SvgChild::Text(_) => 0,
        })
        .sum::<usize>()
}

fn serialize_node(node: &SvgNode, out: &mut String) {
    out.push('<');
    out.push_str(&node.name);
    for (k, v) in &node.attrs {
        out.push(' ');
        out.push_str(k);
        out.push_str("=\"");
        escape_xml_attr(v, out);
        out.push('"');
    }
    if node.children.is_empty() {
        out.push_str("/>");
        return;
    }
    out.push('>');
    for child in &node.children {
        match child {
            SvgChild::Element(el) => serialize_node(el, out),
            SvgChild::Text(text) => escape_xml_text(text, out),
        }
    }
    out.push_str("</");
    out.push_str(&node.name);
    out.push('>');
}

// -------------------------------------------------------------------------------------------------
// Lightweight Pure-Rust Derived RGB Preview Rasterizer
// -------------------------------------------------------------------------------------------------

/// Render a bounded RGB preview [`Image`] of a [`CanonicalSvg`] onto a white canvas.
///
/// Supports basic static SVG primitives (`rect`, `circle`, `ellipse`, `line`, `polygon`,
/// `polyline`, and `path` commands) with viewBox-to-pixel projection so every [`VectorOutput`]
/// carries a real aspect-preserving RGB preview without pulling in a C/native rasterizer.
pub fn render_svg_preview(canonical: &CanonicalSvg, max_dimension: u32) -> Result<Image> {
    if max_dimension == 0 || max_dimension > MAX_PREVIEW_DIMENSION {
        return Err(Error::Msg(format!(
            "vector preview max_dimension must be in 1..={MAX_PREVIEW_DIMENSION}, got {max_dimension}"
        )));
    }
    let src_w = canonical.width.max(1) as f64;
    let src_h = canonical.height.max(1) as f64;
    let scale = (f64::from(max_dimension) / src_w.max(src_h)).min(1.0);
    let out_w = ((src_w * scale).round() as u32).clamp(1, max_dimension);
    let out_h = ((src_h * scale).round() as u32).clamp(1, max_dimension);

    let mut pixels = vec![255u8; (out_w as usize) * (out_h as usize) * 3];
    let roots = parse_and_sanitize_document(&canonical.svg)?;
    if let Some(root) = roots.first() {
        let viewbox = root
            .attrs
            .iter()
            .find(|(k, _)| k == "viewBox")
            .and_then(|(_, v)| parse_viewbox(v))
            .unwrap_or([0.0, 0.0, src_w, src_h]);
        let mut canvas = PreviewCanvas {
            width: out_w,
            height: out_h,
            vb_x: viewbox[0],
            vb_y: viewbox[1],
            scale_x: f64::from(out_w) / viewbox[2].max(1e-6),
            scale_y: f64::from(out_h) / viewbox[3].max(1e-6),
            pixels: &mut pixels,
        };
        let default_style = PaintStyle::default();
        rasterize_node(root, default_style, &mut canvas);
    }

    Ok(Image {
        width: out_w,
        height: out_h,
        pixels,
    })
}

#[derive(Clone, Copy)]
struct PaintStyle {
    fill: Option<[u8; 3]>,
    stroke: Option<[u8; 3]>,
}

impl Default for PaintStyle {
    fn default() -> Self {
        Self {
            fill: Some([0, 0, 0]),
            stroke: None,
        }
    }
}

struct PreviewCanvas<'a> {
    width: u32,
    height: u32,
    vb_x: f64,
    vb_y: f64,
    scale_x: f64,
    scale_y: f64,
    pixels: &'a mut [u8],
}

impl PreviewCanvas<'_> {
    fn map_x(&self, x: f64) -> f64 {
        (x - self.vb_x) * self.scale_x
    }

    fn map_y(&self, y: f64) -> f64 {
        (y - self.vb_y) * self.scale_y
    }

    fn put_pixel(&mut self, x: i32, y: i32, rgb: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let idx = ((y as usize) * (self.width as usize) + (x as usize)) * 3;
        self.pixels[idx..idx + 3].copy_from_slice(&rgb);
    }

    fn draw_line(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, rgb: [u8; 3]) {
        let px0 = self.map_x(x0);
        let py0 = self.map_y(y0);
        let px1 = self.map_x(x1);
        let py1 = self.map_y(y1);
        let steps = (px1 - px0).abs().max((py1 - py0).abs()).ceil().max(1.0) as i32;
        let steps = steps.min((self.width.max(self.height) * 4) as i32);
        for i in 0..=steps {
            let t = f64::from(i) / f64::from(steps);
            let x = (px0 + (px1 - px0) * t).round() as i32;
            let y = (py0 + (py1 - py0) * t).round() as i32;
            self.put_pixel(x, y, rgb);
        }
    }

    fn fill_polygon(&mut self, pts: &[(f64, f64)], rgb: [u8; 3]) {
        if pts.len() < 3 {
            return;
        }
        let mapped: Vec<(f64, f64)> = pts
            .iter()
            .map(|&(x, y)| (self.map_x(x), self.map_y(y)))
            .collect();
        let min_y = mapped
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as i32;
        let max_y = mapped
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            .min(f64::from(self.height) - 1.0) as i32;
        let mut xs = Vec::new();
        for py in min_y..=max_y {
            let sy = f64::from(py) + 0.5;
            xs.clear();
            for i in 0..mapped.len() {
                let (x0, y0) = mapped[i];
                let (x1, y1) = mapped[(i + 1) % mapped.len()];
                if (y0 <= sy && y1 > sy) || (y1 <= sy && y0 > sy) {
                    let t = (sy - y0) / (y1 - y0);
                    xs.push(x0 + t * (x1 - x0));
                }
            }
            xs.sort_by(|a, b| a.total_cmp(b));
            for pair in xs.chunks_exact(2) {
                let x_start = pair[0].floor().max(0.0) as i32;
                let x_end = pair[1].ceil().min(f64::from(self.width) - 1.0) as i32;
                for px in x_start..=x_end {
                    self.put_pixel(px, py, rgb);
                }
            }
        }
    }
}

fn parse_color(raw: &str) -> Option<Option<[u8; 3]>> {
    let s = raw.trim().to_ascii_lowercase();
    if s == "none" || s == "transparent" {
        return Some(None);
    }
    if let Some(hex) = s.strip_prefix('#') {
        if hex.len() == 6 {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            return Some(Some([r, g, b]));
        }
        if hex.len() == 3 {
            let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
            return Some(Some([r, g, b]));
        }
    }
    if let Some(inner) = s.strip_prefix("rgb(").and_then(|r| r.strip_suffix(')')) {
        let parts: Vec<u8> = inner
            .split(',')
            .filter_map(|p| p.trim().parse::<f64>().ok())
            .map(|v| v.clamp(0.0, 255.0).round() as u8)
            .collect();
        if parts.len() == 3 {
            return Some(Some([parts[0], parts[1], parts[2]]));
        }
    }
    let named = match s.as_str() {
        "black" => [0, 0, 0],
        "white" => [255, 255, 255],
        "red" => [255, 0, 0],
        "green" => [0, 128, 0],
        "blue" => [0, 0, 255],
        "yellow" => [255, 255, 0],
        "cyan" => [0, 255, 255],
        "magenta" => [255, 0, 255],
        "gray" | "grey" => [128, 128, 128],
        "orange" => [255, 165, 0],
        "purple" => [128, 0, 128],
        _ => return None,
    };
    Some(Some(named))
}

fn resolve_style(attrs: &[(String, String)], inherited: PaintStyle) -> PaintStyle {
    let mut style = inherited;
    let mut apply = |key: &str, value: &str| match key {
        "fill" => {
            if let Some(c) = parse_color(value) {
                style.fill = c;
            }
        }
        "stroke" => {
            if let Some(c) = parse_color(value) {
                style.stroke = c;
            }
        }
        _ => {}
    };
    for (k, v) in attrs {
        if k == "style" {
            for decl in v.split(';') {
                if let Some((prop, val)) = decl.split_once(':') {
                    apply(prop.trim(), val.trim());
                }
            }
        } else {
            apply(k.as_str(), v.as_str());
        }
    }
    style
}

fn attr_f64(attrs: &[(String, String)], key: &str, default: f64) -> f64 {
    attrs
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| v.trim().strip_suffix("px").unwrap_or(v.trim()).parse().ok())
        .filter(|v: &f64| v.is_finite())
        .unwrap_or(default)
}

fn parse_points(raw: &str) -> Vec<(f64, f64)> {
    let nums: Vec<f64> = raw
        .split(|c: char| c.is_ascii_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .collect();
    nums.chunks_exact(2).map(|c| (c[0], c[1])).collect()
}

fn parse_path_subpaths(d: &str) -> Vec<(Vec<(f64, f64)>, bool)> {
    let mut tokens = Vec::new();
    let bytes = d.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() || b == b',' {
            i += 1;
            continue;
        }
        if b.is_ascii_alphabetic() {
            tokens.push(&d[i..i + 1]);
            i += 1;
            continue;
        }
        let start = i;
        if b == b'+' || b == b'-' {
            i += 1;
        }
        let mut saw_dot = false;
        while i < bytes.len() {
            let c = bytes[i];
            if c.is_ascii_digit() {
                i += 1;
            } else if c == b'.' && !saw_dot {
                saw_dot = true;
                i += 1;
            } else if (c == b'e' || c == b'E') && i > start {
                i += 1;
                if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                    i += 1;
                }
            } else {
                break;
            }
        }
        if i > start {
            tokens.push(&d[start..i]);
        } else {
            i += 1;
        }
    }

    const CURVE_SEGMENTS: usize = 16;
    const MAX_PATH_POINTS: usize = 16_384;

    let mut subpaths = Vec::new();
    let mut current = Vec::new();
    let mut cur = (0.0f64, 0.0f64);
    let mut start_pt = (0.0f64, 0.0f64);
    let mut last_cubic_ctrl: Option<(f64, f64)> = None;
    let mut last_quad_ctrl: Option<(f64, f64)> = None;
    let mut total_points = 0usize;
    let mut cmd = 'M';
    let mut idx = 0;
    let next_num = |idx: &mut usize| -> Option<f64> {
        let val = tokens.get(*idx)?.parse::<f64>().ok()?;
        *idx += 1;
        val.is_finite().then_some(val)
    };
    let push_pt = |current: &mut Vec<(f64, f64)>, total_points: &mut usize, pt: (f64, f64)| {
        if *total_points < MAX_PATH_POINTS {
            current.push(pt);
            *total_points += 1;
        }
    };
    let flatten_cubic = |current: &mut Vec<(f64, f64)>,
                         total_points: &mut usize,
                         p0: (f64, f64),
                         p1: (f64, f64),
                         p2: (f64, f64),
                         p3: (f64, f64)| {
        for step in 1..=CURVE_SEGMENTS {
            let t = step as f64 / CURVE_SEGMENTS as f64;
            let u = 1.0 - t;
            let x = u * u * u * p0.0
                + 3.0 * u * u * t * p1.0
                + 3.0 * u * t * t * p2.0
                + t * t * t * p3.0;
            let y = u * u * u * p0.1
                + 3.0 * u * u * t * p1.1
                + 3.0 * u * t * t * p2.1
                + t * t * t * p3.1;
            if *total_points < MAX_PATH_POINTS {
                current.push((x, y));
                *total_points += 1;
            }
        }
    };
    let flatten_quad = |current: &mut Vec<(f64, f64)>,
                        total_points: &mut usize,
                        p0: (f64, f64),
                        p1: (f64, f64),
                        p2: (f64, f64)| {
        for step in 1..=CURVE_SEGMENTS {
            let t = step as f64 / CURVE_SEGMENTS as f64;
            let u = 1.0 - t;
            let x = u * u * p0.0 + 2.0 * u * t * p1.0 + t * t * p2.0;
            let y = u * u * p0.1 + 2.0 * u * t * p1.1 + t * t * p2.1;
            if *total_points < MAX_PATH_POINTS {
                current.push((x, y));
                *total_points += 1;
            }
        }
    };
    while idx < tokens.len() {
        let tok = tokens[idx];
        if tok.len() == 1 && tok.as_bytes()[0].is_ascii_alphabetic() {
            cmd = tok.as_bytes()[0] as char;
            idx += 1;
            if cmd == 'Z' || cmd == 'z' {
                if !current.is_empty() {
                    subpaths.push((std::mem::take(&mut current), true));
                }
                cur = start_pt;
                last_cubic_ctrl = None;
                last_quad_ctrl = None;
            }
            continue;
        }
        match cmd {
            'M' | 'm' => {
                let (Some(x), Some(y)) = (next_num(&mut idx), next_num(&mut idx)) else {
                    break;
                };
                if !current.is_empty() {
                    subpaths.push((std::mem::take(&mut current), false));
                }
                cur = if cmd == 'm' {
                    (cur.0 + x, cur.1 + y)
                } else {
                    (x, y)
                };
                start_pt = cur;
                last_cubic_ctrl = None;
                last_quad_ctrl = None;
                push_pt(&mut current, &mut total_points, cur);
                cmd = if cmd == 'm' { 'l' } else { 'L' };
            }
            'L' | 'l' => {
                let (Some(x), Some(y)) = (next_num(&mut idx), next_num(&mut idx)) else {
                    break;
                };
                cur = if cmd == 'l' {
                    (cur.0 + x, cur.1 + y)
                } else {
                    (x, y)
                };
                last_cubic_ctrl = None;
                last_quad_ctrl = None;
                push_pt(&mut current, &mut total_points, cur);
            }
            'H' | 'h' => {
                let Some(x) = next_num(&mut idx) else { break };
                cur.0 = if cmd == 'h' { cur.0 + x } else { x };
                last_cubic_ctrl = None;
                last_quad_ctrl = None;
                push_pt(&mut current, &mut total_points, cur);
            }
            'V' | 'v' => {
                let Some(y) = next_num(&mut idx) else { break };
                cur.1 = if cmd == 'v' { cur.1 + y } else { y };
                last_cubic_ctrl = None;
                last_quad_ctrl = None;
                push_pt(&mut current, &mut total_points, cur);
            }
            'C' | 'c' => {
                let (Some(x1), Some(y1), Some(x2), Some(y2), Some(x), Some(y)) = (
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                ) else {
                    break;
                };
                let (p1, p2, p3) = if cmd == 'c' {
                    (
                        (cur.0 + x1, cur.1 + y1),
                        (cur.0 + x2, cur.1 + y2),
                        (cur.0 + x, cur.1 + y),
                    )
                } else {
                    ((x1, y1), (x2, y2), (x, y))
                };
                flatten_cubic(&mut current, &mut total_points, cur, p1, p2, p3);
                last_cubic_ctrl = Some(p2);
                last_quad_ctrl = None;
                cur = p3;
            }
            'S' | 's' => {
                let (Some(x2), Some(y2), Some(x), Some(y)) = (
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                ) else {
                    break;
                };
                let p1 = match last_cubic_ctrl {
                    Some(prev) => (2.0 * cur.0 - prev.0, 2.0 * cur.1 - prev.1),
                    None => cur,
                };
                let (p2, p3) = if cmd == 's' {
                    ((cur.0 + x2, cur.1 + y2), (cur.0 + x, cur.1 + y))
                } else {
                    ((x2, y2), (x, y))
                };
                flatten_cubic(&mut current, &mut total_points, cur, p1, p2, p3);
                last_cubic_ctrl = Some(p2);
                last_quad_ctrl = None;
                cur = p3;
            }
            'Q' | 'q' => {
                let (Some(x1), Some(y1), Some(x), Some(y)) = (
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                ) else {
                    break;
                };
                let (p1, p2) = if cmd == 'q' {
                    ((cur.0 + x1, cur.1 + y1), (cur.0 + x, cur.1 + y))
                } else {
                    ((x1, y1), (x, y))
                };
                flatten_quad(&mut current, &mut total_points, cur, p1, p2);
                last_quad_ctrl = Some(p1);
                last_cubic_ctrl = None;
                cur = p2;
            }
            'T' | 't' => {
                let (Some(x), Some(y)) = (next_num(&mut idx), next_num(&mut idx)) else {
                    break;
                };
                let p1 = match last_quad_ctrl {
                    Some(prev) => (2.0 * cur.0 - prev.0, 2.0 * cur.1 - prev.1),
                    None => cur,
                };
                let p2 = if cmd == 't' {
                    (cur.0 + x, cur.1 + y)
                } else {
                    (x, y)
                };
                flatten_quad(&mut current, &mut total_points, cur, p1, p2);
                last_quad_ctrl = Some(p1);
                last_cubic_ctrl = None;
                cur = p2;
            }
            _ => {
                last_cubic_ctrl = None;
                last_quad_ctrl = None;
                idx += 1;
            }
        }
    }
    if !current.is_empty() {
        subpaths.push((current, false));
    }
    subpaths
}

fn rasterize_node(node: &SvgNode, inherited: PaintStyle, canvas: &mut PreviewCanvas<'_>) {
    if matches!(node.name.as_str(), "defs" | "clipPath" | "filter" | "style") {
        return;
    }
    if node
        .attrs
        .iter()
        .any(|(k, v)| k == "display" && v.trim() == "none")
    {
        return;
    }
    let style = resolve_style(&node.attrs, inherited);
    match node.name.as_str() {
        "rect" => {
            let x = attr_f64(&node.attrs, "x", 0.0);
            let y = attr_f64(&node.attrs, "y", 0.0);
            let w = attr_f64(&node.attrs, "width", 0.0);
            let h = attr_f64(&node.attrs, "height", 0.0);
            if w > 0.0 && h > 0.0 {
                let pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)];
                if let Some(fill) = style.fill {
                    canvas.fill_polygon(&pts, fill);
                }
                if let Some(stroke) = style.stroke {
                    for i in 0..4 {
                        let (x0, y0) = pts[i];
                        let (x1, y1) = pts[(i + 1) % 4];
                        canvas.draw_line(x0, y0, x1, y1, stroke);
                    }
                }
            }
        }
        "circle" | "ellipse" => {
            let cx = attr_f64(&node.attrs, "cx", 0.0);
            let cy = attr_f64(&node.attrs, "cy", 0.0);
            let (rx, ry) = if node.name == "circle" {
                let r = attr_f64(&node.attrs, "r", 0.0);
                (r, r)
            } else {
                (
                    attr_f64(&node.attrs, "rx", 0.0),
                    attr_f64(&node.attrs, "ry", 0.0),
                )
            };
            if rx > 0.0 && ry > 0.0 {
                let segments = 32usize;
                let pts: Vec<(f64, f64)> = (0..segments)
                    .map(|i| {
                        let a = std::f64::consts::TAU * (i as f64) / (segments as f64);
                        (cx + rx * a.cos(), cy + ry * a.sin())
                    })
                    .collect();
                if let Some(fill) = style.fill {
                    canvas.fill_polygon(&pts, fill);
                }
                if let Some(stroke) = style.stroke {
                    for i in 0..pts.len() {
                        let (x0, y0) = pts[i];
                        let (x1, y1) = pts[(i + 1) % pts.len()];
                        canvas.draw_line(x0, y0, x1, y1, stroke);
                    }
                }
            }
        }
        "line" => {
            if let Some(stroke) = style.stroke.or(style.fill) {
                canvas.draw_line(
                    attr_f64(&node.attrs, "x1", 0.0),
                    attr_f64(&node.attrs, "y1", 0.0),
                    attr_f64(&node.attrs, "x2", 0.0),
                    attr_f64(&node.attrs, "y2", 0.0),
                    stroke,
                );
            }
        }
        "polygon" | "polyline" => {
            if let Some((_, raw)) = node.attrs.iter().find(|(k, _)| k == "points") {
                let pts = parse_points(raw);
                if node.name == "polygon" {
                    if let Some(fill) = style.fill {
                        canvas.fill_polygon(&pts, fill);
                    }
                }
                if let Some(stroke) = style.stroke {
                    let count = if node.name == "polygon" {
                        pts.len()
                    } else {
                        pts.len().saturating_sub(1)
                    };
                    for i in 0..count {
                        let (x0, y0) = pts[i];
                        let (x1, y1) = pts[(i + 1) % pts.len()];
                        canvas.draw_line(x0, y0, x1, y1, stroke);
                    }
                }
            }
        }
        "path" => {
            if let Some((_, d)) = node.attrs.iter().find(|(k, _)| k == "d") {
                for (pts, closed) in parse_path_subpaths(d) {
                    if let Some(fill) = style.fill {
                        canvas.fill_polygon(&pts, fill);
                    }
                    if let Some(stroke) = style.stroke {
                        let count = if closed {
                            pts.len()
                        } else {
                            pts.len().saturating_sub(1)
                        };
                        for i in 0..count {
                            let (x0, y0) = pts[i];
                            let (x1, y1) = pts[(i + 1) % pts.len()];
                            canvas.draw_line(x0, y0, x1, y1, stroke);
                        }
                    }
                }
            }
        }
        _ => {}
    }
    for child in &node.children {
        if let SvgChild::Element(el) = child {
            rasterize_node(el, style, canvas);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_valid_svg_and_populates_canonical_metadata_and_preview() {
        let raw = r##"<svg viewBox="0 0 100 50"><rect x="10" y="10" width="80" height="30" fill="#ff0000"/></svg>"##;
        let canonical = sanitize_svg(raw).expect("valid static SVG must sanitize");
        assert!(canonical.svg.starts_with(XML_DECLARATION));
        assert!(canonical.svg.contains(SVG_NAMESPACE));
        assert_eq!((canonical.width, canonical.height), (100, 50));
        assert_eq!(canonical.element_count, 2);
        assert_eq!(canonical.svg_sha256.len(), 64);
        assert_eq!(canonical.svg_sha256, sha256_hex(canonical.svg.as_bytes()));

        let preview = render_svg_preview(&canonical, 64).expect("preview must render");
        assert_eq!((preview.width, preview.height), (64, 32));
        assert_eq!(preview.pixels.len(), 64 * 32 * 3);
        // Center pixel should be painted red by the rect, corner pixel stays white.
        let center_idx = (16 * 64 + 32) * 3;
        assert_eq!(&preview.pixels[center_idx..center_idx + 3], &[255, 0, 0]);
        assert_eq!(&preview.pixels[0..3], &[255, 255, 255]);
    }

    #[test]
    fn rejects_active_elements_event_handlers_and_external_hrefs_fail_closed() {
        for hostile in [
            r##"<svg width="64" height="64" onload="alert(1)"><rect x="0" y="0" width="64" height="64"/></svg>"##,
            r##"<svg width="64" height="64"><script>alert('xss')</script></svg>"##,
            r##"<svg width="64" height="64"><foreignObject width="64" height="64"><g/></foreignObject></svg>"##,
            r##"<svg width="64" height="64"><defs><path id="ok" d="M0 0 L10 10 Z"/></defs><use href="javascript:alert(2)"/></svg>"##,
            r##"<svg width="64" height="64" xmlns:xlink="http://www.w3.org/1999/xlink"><use xlink:href="https://evil.example/x.svg#a"/></svg>"##,
            r##"<svg width="64" height="64"><circle cx="32" cy="32" r="16" fill="blue" onerror="boom()"/></svg>"##,
            r##"<svg width="64" height="64"><iframe/></svg>"##,
            r##"<svg width="64" height="64"><animate attributeName="x" from="0" to="10"/></svg>"##,
            r##"<svg width="64" height="64"><image href="https://evil.example/a.png"/></svg>"##,
            r##"<svg width="64" height="64"><a href="#ok"><rect width="10" height="10"/></a></svg>"##,
        ] {
            assert!(
                sanitize_svg(hostile).is_err(),
                "sceneworks-static-svg-v2 must fail closed on active markup: {hostile}"
            );
        }
    }

    #[test]
    fn rejects_css_escape_at_rule_and_image_set_exfiltration_bypasses() {
        for bypass in [
            // CSS hex escape for `url(...)`
            r##"<svg width="64" height="64"><rect width="10" height="10" style="fill:\75 rl(https://evil.example/a)"/></svg>"##,
            r##"<svg width="64" height="64"><style>rect { fill: \75 rl(https://evil.example/a); }</style><rect width="10" height="10"/></svg>"##,
            // Escaped and raw @import rules
            r##"<svg width="64" height="64"><style>@import url(https://evil.example/a.css);</style><rect width="10" height="10"/></svg>"##,
            r##"<svg width="64" height="64"><style>@\69 mport "https://evil.example/a.css";</style><rect width="10" height="10"/></svg>"##,
            // image-set / -webkit-image-set
            r##"<svg width="64" height="64"><rect width="10" height="10" style="fill:image-set(url(https://evil.example/x.png) 1x)"/></svg>"##,
            r##"<svg width="64" height="64"><style>rect { fill: -webkit-image-set("https://evil.example/x.png" 1x); }</style><rect width="10" height="10"/></svg>"##,
            // CSS comment splitting & disallowed properties
            r##"<svg width="64" height="64"><rect width="10" height="10" style="fill:u/*x*/rl(#g)"/></svg>"##,
            r##"<svg width="64" height="64"><rect width="10" height="10" style="background:url(https://evil.example/bg.png)"/></svg>"##,
            r##"<svg width="64" height="64"><rect width="10" height="10" style="fill:red;behavior:url(x)"/></svg>"##,
            r##"<svg width="64" height="64"><rect width="10" height="10" style="--x:red;fill:var(--x)"/></svg>"##,
            r##"<svg width="64" height="64"><rect width="10" height="10" fill="url(https://evil.example/p#a)"/></svg>"##,
        ] {
            assert!(
                sanitize_svg(bypass).is_err(),
                "CSS bypass must be rejected: {bypass}"
            );
        }

        // Valid static CSS with safe local gradient reference succeeds.
        let valid = r##"<svg width="64" height="64"><defs><linearGradient id="g1"><stop offset="0%" stop-color="#ff0000"/><stop offset="100%" stop-color="#0000ff"/></linearGradient></defs><style>rect { fill: url(#g1); stroke: rgb(10, 20, 30); }</style><rect x="0" y="0" width="64" height="64" style="opacity:0.9"/></svg>"##;
        let canonical = sanitize_svg(valid).expect("valid static CSS and local gradient must pass");
        assert!(canonical.svg.contains("url(#g1)"));
        assert!(canonical.svg.contains("opacity:0.9"));
    }

    #[test]
    fn rejects_xml_base_child_xmlns_and_unbound_xlink_href() {
        for bad in [
            // xml:base smuggling
            r##"<svg width="64" height="64" xml:base="https://evil.example/"><rect width="10" height="10"/></svg>"##,
            r##"<svg width="64" height="64"><g xml:base="https://evil.example/"><rect width="10" height="10"/></g></svg>"##,
            // child xmlns namespace switch
            r##"<svg width="64" height="64"><g xmlns="http://www.w3.org/1999/xhtml"><rect width="10" height="10"/></g></svg>"##,
            r##"<svg width="64" height="64" xmlns="http://www.w3.org/1999/xhtml"><rect width="10" height="10"/></svg>"##,
            // xlink:href without root xmlns:xlink
            r##"<svg width="64" height="64"><defs><path id="p1" d="M0 0 L10 10 Z"/></defs><use xlink:href="#p1"/></svg>"##,
            // wrong xmlns:xlink URI
            r##"<svg width="64" height="64" xmlns:xlink="https://evil.example/xlink"><defs><path id="p1" d="M0 0 L10 10 Z"/></defs><use xlink:href="#p1"/></svg>"##,
        ] {
            assert!(
                sanitize_svg(bad).is_err(),
                "namespace/attribute bypass must be rejected: {bad}"
            );
        }

        // Valid xlink:href WITH exact root xmlns:xlink passes and preserves valid XML binding.
        let ok_xlink = r##"<svg width="64" height="64" xmlns:xlink="http://www.w3.org/1999/xlink"><defs><path id="p1" d="M0 0 L10 10 Z"/></defs><use xlink:href="#p1"/></svg>"##;
        let canonical = sanitize_svg(ok_xlink).expect("bound xlink:href must pass");
        assert!(canonical.svg.contains(XLINK_NAMESPACE));
        assert!(canonical.svg.contains("xlink:href=\"#p1\""));
    }

    #[test]
    fn rejects_entity_nul_control_chars_duplicate_attrs_and_arbitrary_pis() {
        for bad in [
            // NUL and illegal XML 1.0 control chars after entity decoding
            r##"<svg width="64" height="64"><rect id="a&#0;b" width="10" height="10"/></svg>"##,
            r##"<svg width="64" height="64"><rect id="a&#x0;b" width="10" height="10"/></svg>"##,
            r##"<svg width="64" height="64"><title>hello&#0;world</title></svg>"##,
            r##"<svg width="64" height="64"><title>ctrl&#x1f;char</title></svg>"##,
            // Case-insensitive duplicate attributes
            r##"<svg width="64" height="64"><rect width="10" height="10" style="fill:red" STYLE="fill:blue"/></svg>"##,
            // Arbitrary processing instructions
            r##"<?xml-model href="https://evil.example/schema.rng"?><svg width="64" height="64"/>"##,
            r##"<?xml-stylesheet href="https://evil.example/a.css"?><svg width="64" height="64"/>"##,
            r##"<?xml version="1.0" evil="1"?><svg width="64" height="64"/>"##,
            r##"<?xml version="1.0" encoding="UTF-8"?><?xml version="1.0"?><svg width="64" height="64"/>"##,
        ] {
            assert!(
                sanitize_svg(bad).is_err(),
                "entity/duplicate/PI bypass must be rejected: {bad}"
            );
        }

        // Standard leading XML declaration succeeds.
        let ok_decl = r##"<?xml version="1.0" encoding="UTF-8"?><svg width="16" height="16"><rect width="8" height="8"/></svg>"##;
        assert!(sanitize_svg(ok_decl).is_ok());
    }

    #[test]
    fn rejects_use_cycles_unresolved_refs_billion_laughs_and_filter_budget_abuse() {
        // Direct <use> cycle
        let direct_cycle = r##"<svg width="64" height="64"><g id="a"><use href="#a"/></g></svg>"##;
        assert!(sanitize_svg(direct_cycle).is_err());

        // Mutual <use> cycle
        let mutual_cycle = r##"<svg width="64" height="64"><defs><g id="a"><use href="#b"/></g><g id="b"><use href="#a"/></g></defs><use href="#a"/></svg>"##;
        assert!(sanitize_svg(mutual_cycle).is_err());

        // Unresolved #id ref
        let unresolved = r##"<svg width="64" height="64"><use href="#does_not_exist"/></svg>"##;
        assert!(sanitize_svg(unresolved).is_err());

        // Billion-laughs <use> exponential amplification (2^12 = 4096 > MAX_SVG_ELEMENTS = 2000)
        let mut laughs = String::from(
            r##"<svg width="64" height="64"><defs><rect id="u0" width="1" height="1"/>"##,
        );
        for level in 1..=12 {
            let prev = level - 1;
            laughs.push_str(&format!(
                r##"<g id="u{level}"><use href="#u{prev}"/><use href="#u{prev}"/></g>"##
            ));
        }
        laughs.push_str(r##"</defs><use href="#u12"/></svg>"##);
        assert!(
            sanitize_svg(&laughs).is_err(),
            "billion-laughs <use> amplification must be rejected"
        );

        // Filter stdDeviation > MAX_SVG_FILTER_BLUR_SIGMA (64.0)
        let huge_blur = r##"<svg width="64" height="64"><defs><filter id="f1"><feGaussianBlur in="SourceGraphic" stdDeviation="65" result="b"/></filter></defs><rect width="10" height="10" filter="url(#f1)"/></svg>"##;
        assert!(sanitize_svg(huge_blur).is_err());

        // Filter primitive count > MAX_SVG_FILTER_PRIMITIVES (64)
        let mut many_prims =
            String::from(r##"<svg width="64" height="64"><defs><filter id="f1">"##);
        for i in 0..=MAX_SVG_FILTER_PRIMITIVES {
            many_prims.push_str(&format!(
                r##"<feGaussianBlur in="SourceGraphic" stdDeviation="1" result="r{i}"/>"##
            ));
        }
        many_prims.push_str(
            r##"</filter></defs><rect width="10" height="10" filter="url(#f1)"/></svg>"##,
        );
        assert!(sanitize_svg(&many_prims).is_err());

        // Filter primitive outside <filter>
        let stray_prim = r##"<svg width="64" height="64"><feGaussianBlur in="SourceGraphic" stdDeviation="2"/></svg>"##;
        assert!(sanitize_svg(stray_prim).is_err());

        // Bounded valid filter passes
        let ok_filter = r##"<svg width="64" height="64"><defs><filter id="f1" width="120%" height="120%"><feGaussianBlur in="SourceGraphic" stdDeviation="4" result="b1"/><feOffset in="b1" dx="2" dy="2" result="o1"/></filter></defs><rect width="10" height="10" filter="url(#f1)"/></svg>"##;
        assert!(sanitize_svg(ok_filter).is_ok());
    }

    #[test]
    fn refuses_malformed_truncated_doctype_and_oversized_svgs() {
        for bad in [
            "",
            "   ",
            "not svg at all",
            "<svg width=\"10\" height=\"10\"><rect x=\"0\" y=\"0\" width=\"5\" height=\"5\"/>",
            "<svg width=\"10\" height=\"10\"><g></svg>",
            "<!DOCTYPE svg SYSTEM \"http://evil.example/x.dtd\"><svg width=\"10\" height=\"10\"/>",
            "<?xml-stylesheet href=\"https://evil.example/a.css\"?><svg width=\"10\" height=\"10\"/>",
            "<svg width=\"0\" height=\"10\"/>",
            "<svg/>",
            "<svg width=\"10\" height=\"10\"/>trailing",
        ] {
            assert!(
                sanitize_svg(bad).is_err(),
                "expected refusal for invalid SVG: {bad:?}"
            );
        }

        let opts = VectorSanitizeOptions {
            max_bytes: 32,
            ..Default::default()
        };
        let too_big = b"<svg width=\"10\" height=\"10\"><rect x=\"0\" y=\"0\" width=\"10\" height=\"10\"/></svg>";
        assert!(sanitize_svg_with_options(too_big, opts).is_err());
    }

    #[test]
    fn build_vector_output_populates_finish_reason_preview_and_terminal_metadata() {
        let raw = r##"<svg width="32" height="32"><circle cx="16" cy="16" r="10" fill="#00ff00"/></svg>"##;
        let out = build_vector_output(
            Some(raw),
            StarVectorFinishReason::CompleteRoot,
            42,
            Some(259),
            raw.len(),
            "mlx-starvector-1b",
            VectorSanitizeOptions::default(),
            false,
        )
        .expect("complete root builds VectorOutput");
        assert_eq!(out.finish_reason, VectorFinishReason::Stop);
        assert!(!out.truncated);
        assert_eq!(out.generated_tokens, 42);
        assert_eq!(out.prompt_tokens, Some(259));
        assert!(out.preview.is_some());
        assert_eq!(
            out.terminal.sanitizer_id.as_deref(),
            Some(VECTOR_SANITIZER_VERSION)
        );
        assert_eq!(out.terminal.viewport, Some([32, 32]));
        assert_eq!(out.terminal.element_count, Some(2));
        assert_eq!(
            out.terminal.provider_finish_reason.as_deref(),
            Some("complete_root")
        );
        assert_eq!(out.terminal.generated_bytes, Some(raw.len()));
        assert_eq!(
            out.terminal.provider_id.as_deref(),
            Some("mlx-starvector-1b")
        );
        assert!(out.terminal.svg_sha256.is_some());

        // Token limit refuses by default, or returns typed truncated metadata when opted in.
        assert!(build_vector_output(
            None,
            StarVectorFinishReason::TokenLimit,
            100,
            Some(259),
            512,
            "candle-starvector-8b",
            VectorSanitizeOptions::default(),
            false,
        )
        .is_err());
        let limited = build_vector_output(
            None,
            StarVectorFinishReason::TokenLimit,
            100,
            Some(259),
            512,
            "candle-starvector-8b",
            VectorSanitizeOptions::default(),
            true,
        )
        .expect("non-publishable terminal outcome returns typed metadata when allowed");
        assert_eq!(limited.finish_reason, VectorFinishReason::Length);
        assert!(limited.truncated);
        assert!(limited.svg.is_empty());
        assert!(limited.preview.is_none());
        assert_eq!(
            limited.terminal.provider_finish_reason.as_deref(),
            Some("token_limit")
        );

        // Cancelled maps to typed Error::Canceled.
        assert!(matches!(
            build_vector_output(
                None,
                StarVectorFinishReason::Cancelled,
                5,
                None,
                10,
                "mlx-starvector-1b",
                VectorSanitizeOptions::default(),
                true,
            ),
            Err(Error::Canceled)
        ));
    }

    #[test]
    fn preview_rasterizer_flattens_cubic_quadratic_and_smooth_bezier_curves() {
        // Cubic dome `M0 10 C0 0 10 0 10 10 Z` encloses a filled interior that disappears if
        // control points are discarded.
        let cubic_svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 10 10"><path d="M0 10 C0 0 10 0 10 10 Z" fill="#000000"/></svg>"##;
        let cubic_out = build_vector_output(
            Some(cubic_svg),
            StarVectorFinishReason::CompleteRoot,
            10,
            None,
            100,
            "test-vector",
            VectorSanitizeOptions {
                preview_max_dimension: 64,
                ..Default::default()
            },
            false,
        )
        .expect("cubic curve SVG must sanitize and rasterize");
        let cubic_preview = cubic_out.preview.expect("cubic preview must render");
        let dark_pixels = cubic_preview
            .pixels
            .chunks_exact(3)
            .filter(|rgb| rgb[0] < 64 && rgb[1] < 64 && rgb[2] < 64)
            .count();
        assert!(
            dark_pixels > 1000,
            "cubic Bezier filled interior must be visible in preview (got {dark_pixels} dark pixels)"
        );

        // Quadratic + smooth quadratic (`Q` / `T`) and smooth cubic (`S`) also render filled regions.
        let quad_svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="32" viewBox="0 0 20 10"><path d="M0 10 Q5 0 10 10 T20 10 Z" fill="#000000"/></svg>"##;
        let quad_out = build_vector_output(
            Some(quad_svg),
            StarVectorFinishReason::CompleteRoot,
            10,
            None,
            100,
            "test-vector",
            VectorSanitizeOptions {
                preview_max_dimension: 64,
                ..Default::default()
            },
            false,
        )
        .expect("quadratic curve SVG must sanitize and rasterize");
        let quad_preview = quad_out.preview.expect("quad preview must render");
        let quad_dark = quad_preview
            .pixels
            .chunks_exact(3)
            .filter(|rgb| rgb[0] < 64 && rgb[1] < 64 && rgb[2] < 64)
            .count();
        assert!(
            quad_dark > 50,
            "quadratic/smooth Bezier interior must be visible in preview (got {quad_dark} dark pixels)"
        );
    }
}
