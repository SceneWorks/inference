//! Bounded, inert static SVG sanitization, canonicalization, derived preview rasterization, and
//! typed [`VectorOutput`] construction (`PIPE-12`, `E4`).
//!
//! This module is deliberately **tensor-free** and **zero-external-XML-dependency** so it lives
//! inside `gen-core` and is shared identically by MLX and Candle StarVector routes as well as
//! SceneWorks consumers.

use sha2::{Digest, Sha256};

use crate::core_llm::StarVectorFinishReason;
use crate::{Error, Image, Result, VectorFinishReason, VectorOutput, VectorTerminalMetadata};

/// Canonical sanitizer policy/version identifier embedded in [`VectorTerminalMetadata::sanitizer_id`].
pub const VECTOR_SANITIZER_VERSION: &str = "sceneworks-static-svg-v2";
/// Canonical SVG XML namespace enforced on every sanitized root element.
pub const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";
/// Maximum accepted raw/canonical SVG document size in UTF-8 bytes (256 KiB).
pub const MAX_SVG_BYTES: usize = 256 * 1024;
/// Maximum allowed XML element nesting depth.
pub const MAX_SVG_DEPTH: usize = 32;
/// Maximum allowed retained SVG elements per document.
pub const MAX_SVG_ELEMENTS: usize = 2_000;
/// Maximum allowed attributes across the entire SVG document.
pub const MAX_SVG_ATTRIBUTES: usize = 4_096;
/// Maximum byte length of any single attribute value (64 KiB).
pub const MAX_SVG_ATTRIBUTE_VALUE_BYTES: usize = 64 * 1024;
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
        return Err(Error::Msg(
            "vector SVG contains forbidden NUL bytes".into(),
        ));
    }
    let text = std::str::from_utf8(input)
        .map_err(|err| Error::Msg(format!("vector SVG is not valid UTF-8: {err}")))?;
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

fn is_stripped_active_element(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "script"
            | "foreignobject"
            | "iframe"
            | "object"
            | "embed"
            | "applet"
            | "audio"
            | "video"
            | "animate"
            | "animatemotion"
            | "animatetransform"
            | "set"
            | "handler"
            | "discard"
            | "a"
            | "image"
            | "feimage"
    )
}

fn is_allowed_static_element(name: &str) -> bool {
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
            | "text"
            | "tspan"
    )
}

fn is_forbidden_attribute_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("on")
        || matches!(lower.as_str(), "src" | "action" | "formaction" | "srcdoc")
}

fn strip_ascii_control_and_ws(value: &str) -> String {
    value
        .chars()
        .filter(|c| !c.is_ascii_control() && !c.is_ascii_whitespace())
        .collect()
}

fn is_safe_local_fragment(value: &str) -> bool {
    let trimmed = value.trim();
    let Some(id) = trimmed.strip_prefix('#') else {
        return false;
    };
    !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

fn contains_unsafe_url_or_script(value: &str) -> bool {
    let compact = strip_ascii_control_and_ws(value).to_ascii_lowercase();
    if compact.contains("javascript:")
        || compact.contains("vbscript:")
        || compact.contains("expression(")
        || compact.contains("behavior:")
        || compact.contains("@import")
    {
        return true;
    }
    // Check every `url(...)` occurrence: only local `url(#id)` is permitted.
    let mut rest = compact.as_str();
    while let Some(pos) = rest.find("url(") {
        let after = &rest[pos + 4..];
        let Some(end) = after.find(')') else {
            return true;
        };
        let target = after[..end].trim_matches(|c| c == '\'' || c == '"');
        if !is_safe_local_fragment(target) {
            return true;
        }
        rest = &after[end + 1..];
    }
    false
}

fn sanitize_style_value(style: &str) -> Option<String> {
    let mut kept = Vec::new();
    for decl in style.split(';') {
        let decl = decl.trim();
        if decl.is_empty() {
            continue;
        }
        if contains_unsafe_url_or_script(decl) {
            continue;
        }
        if let Some((prop, val)) = decl.split_once(':') {
            let prop = prop.trim();
            let val = val.trim();
            if prop.is_empty() || val.is_empty() || prop.starts_with("--") {
                continue;
            }
            kept.push(format!("{prop}:{val}"));
        }
    }
    (!kept.is_empty()).then(|| kept.join(";"))
}

fn sanitize_stylesheet(css: &str) -> String {
    let mut kept = Vec::new();
    for rule in css.split('}') {
        let rule = rule.trim();
        if rule.is_empty() || contains_unsafe_url_or_script(rule) {
            continue;
        }
        if let Some((selector, body)) = rule.split_once('{') {
            if let Some(clean_body) = sanitize_style_value(body) {
                let selector = selector.trim();
                if !selector.is_empty() && !selector.starts_with('@') {
                    kept.push(format!("{selector}{{{clean_body}}}"));
                }
            }
        }
    }
    kept.join("\n")
}

fn sanitize_attribute(name: &str, value: &str) -> Option<(String, String)> {
    if is_forbidden_attribute_name(name) {
        return None;
    }
    let lower = name.to_ascii_lowercase();
    if matches!(lower.as_str(), "href" | "xlink:href") {
        return is_safe_local_fragment(value).then(|| (name.to_owned(), value.trim().to_owned()));
    }
    if lower == "style" {
        return sanitize_style_value(value).map(|v| ("style".to_owned(), v));
    }
    if contains_unsafe_url_or_script(value) {
        return None;
    }
    Some((name.to_owned(), value.to_owned()))
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

fn decode_xml_entities(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    let bytes = input.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'&' {
            if let Some(semi) = input[i + 1..].find(';') {
                let entity = &input[i + 1..i + 1 + semi];
                let decoded = match entity {
                    "amp" => Some("&".to_owned()),
                    "lt" => Some("<".to_owned()),
                    "gt" => Some(">".to_owned()),
                    "quot" => Some("\"".to_owned()),
                    "apos" => Some("'".to_owned()),
                    num if num.starts_with("#x") || num.starts_with("#X") => {
                        u32::from_str_radix(&num[2..], 16)
                            .ok()
                            .and_then(char::from_u32)
                            .map(|c| c.to_string())
                    }
                    num if num.starts_with('#') => num[1..]
                        .parse::<u32>()
                        .ok()
                        .and_then(char::from_u32)
                        .map(|c| c.to_string()),
                    _ => None,
                };
                if let Some(ch) = decoded {
                    out.push_str(&ch);
                    i += semi + 2;
                    continue;
                }
            }
        }
        let ch = input[i..].chars().next().expect("non-empty slice");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
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

fn parse_attributes(
    cursor: &mut XmlCursor<'_>,
    total_attrs: &mut usize,
) -> Result<(Vec<(String, String)>, bool)> {
    let mut attrs = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
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
        *total_attrs = total_attrs
            .checked_add(1)
            .ok_or_else(|| Error::Msg("vector SVG attribute count overflow".into()))?;
        if *total_attrs > MAX_SVG_ATTRIBUTES {
            return Err(Error::Msg(format!(
                "vector SVG exceeds the {MAX_SVG_ATTRIBUTES}-attribute budget"
            )));
        }
        if !seen.insert(name.clone()) {
            return Err(Error::Msg(format!(
                "vector SVG contains duplicate attribute '{name}'"
            )));
        }
        let decoded = decode_xml_entities(raw_value);
        attrs.push((name, decoded));
    }
}

fn parse_and_sanitize_document(input: &str) -> Result<Vec<SvgNode>> {
    let mut cursor = XmlCursor::new(input);
    cursor.skip_ws();

    // Permit a single standard leading XML declaration, refusing xml-stylesheet / DOCTYPE.
    if cursor.remaining().starts_with("<?xml") {
        let Some(end) = cursor.remaining().find("?>") else {
            return Err(Error::Msg(
                "vector SVG has an unterminated XML declaration".into(),
            ));
        };
        let decl = &cursor.remaining()[..end + 2];
        if decl.to_ascii_lowercase().starts_with("<?xml-stylesheet") {
            return Err(Error::Msg(
                "vector SVG processing instruction <?xml-stylesheet?> is forbidden".into(),
            ));
        }
        cursor.pos += end + 2;
        cursor.skip_ws();
    }

    let mut roots: Vec<SvgNode> = Vec::new();
    let mut stack: Vec<(Option<SvgNode>, String)> = Vec::new();
    let mut total_elements = 0usize;
    let mut total_attrs = 0usize;
    let mut stripped_depth = 0usize;

    while !cursor.is_eof() {
        let rem = cursor.remaining();
        if !rem.starts_with('<') {
            let next_lt = rem.find('<').unwrap_or(rem.len());
            let text_raw = &rem[..next_lt];
            cursor.pos += next_lt;
            if stripped_depth > 0 {
                continue;
            }
            if stack.is_empty() {
                if !text_raw.trim().is_empty() {
                    return Err(Error::Msg(
                        "vector SVG contains content outside the <svg> root".into(),
                    ));
                }
                continue;
            }
            let decoded = decode_xml_entities(text_raw);
            if !decoded.trim().is_empty() {
                if let Some((Some(parent), _)) = stack.last_mut() {
                    if parent.name == "style" {
                        let clean = sanitize_stylesheet(&decoded);
                        if !clean.is_empty() {
                            parent.children.push(SvgChild::Text(clean));
                        }
                    } else if matches!(
                        parent.name.as_str(),
                        "title" | "desc" | "text" | "tspan"
                    ) {
                        parent.children.push(SvgChild::Text(decoded));
                    }
                }
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

        // CDATA
        if let Some(after_cdata) = rem.strip_prefix("<![CDATA[") {
            let Some(end) = after_cdata.find("]]>") else {
                return Err(Error::Msg(
                    "vector SVG contains an unterminated CDATA section".into(),
                ));
            };
            let cdata = &after_cdata[..end];
            cursor.pos += 9 + end + 3;
            if stripped_depth == 0 {
                if let Some((Some(parent), _)) = stack.last_mut() {
                    if parent.name == "style" {
                        let clean = sanitize_stylesheet(cdata);
                        if !clean.is_empty() {
                            parent.children.push(SvgChild::Text(clean));
                        }
                    }
                }
            }
            continue;
        }

        // Declarations (DOCTYPE / ENTITY / etc.) and processing instructions are forbidden.
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
            let Some((node_opt, opened_name)) = stack.pop() else {
                return Err(Error::Msg(format!(
                    "vector SVG has unexpected closing tag </{name}>"
                )));
            };
            if opened_name != name {
                return Err(Error::Msg(format!(
                    "vector SVG closes </{name}> while <{opened_name}> is open"
                )));
            }
            if node_opt.is_none() {
                stripped_depth = stripped_depth.saturating_sub(1);
                continue;
            }
            let node = node_opt.expect("checked Some above");
            if let Some((Some(parent), _)) = stack.last_mut() {
                parent.children.push(SvgChild::Element(node));
            } else if stack.is_empty() {
                roots.push(node);
            }
            continue;
        }

        // Start or self-closing tag
        cursor.pos += 1;
        let name = parse_tag_name(&mut cursor)?;
        let (raw_attrs, self_closing) = parse_attributes(&mut cursor, &mut total_attrs)?;

        if stack.len() >= MAX_SVG_DEPTH {
            return Err(Error::Msg(format!(
                "vector SVG exceeds the {MAX_SVG_DEPTH}-level nesting depth budget"
            )));
        }

        if stripped_depth > 0 || is_stripped_active_element(&name) {
            if !self_closing {
                stripped_depth += 1;
                stack.push((None, name));
            }
            continue;
        }

        if !is_allowed_static_element(&name) {
            return Err(Error::Msg(format!(
                "vector SVG element <{name}> is not permitted by the static SVG allowlist"
            )));
        }

        total_elements = total_elements
            .checked_add(1)
            .ok_or_else(|| Error::Msg("vector SVG element count overflow".into()))?;
        if total_elements > MAX_SVG_ELEMENTS {
            return Err(Error::Msg(format!(
                "vector SVG exceeds the {MAX_SVG_ELEMENTS}-element budget"
            )));
        }

        let clean_attrs: Vec<(String, String)> = raw_attrs
            .into_iter()
            .filter_map(|(k, v)| sanitize_attribute(&k, &v))
            .collect();
        let node = SvgNode {
            name: name.clone(),
            attrs: clean_attrs,
            children: Vec::new(),
        };

        if self_closing {
            if let Some((Some(parent), _)) = stack.last_mut() {
                parent.children.push(SvgChild::Element(node));
            } else if stack.is_empty() {
                roots.push(node);
            }
        } else {
            stack.push((Some(node), name));
        }
    }

    if let Some((_, unclosed)) = stack.last() {
        return Err(Error::Msg(format!(
            "vector SVG is truncated: unclosed <{unclosed}> element"
        )));
    }
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

    let mut subpaths = Vec::new();
    let mut current = Vec::new();
    let mut cur = (0.0f64, 0.0f64);
    let mut start_pt = (0.0f64, 0.0f64);
    let mut cmd = 'M';
    let mut idx = 0;
    let next_num = |idx: &mut usize| -> Option<f64> {
        let val = tokens.get(*idx)?.parse::<f64>().ok()?;
        *idx += 1;
        val.is_finite().then_some(val)
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
                current.push(cur);
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
                current.push(cur);
            }
            'H' | 'h' => {
                let Some(x) = next_num(&mut idx) else { break };
                cur.0 = if cmd == 'h' { cur.0 + x } else { x };
                current.push(cur);
            }
            'V' | 'v' => {
                let Some(y) = next_num(&mut idx) else { break };
                cur.1 = if cmd == 'v' { cur.1 + y } else { y };
                current.push(cur);
            }
            'C' | 'c' => {
                let (Some(_), Some(_), Some(_), Some(_), Some(x), Some(y)) = (
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                ) else {
                    break;
                };
                cur = if cmd == 'c' {
                    (cur.0 + x, cur.1 + y)
                } else {
                    (x, y)
                };
                current.push(cur);
            }
            'Q' | 'q' | 'S' | 's' => {
                let (Some(_), Some(_), Some(x), Some(y)) = (
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                    next_num(&mut idx),
                ) else {
                    break;
                };
                cur = if cmd.is_ascii_lowercase() {
                    (cur.0 + x, cur.1 + y)
                } else {
                    (x, y)
                };
                current.push(cur);
            }
            _ => {
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
    fn strips_script_foreign_object_event_handlers_and_javascript_hrefs() {
        let hostile = r##"
<svg width="64" height="64" onload="alert(1)">
  <script>alert('xss')</script>
  <foreignObject width="64" height="64"><div>owned</div></foreignObject>
  <defs><path id="ok" d="M0 0 L10 10 Z"/></defs>
  <use href="javascript:alert(2)" onclick="evil()"/>
  <use xlink:href="https://evil.example/x.svg#a" onmouseover="evil()"/>
  <use href="#ok" style="fill:red;background:url(https://evil.example/bg.png);behavior:url(x)"/>
  <circle cx="32" cy="32" r="16" fill="blue" onerror="boom()"/>
</svg>"##;
        let canonical = sanitize_svg(hostile).expect("hostile active markup should be stripped");
        let lower = canonical.svg.to_ascii_lowercase();
        for forbidden in [
            "<script",
            "alert(",
            "foreignobject",
            "onload",
            "onclick",
            "onmouseover",
            "onerror",
            "javascript:",
            "https://evil.example",
            "behavior:",
        ] {
            assert!(
                !lower.contains(forbidden),
                "sanitized SVG leaked forbidden token {forbidden:?}: {}",
                canonical.svg
            );
        }
        assert!(canonical.svg.contains("href=\"#ok\""));
        assert!(canonical.svg.contains("fill:red"));
        assert!(canonical.svg.contains("<circle"));
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
        assert_eq!(out.terminal.sanitizer_id.as_deref(), Some(VECTOR_SANITIZER_VERSION));
        assert_eq!(out.terminal.viewport, Some([32, 32]));
        assert_eq!(out.terminal.element_count, Some(2));
        assert_eq!(
            out.terminal.provider_finish_reason.as_deref(),
            Some("complete_root")
        );
        assert_eq!(out.terminal.generated_bytes, Some(raw.len()));
        assert_eq!(out.terminal.provider_id.as_deref(), Some("mlx-starvector-1b"));
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
}
