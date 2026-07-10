//! ComfyUI-compatible `(text:weight)` prompt-emphasis parsing for Anima (sc-10566).
//!
//! ## Where the weights go (the directional trap)
//! Anima's prompt weighting applies **only to the T5 query-token path** — the per-token weights scale
//! the `AnimaTextConditioner` (LLM-adapter) OUTPUT vectors — while the **Qwen** token weights are
//! forced to `1.0` (a strict no-op on the Qwen tower). This is dictated by the reference
//! implementation:
//!
//! - ComfyUI `comfy/text_encoders/anima.py`, `AnimaTokenizer.tokenize_with_weights` (lines 23-28):
//!   the Qwen (`qwen3_06b`) token list is rebuilt as `(id, 1.0)` — *"Set weights to 1.0"* — while
//!   `out["t5xxl"]` keeps the parsed weights unchanged.
//! - ComfyUI `comfy/text_encoders/anima.py`, `AnimaTEModel.encode_token_weights` (lines 48-52):
//!   `t5xxl_ids` and `t5xxl_weights` are carried as parallel per-token tensors.
//! - ComfyUI `comfy/ldm/anima/model.py`, `Anima.preprocess_text_embeds` (lines 198-206):
//!   `out = self.llm_adapter(text_embeds, text_ids); if t5xxl_weights is not None: out = out *
//!   t5xxl_weights` — the weights multiply the adapter **output**, per token, before the pad-to-512.
//! - ComfyUI `comfy/model_base.py`, `Anima.extra_conds` (line 1470): the weight tensor is reshaped
//!   `t5xxl_weights.unsqueeze(0).unsqueeze(-1)` → `[1, St, 1]`, i.e. each token's full output vector
//!   is scaled by its scalar weight (`out[:, i, :] *= w[i]`).
//!
//! ## The grammar (matched, not copied)
//! The functions below are a **clean-room Rust reimplementation** of the emphasis grammar in ComfyUI
//! `comfy/sd1_clip.py` (`parse_parentheses` L320-346, `token_weights` L348-366, `escape_important` /
//! `unescape_important` L368-375). ComfyUI is GPL-3.0 and mlx-gen is Apache-2.0, so **nothing is
//! copied** — only the observable grammar is reproduced:
//!
//! - `(text)`      → weight ×1.1, nestable (`((text))` → ×1.21).
//! - `(text:1.5)`  → weight = 1.5 (absolute; overrides the ×1.1 bump **and** any parent weight).
//! - `\(` / `\)`   → literal parentheses (escaped out of the emphasis grammar, restored afterwards).
//! - `[text]`      → **NOT** special. ComfyUI parses only round parens; square brackets are literal
//!   text (verified against `sd1_clip.py`, which never inspects `[`/`]`). A1111-style `[de-emphasis]`
//!   is not part of this grammar.
//! - Malformed weight (`(text:abc)`) → the `float(...)` parse fails and the ×1.1 bump stands
//!   (ComfyUI's `try/except: pass`); unbalanced parens degrade to literal text. Never panics.

/// Placeholder bytes for escaped `)` / `(` (ComfyUI uses the same `\0\1` / `\0\2` sentinels so an
/// escaped paren is invisible to the paren splitter, then restored before tokenizing).
const ESC_CLOSE: &str = "\u{0}\u{1}";
const ESC_OPEN: &str = "\u{0}\u{2}";

fn escape_important(text: &str) -> String {
    text.replace("\\)", ESC_CLOSE).replace("\\(", ESC_OPEN)
}

fn unescape_important(text: &str) -> String {
    text.replace(ESC_CLOSE, ")").replace(ESC_OPEN, "(")
}

/// Split a string into top-level segments: each balanced `(...)` group is one segment (parens
/// included), and each run of characters outside a top-level group is its own segment. Faithful port
/// of ComfyUI `parse_parentheses` — including its treatment of unbalanced parens (a stray `)` at
/// nesting 0 is kept as literal text rather than erroring).
fn parse_parentheses(s: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut nesting: i32 = 0;
    for ch in s.chars() {
        match ch {
            '(' => {
                if nesting == 0 && !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
                current.push('(');
                nesting += 1;
            }
            ')' => {
                nesting -= 1;
                if nesting == 0 {
                    current.push(')');
                    result.push(std::mem::take(&mut current));
                } else {
                    current.push(')');
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

/// Recursively resolve emphasis into `(segment_text, weight)` pairs. Faithful port of ComfyUI
/// `token_weights`: a `(...)`-wrapped segment multiplies the current weight by `1.1`, an explicit
/// `:weight` suffix (last `:`) sets the weight **absolutely**, and both recurse into the inner text so
/// nesting composes; a bad `:weight` leaves the ×1.1 bump in place.
fn token_weights(s: &str, current_weight: f32) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    for seg in parse_parentheses(s) {
        if seg.len() >= 2 && seg.starts_with('(') && seg.ends_with(')') {
            let mut inner = seg[1..seg.len() - 1].to_string();
            let mut weight = current_weight * 1.1;
            if let Some(pos) = inner.rfind(':') {
                if pos > 0 {
                    if let Ok(w) = inner[pos + 1..].trim().parse::<f32>() {
                        weight = w;
                        inner.truncate(pos);
                    }
                }
            }
            out.extend(token_weights(&inner, weight));
        } else {
            out.push((seg, current_weight));
        }
    }
    out
}

/// Parse `prompt` into `(text_span, weight)` segments (weight `1.0` = no emphasis). Escaped parens are
/// restored to literal `(`/`)` in the returned spans. Concatenating the spans reproduces the prompt
/// with the emphasis syntax removed (see [`strip_prompt_weights`]).
pub fn parse_prompt_weights(prompt: &str) -> Vec<(String, f32)> {
    let escaped = escape_important(prompt);
    token_weights(&escaped, 1.0)
        .into_iter()
        .map(|(seg, w)| (unescape_important(&seg), w))
        .collect()
}

/// The prompt with all emphasis syntax removed — what the **Qwen** tower (weight-blind, forced to
/// `1.0`) is tokenized on. For a prompt with no emphasis this is the input unchanged.
pub fn strip_prompt_weights(prompt: &str) -> String {
    parse_prompt_weights(prompt)
        .into_iter()
        .map(|(s, _)| s)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_one_unit_weighted_segment() {
        assert_eq!(
            parse_prompt_weights("1girl, silver hair"),
            vec![("1girl, silver hair".to_string(), 1.0)]
        );
        assert_eq!(
            strip_prompt_weights("1girl, silver hair"),
            "1girl, silver hair"
        );
    }

    #[test]
    fn explicit_weight_is_absolute() {
        // `(chibi:2)` → the span `chibi` at weight 2.0; surrounding text stays 1.0.
        let p = parse_prompt_weights("1girl, (chibi:2), masterpiece");
        assert_eq!(
            p,
            vec![
                ("1girl, ".to_string(), 1.0),
                ("chibi".to_string(), 2.0),
                (", masterpiece".to_string(), 1.0),
            ]
        );
        // The de-weighted text (Qwen input) is the prompt minus the emphasis syntax.
        assert_eq!(
            strip_prompt_weights("1girl, (chibi:2), masterpiece"),
            "1girl, chibi, masterpiece"
        );
    }

    #[test]
    fn bare_parens_multiply_by_1_1_and_nest() {
        assert_eq!(
            parse_prompt_weights("(chibi)"),
            vec![("chibi".to_string(), 1.1)]
        );
        // ((chibi)) → 1.1 * 1.1 = 1.21.
        let p = parse_prompt_weights("((chibi))");
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].0, "chibi");
        assert!((p[0].1 - 1.21).abs() < 1e-5, "got {}", p[0].1);
    }

    #[test]
    fn nested_explicit_weight_overrides_parent() {
        // `(a (b:2) c:3)` → a@3, b@2 (absolute, parent-independent), c@3.
        let p = parse_prompt_weights("(a (b:2) c:3)");
        assert_eq!(
            p,
            vec![
                ("a ".to_string(), 3.0),
                ("b".to_string(), 2.0),
                (" c".to_string(), 3.0),
            ]
        );
    }

    #[test]
    fn weight_one_is_identity() {
        assert_eq!(
            parse_prompt_weights("(chibi:1.0)"),
            vec![("chibi".to_string(), 1.0)]
        );
    }

    #[test]
    fn escaped_parens_are_literal() {
        // `\(` / `\)` are literal characters, not emphasis.
        let p = parse_prompt_weights("a \\(smile\\) b");
        assert_eq!(p, vec![("a (smile) b".to_string(), 1.0)]);
        assert_eq!(strip_prompt_weights("a \\(smile\\) b"), "a (smile) b");
    }

    #[test]
    fn square_brackets_are_literal_not_deemphasis() {
        // ComfyUI does NOT treat `[...]` as emphasis — it is literal text at weight 1.0.
        let p = parse_prompt_weights("[chibi]");
        assert_eq!(p, vec![("[chibi]".to_string(), 1.0)]);
    }

    #[test]
    fn malformed_weight_keeps_the_1_1_bump_and_never_panics() {
        // `(chibi:abc)` → float parse fails → the ×1.1 bump stands, text unchanged.
        let p = parse_prompt_weights("(chibi:abc)");
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].0, "chibi:abc");
        assert!((p[0].1 - 1.1).abs() < 1e-5, "got {}", p[0].1);
        // Unbalanced parens degrade to literal text without panicking.
        let _ = parse_prompt_weights("a (b");
        let _ = parse_prompt_weights("a )b(");
        let _ = parse_prompt_weights("((((");
        let _ = parse_prompt_weights("))))");
    }

    #[test]
    fn colon_at_index_zero_is_not_a_weight() {
        // `(:2)` — the `:` is at index 0, so it is NOT parsed as a weight (ComfyUI `xx > 0`).
        let p = parse_prompt_weights("(:2)");
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].0, ":2");
        assert!((p[0].1 - 1.1).abs() < 1e-5, "got {}", p[0].1);
    }

    #[test]
    fn empty_prompt_yields_nothing() {
        assert!(parse_prompt_weights("").is_empty());
        assert_eq!(strip_prompt_weights(""), "");
    }
}
