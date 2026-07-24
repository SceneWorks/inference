//! Prompt templating, truncation and the system-prompt drop — the text-side half of the reference's
//! `_encode_texts_packed` / `_encode_edits_packed` (`_vendor/mage_flow/pipeline.py:218-231`,
//! `:396-417`) plus `PROMPT_TEMPLATE` (`models/utils.py:46-66`).
//!
//! Three values decide whether the conditioning is right, and **none of them is observable in a
//! parity golden** (every committed golden uses a short prompt), which is why all three are pinned
//! constants in [`crate::config`] rather than transcriptions here:
//!
//! | | generation | editing |
//! | --- | --- | --- |
//! | template | [`PROMPT_TEMPLATE_GEN`] | [`PROMPT_TEMPLATE_EDIT`] |
//! | `drop_idx` | 34 | 64 |
//! | truncation | 2082 | 2112 |
//!
//! The truncation budget is [`TXT_MAX_LENGTH`](crate::config::TXT_MAX_LENGTH) **plus** `drop_idx`
//! (`pipeline.py:225`), applied to the *templated* string — so 2048 conditioning tokens survive the
//! drop either way. The reference's `ModelConfig` dataclass default of 4096
//! (`models/mage_flow.py:31`) is overridden by `load_from_repo` (`pipeline.py:745`) and must not be
//! used.

use crate::config::{
    max_prompt_tokens, DROP_IDX_EDIT, DROP_IDX_GEN, PROMPT_TEMPLATE_EDIT, PROMPT_TEMPLATE_GEN,
};

/// Which `PROMPT_TEMPLATE` entry a prompt is encoded under. Selects the ChatML wrapper, the
/// number of leading system-prompt tokens dropped, and the truncation budget together — they are
/// three faces of one choice and are never mixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptKind {
    /// `PROMPT_TEMPLATE["mage-flow"]` — text-to-image (`utils.py:49-56`).
    Gen,
    /// `PROMPT_TEMPLATE["mage-flow-edit"]` — instruction editing (`utils.py:57-65`). The body is
    /// built by [`edit_body`] and carries one `<|image_pad|>` placeholder per reference image,
    /// which the Qwen3-VL **vision** tower (sc-14048) expands and fills.
    Edit,
}

impl PromptKind {
    /// The verbatim ChatML template; `{}` marks the body.
    pub fn template(self) -> &'static str {
        match self {
            Self::Gen => PROMPT_TEMPLATE_GEN,
            Self::Edit => PROMPT_TEMPLATE_EDIT,
        }
    }

    /// Leading system-prompt tokens dropped from the encoded sequence (`start_idx`).
    pub fn drop_idx(self) -> usize {
        match self {
            Self::Gen => DROP_IDX_GEN,
            Self::Edit => DROP_IDX_EDIT,
        }
    }

    /// Truncation length for the **templated** string: `TXT_MAX_LENGTH + drop_idx`.
    pub fn max_prompt_tokens(self) -> usize {
        max_prompt_tokens(self.drop_idx())
    }

    /// Wrap `body` in the template — the reference's `template.format(body)`. Exactly one `{}`
    /// placeholder is substituted, and the body is inserted literally (a `{}` inside a user prompt
    /// is not re-expanded, matching `str.format`).
    pub fn render(self, body: &str) -> String {
        self.template().replacen("{}", body, 1)
    }
}

/// Fixed per-reference image placeholder (`pipeline.py:386`).
pub const EDIT_IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// The multi-reference edit body: `Image 1: <ph>Image 2: <ph>…{instruction}`
/// (`_edit_prompt_body`, `pipeline.py:389-392`). Note there is **no separator** between the last
/// placeholder and the instruction — the numbering is 1-based and the concatenation is bare.
pub fn edit_body(instruction: &str, num_refs: usize) -> String {
    let mut out = String::new();
    for j in 1..=num_refs {
        out.push_str("Image ");
        out.push_str(&j.to_string());
        out.push_str(": ");
        out.push_str(EDIT_IMAGE_PLACEHOLDER);
    }
    out.push_str(instruction);
    out
}

/// Right-truncate `ids` to `kind`'s budget, mirroring HF `tokenizer(..., max_length=…,
/// truncation=True)` for a single sequence.
pub fn truncate(ids: &mut Vec<i32>, kind: PromptKind) {
    ids.truncate(kind.max_prompt_tokens());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TXT_MAX_LENGTH;

    /// The budget is `TXT_MAX_LENGTH + drop_idx`, not `TXT_MAX_LENGTH` and not the reference's
    /// misleading 4096 dataclass default. Written as literals so a change to either operand shows
    /// up here rather than cancelling out.
    #[test]
    fn truncation_budget_is_txt_max_length_plus_drop_idx() {
        assert_eq!(PromptKind::Gen.max_prompt_tokens(), 2082);
        assert_eq!(PromptKind::Edit.max_prompt_tokens(), 2112);
        assert_eq!(
            PromptKind::Gen.max_prompt_tokens(),
            TXT_MAX_LENGTH + PromptKind::Gen.drop_idx()
        );
        assert_ne!(PromptKind::Gen.max_prompt_tokens(), 4096);
        assert_ne!(PromptKind::Gen.max_prompt_tokens(), TXT_MAX_LENGTH);
    }

    /// Truncation happens on the templated sequence and leaves exactly `TXT_MAX_LENGTH`
    /// conditioning tokens once `drop_idx` is removed — the property the `+ drop_idx` term exists
    /// to preserve.
    #[test]
    fn an_overlong_prompt_still_yields_txt_max_length_conditioning_tokens() {
        for kind in [PromptKind::Gen, PromptKind::Edit] {
            let mut ids: Vec<i32> = (0..9000).collect();
            truncate(&mut ids, kind);
            assert_eq!(ids.len(), kind.max_prompt_tokens());
            assert_eq!(ids.len() - kind.drop_idx(), TXT_MAX_LENGTH);
        }
    }

    /// A short prompt is untouched — truncation must not pad.
    #[test]
    fn a_short_prompt_is_not_padded_or_truncated() {
        let mut ids: Vec<i32> = (0..54).collect();
        truncate(&mut ids, PromptKind::Gen);
        assert_eq!(ids.len(), 54);
    }

    /// `render` substitutes the single `{}` and inserts the body verbatim; a `{}` in the user's own
    /// prompt is not re-expanded (Python `str.format` semantics).
    #[test]
    fn render_substitutes_exactly_one_placeholder() {
        let out = PromptKind::Gen.render("a {} b");
        assert!(out.contains("<|im_start|>user\na {} b<|im_end|>"), "{out}");
        assert_eq!(
            out.matches("{}").count(),
            1,
            "only the body's braces remain"
        );
        assert!(out.ends_with("<|im_start|>assistant\n"));
        assert!(out.starts_with("<|im_start|>system\nDescribe the image"));
    }

    /// The edit body is `Image j: <ph>` repeated with **no** separator before the instruction, and
    /// the placeholder is the exact vision-token triple the processor expands.
    #[test]
    fn edit_body_matches_the_reference_layout() {
        assert_eq!(
            edit_body("make it night", 1),
            "Image 1: <|vision_start|><|image_pad|><|vision_end|>make it night"
        );
        assert_eq!(
            edit_body("blend them", 2),
            "Image 1: <|vision_start|><|image_pad|><|vision_end|>\
             Image 2: <|vision_start|><|image_pad|><|vision_end|>blend them"
        );
        assert_eq!(edit_body("nothing", 0), "nothing");
    }

    /// The two templates are genuinely different and carry different drops — a copy-paste that
    /// pointed the edit kind at the generation template would keep every other test green.
    #[test]
    fn the_two_kinds_do_not_share_a_template_or_a_drop() {
        assert_ne!(PromptKind::Gen.template(), PromptKind::Edit.template());
        assert_ne!(PromptKind::Gen.drop_idx(), PromptKind::Edit.drop_idx());
        assert_eq!(PromptKind::Gen.drop_idx(), 34);
        assert_eq!(PromptKind::Edit.drop_idx(), 64);
    }
}
