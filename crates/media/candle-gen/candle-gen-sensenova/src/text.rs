//! The `neo1_0` conversation template, tokenizer loading, and the (t,h,w) position-index builders —
//! the candle port of `mlx-gen-sensenova`'s `text.rs`.
//!
//! SenseNova-U1 uses a Qwen2/3 byte-level BPE tokenizer (vocab 151936) with NEO-Unify's added
//! special tokens (`<img>`/`</img>`/`<IMG_CONTEXT>`, `<think>`/`</think>`, the ChatML markers). The
//! snapshot ships only `vocab.json` + `merges.txt` + `added_tokens.json`, so — mirroring the
//! Qwen-Image / mlx provider — a fast `tokenizer.json` is materialized into the snapshot by
//! `tools/build_sensenova_tokenizer.py`; [`SenseNovaTokenizer::from_dir`] reads it.
//!
//! The `neo1_0` template is ChatML (the reference `conversation.py` MPT style): an optional system
//! block, the user turn, and the empty assistant turn that primes generation. Image generation
//! prepends [`SYSTEM_MESSAGE_FOR_GEN`]. The Generator contract only ever drives the **non-think** T2I
//! path (think-mode is an internal flag the registry path leaves off), so this slice needs neither
//! the AR text decode nor the image-token-id constants the it2i/think/interleave surfaces use.

use std::path::{Path, PathBuf};

use candle_gen::{CandleError, Result};
use tokenizers::Tokenizer;

/// The quant-matrix sibling tier dirs a tokenizer may be borrowed from, in preference order (sc-14432).
/// A model's fast tokenizer is identical across its quant tiers (same vocab), so when a tier ships no
/// `tokenizer.json` of its own, any sibling tier's copy is byte-correct. `q8` first because it is the
/// tier that reliably carries it (the base `SceneWorks/sensenova-u1-8b-mlx` repo shipped it ONLY in
/// `q8/`, which is the whole defect); the order also makes resolution deterministic when more than one
/// sibling has it.
const SIBLING_TIER_DIRS: [&str; 3] = ["q8", "bf16", "q4"];

/// NEO-Unify special-token ids (from the snapshot's `added_tokens.json`). Used by the understanding
/// surface (VQA / interleave) — the `<IMG_CONTEXT>` splice target, the `<img>`/`</img>` image
/// boundary, and the ChatML / think stop ids the AR decode watches for. The full set is kept as the
/// documented reference vocab (the candle slice consumes only a subset today).
#[allow(dead_code)]
pub mod tokens {
    pub const ENDOFTEXT: i32 = 151643;
    pub const IM_START: i32 = 151644;
    pub const IM_END: i32 = 151645;
    pub const THINK: i32 = 151667;
    pub const THINK_END: i32 = 151668;
    pub const IMG_CONTEXT: i32 = 151669;
    /// `<img>` — the reference's `img_start_token_id`.
    pub const IMG_START: i32 = 151670;
    /// `</img>`.
    pub const IMG_END: i32 = 151671;
    pub const PAD: i32 = 151643;
}

/// The image-generation system message (verbatim from the reference `utils.SYSTEM_MESSAGE_FOR_GEN`).
pub const SYSTEM_MESSAGE_FOR_GEN: &str = concat!(
    "You are an image generation and editing assistant that accurately understands and executes ",
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you ",
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. ",
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final ",
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n",
    "Task Types:\n\nA. Text-to-Image Generation:\n",
    "- Generate a high-quality image based on the user's description.\n",
    "- Ensure visual clarity, semantic consistency, and completeness.\n",
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n",
    "B. Image Editing:\n",
    "- Use the provided image(s) as input or reference for modification or transformation.\n",
    "- The result can be an edited image or a new image based on the reference(s).\n",
    "- Preserve all unspecified attributes unless explicitly changed.\n\n",
    "General Rules:\n",
    "- For any visible text in the image, follow the language specified for the rendered text in ",
    "the user's description, not the language of the prompt. If no language is specified, use the ",
    "user's input language."
);

/// The interleaved text-image system message (verbatim from the reference
/// `examples/interleave/inference.py::DEFAULT_SYSTEM_MESSAGE`) — required for Document Studio's
/// think-mode interleave protocol or the model won't interleave correctly.
pub const INTERLEAVE_SYSTEM_MESSAGE: &str = concat!(
    "You are a multimodal assistant capable of reasoning with both text and images. You support ",
    "two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> ",
    "block and place all reasoning inside it. You MUST interleave text with generated images using ",
    "tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and ",
    "may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, directly ",
    "provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any ",
    "images naturally alongside the text.\n\nAfter the think block, always provide a concise, ",
    "user-facing final answer. The answer may include text, images, or both. Match the user's ",
    "language in both reasoning and the final answer."
);

/// Build the `neo1_0` ChatML prompt: optional system block + the user turn + the empty assistant
/// turn that primes generation. Mirrors the reference `conversation.py` MPT style — an empty
/// `system_message` omits the system block entirely.
pub fn build_neo1_query(prompt: &str, system_message: &str) -> String {
    let mut s = String::new();
    if !system_message.is_empty() {
        s.push_str("<|im_start|>system\n");
        s.push_str(system_message);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>user\n");
    s.push_str(prompt);
    s.push_str("<|im_end|>\n<|im_start|>assistant\n");
    s
}

/// The three position rows for a run of `len` **text** tokens: temporal = `0..len`, height = width
/// = 0 (the reference `_build_t2i_text_inputs`).
pub fn text_indexes(len: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let t = (0..len as i32).collect();
    let zeros = vec![0i32; len];
    (t, zeros.clone(), zeros)
}

/// The three position rows for a `token_h × token_w` image block placed after `text_len` text
/// tokens: temporal = `text_len` (all image tokens share one block index → bidirectional attention),
/// height = `idx / token_w`, width = `idx % token_w` (row-major; the reference
/// `_build_t2i_image_indexes`).
pub fn image_indexes(
    token_h: usize,
    token_w: usize,
    text_len: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let n = token_h * token_w;
    let mut t = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        t.push(text_len as i32);
        h.push((i / token_w) as i32);
        w.push((i % token_w) as i32);
    }
    (t, h, w)
}

/// The SenseNova-U1 prompt tokenizer (Qwen2/3 byte-level BPE + NEO-Unify added tokens). The crate
/// builds the neo1_0 prompt strings itself and tokenizes them here; the added special tokens
/// (`<|im_start|>`, `<think>`, `<img>`, …) are recognized atomically from the materialized
/// `tokenizer.json`.
pub struct SenseNovaTokenizer {
    inner: Tokenizer,
}

impl SenseNovaTokenizer {
    /// Load the fast tokenizer for the tier dir `root` (materialized by
    /// `tools/build_sensenova_tokenizer.py` from the snapshot's `vocab.json` + `merges.txt`).
    ///
    /// Prefers `<root>/tokenizer.json`, and falls back to a **sibling tier's** copy when this tier
    /// ships none (sc-14432 — `resolve_tokenizer_path`). The base `SceneWorks/sensenova-u1-8b-mlx`
    /// re-host shipped `tokenizer.json` ONLY in `q8/`, so its `q4/` and `bf16/` tiers reported
    /// "complete" (the `<tier>/*` download glob resolved fine) yet failed to load here. Borrowing a
    /// sibling's is byte-correct — the tokenizer is model-wide, identical across quant tiers.
    pub fn from_dir(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let path = resolve_tokenizer_path(root).ok_or_else(|| {
            CandleError::Msg(format!(
                "missing tokenizer.json under {} (and no sibling q4/q8/bf16 tier provides one): the \
                 SenseNova-U1 snapshot ships only vocab.json + merges.txt; run \
                 tools/build_sensenova_tokenizer.py to materialize the fast tokenizer.json",
                root.display()
            ))
        })?;
        let inner = Tokenizer::from_file(&path)
            .map_err(|e| CandleError::Msg(format!("sensenova: load tokenizer.json: {e}")))?;
        Ok(Self { inner })
    }

    /// Tokenize `text` → `i32` ids. `add_special` controls the post-processor's template tokens;
    /// the NEO-Unify added tokens embedded in the prompt string are split out regardless.
    pub fn encode_ids(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        let enc = self
            .inner
            .encode(text, add_special)
            .map_err(|e| CandleError::Msg(format!("sensenova: tokenize: {e}")))?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    }

    /// Decode token `ids` back to text. `skip_special` drops the NEO-Unify / ChatML special tokens
    /// (the understanding decode passes `true` so the answer text excludes `<|im_end|>` etc.).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| CandleError::Msg(format!("sensenova: detokenize: {e}")))
    }
}

/// Resolve the `tokenizer.json` to load for the tier dir `root` (sc-14432).
///
/// `<root>/tokenizer.json` wins. If this tier ships none, borrow a **sibling tier's** — a
/// quant-matrix turnkey lays `q4/`, `q8/`, `bf16/` side by side under one snapshot, and the fast
/// tokenizer is model-wide (byte-identical across quant tiers), so a sibling's copy is correct. Only
/// the fixed [`SIBLING_TIER_DIRS`] names under the same parent are consulted — never an arbitrary
/// directory, and never a path above the snapshot — so this cannot pull a *different* model's
/// tokenizer. Returns `None` (→ a loud, actionable error at the call site) when nothing carries one,
/// e.g. a flat single-tier snapshot with no siblings.
pub(crate) fn resolve_tokenizer_path(root: &Path) -> Option<PathBuf> {
    let own = root.join("tokenizer.json");
    if own.is_file() {
        return Some(own);
    }
    let parent = root.parent()?;
    for tier in SIBLING_TIER_DIRS {
        let sibling_dir = parent.join(tier);
        if sibling_dir == root {
            continue; // don't rediscover our own (already-missing) tier
        }
        let candidate = sibling_dir.join("tokenizer.json");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pid-scoped scratch dir (this crate carries no `tempfile` dep — see `quant.rs`).
    fn scratch(tmp: &tempfile::TempDir, tag: &str) -> PathBuf {
        let dir = tmp.path().join(format!("sensenova-tok-{}", tag));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"{}").unwrap();
    }

    /// sc-14432: a tier ships its own `tokenizer.json` → use it; a tier that ships NONE borrows a
    /// sibling tier's (the base `sensenova-u1-8b-mlx` repo puts it only in `q8/`); nothing anywhere →
    /// `None` (a loud error at the call site, never a wrong-model tokenizer).
    #[test]
    fn resolve_tokenizer_path_prefers_own_then_borrows_a_sibling_tier() {
        let tmp = tempfile::tempdir().unwrap();
        let snap = scratch(&tmp, "resolve");

        // q4 has its own → own wins even with a sibling present.
        touch(&snap.join("q4/tokenizer.json"));
        touch(&snap.join("q8/tokenizer.json"));
        assert_eq!(
            resolve_tokenizer_path(&snap.join("q4")),
            Some(snap.join("q4/tokenizer.json"))
        );

        // bf16 ships none → borrow q8's (the sc-14432 defect shape: only q8 has it).
        assert_eq!(
            resolve_tokenizer_path(&snap.join("bf16")),
            Some(snap.join("q8/tokenizer.json")),
            "a tier without its own tokenizer.json must borrow a sibling tier's"
        );

        // No sibling carries one → None (a flat/torn snapshot with nothing to borrow).
        let bare = scratch(&tmp, "bare");
        std::fs::create_dir_all(bare.join("q4")).unwrap();
        assert_eq!(resolve_tokenizer_path(&bare.join("q4")), None);
    }

    #[test]
    fn neo1_query_empty_system_has_no_system_block() {
        let q = build_neo1_query("a fox", "");
        assert_eq!(
            q,
            "<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn neo1_query_with_system_block() {
        let q = build_neo1_query("a fox", "SYS");
        assert_eq!(
            q,
            "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn text_indexes_are_causal_positions() {
        let (t, h, w) = text_indexes(4);
        assert_eq!(t, vec![0, 1, 2, 3]);
        assert_eq!(h, vec![0, 0, 0, 0]);
        assert_eq!(w, vec![0, 0, 0, 0]);
    }

    #[test]
    fn image_indexes_are_grid_positions_after_text() {
        // 2×3 grid placed after 5 text tokens.
        let (t, h, w) = image_indexes(2, 3, 5);
        assert_eq!(t, vec![5, 5, 5, 5, 5, 5]);
        assert_eq!(h, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(w, vec![0, 1, 2, 0, 1, 2]);
    }
}
