//! The FLUX 3 Image Klein **prompt upsampler** (story sc-26238 of epic sc-26233): the
//! tensor-neutral contract for the separate `black-forest-labs/flux-3-image-klein-pup` package —
//! a 9B Qwen3.5 vision-language model that rewrites the user's short request into the caption
//! the image model was trained on, in four modes selected by its system prompt (`t2i`, `i2i`,
//! `t2i_bbox`, `i2i_bbox`).
//!
//! Everything here is host-side: the frozen identity and sampling policy, the system prompts
//! (verbatim), the two editing scaffolds, the box-annotation grammar of `bbox.py` (canvas rows,
//! the `ORIGINAL PROMPT:` / `LOCKED BOUNDING-BOX ANNOTATIONS` input, detection of a prompt that
//! already carries a table, and the reconcile step that re-imposes the user's locked boxes on
//! the model's answer), the reference-image policy (`policy_reference`), the input contract the
//! editing (sc-26236) and layout (sc-26237) stories compose onto, the token budget, and the
//! driver that runs a plan through any [`TextLlm`] with cancellation and progress routed to the
//! generator's [`CancelFlag`] / [`Progress`]. The MLX provider (`mlx-gen-flux3::pup`) and the
//! Candle twin only load the VLM and hand it to [`run_text_llm`].
//!
//! # Frozen upstream
//!
//! * Code: `src/flux3/pup.py` + `src/flux3/bbox.py` of GitHub
//!   `black-forest-labs/flux-3-image-klein` @ [`super::UPSTREAM_CODE_REVISION`]
//!   ([`PUP_PY_SHA256`], [`BBOX_PY_SHA256`]). `generate.py` decides when the upsampler runs:
//!   `--no-pup` sends the prompt as typed (a box list, if any, appended on the 0–1000 canvas), a
//!   *finished* answer (a prompt whose box table is fully cited as `<id>`s) goes to the DiT
//!   untouched, and a `ValueError` inside the upsampler (a reference beyond the aspect limit)
//!   falls back to the prompt as typed.
//! * Package: Hugging Face [`UPSAMPLER_REPO`] @ [`UPSAMPLER_REVISION`] (ungated, Apache-2.0,
//!   `model_type: qwen3_5`, a full fine-tune of `Qwen/Qwen3.5-9B` with the vision tower frozen).
//!   Its `system_prompts.json` is byte-identical to the source tree's `pup_prompts.json`
//!   ([`SYSTEM_PROMPTS_SHA256`]); the copy embedded here ([`SYSTEM_PROMPTS_JSON`]) is the
//!   fallback upstream uses when the package ships none.
//!
//! # Fidelity notes
//!
//! Text is byte-for-byte: the system prompts, the scaffolds, the chat rendering (`enable_thinking`
//! off → the empty `<think>` block), the canvas-row JSON (Python's `json.dumps` separators and
//! key order — [`pyjson`]). Upstream serves the model through vLLM with greedy decoding
//! (`temperature 0.0`, `top_p 0.95`, `top_k 20`, `max_tokens 3584`, presence penalty `1.5` for
//! text-to-image and `0.0` for edits, `max_model_len 8192`); the native path hands the same
//! policy to the engine's sampler. The reference policy (EXIF transpose, alpha over white, a
//! size budget by reference count, Pillow `LANCZOS`, JPEG quality 92) is reproduced bit for
//! bit: the shared Pillow-exact resampler, Pillow's alpha-compositing fixed point, and
//! libjpeg's own integer codec path ([`jpeg`]) for the quality-92 round trip. The per-reference
//! pixel preprocessing of the VLM itself (`smart_resize`, patchify, normalize) is the serving
//! engine's Qwen processor, as upstream's is vLLM's.
//!
//! [`TextLlm`]: gen_core::core_llm::TextLlm

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use gen_core::core_llm::{
    self, Channel, ChatTemplate, Content, FinishReason, ImageRef, JinjaChatTemplate, Message,
    RenderOptions, Role, Sampling, StreamEvent, TextLlm, TextLlmRequest, ThinkingMode,
};
use gen_core::runtime::{
    CancelFlag, LoadSpec, Progress, PromptEnhancementOutcome, PromptEnhancementReport,
    WeightsSource,
};
use gen_core::{Error, Result};

pub mod jpeg;
pub mod pyjson;

pub use gen_core::flux3::pup::{is_row_origin, MAX_TOKENS};
pub use pyjson::{dumps, loads, JsonValue};

// ---------------------------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------------------------

/// Hugging Face repository of the upsampler package.
pub const UPSAMPLER_REPO: &str = "black-forest-labs/flux-3-image-klein-pup";
/// Pinned revision of the upsampler package (verified readable and ungated on 2026-10-15).
pub const UPSAMPLER_REVISION: &str = "8b069751eb5dcab23829d9c5eccb470eb8b8a4ed";
/// sha256 of the pinned package's `model.safetensors` (the Hub's linked-object digest,
/// `X-Linked-ETag`, 18,819,722,392 bytes; bf16).
pub const UPSAMPLER_WEIGHTS_SHA256: &str =
    "f5062a1fba3e63afe5c88349a311903fec06c959b362f1247127e63d5c860ed4";
/// Byte size of the pinned `model.safetensors`.
pub const UPSAMPLER_WEIGHTS_BYTES: u64 = 18_819_722_392;
/// The model the package is a full fine-tune of (card `base_model`).
pub const UPSAMPLER_BASE_MODEL: &str = "Qwen/Qwen3.5-9B";
/// The package's `config.json` `model_type` — the hybrid linear/full-attention Qwen3.5 decoder
/// with the Qwen3-VL vision tower, which both serving engines already load.
pub const UPSAMPLER_MODEL_TYPE: &str = "qwen3_5";
/// The package's `config.json` `architectures[0]`.
pub const UPSAMPLER_ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
/// sha256 of the frozen `src/flux3/pup.py`.
pub const PUP_PY_SHA256: &str = "e3561ae3cbff1a080ad8854c0b89bd0687c1ea9d84c74f5203b44844cbd4684d";
/// sha256 of the frozen `src/flux3/bbox.py`.
pub const BBOX_PY_SHA256: &str = "fbea10c1c53980df1665d3cb6cc6264b8f50633399bf6809c91650f1dcf4fc37";
/// sha256 of the package's `system_prompts.json` — identical to the source tree's
/// `src/flux3/pup_prompts.json` and to [`SYSTEM_PROMPTS_JSON`].
pub const SYSTEM_PROMPTS_SHA256: &str =
    "b5da3ed8bc04ab59dbdb747930f7dbff847e868903fed8028f85ee1a4ac82fb4";
/// The four system prompts, verbatim (`{"t2i": …, "i2i": …, "t2i_bbox": …, "i2i_bbox": …}`).
pub const SYSTEM_PROMPTS_JSON: &str = include_str!("pup_system_prompts.json");
/// The file the package ships its prompts in (read in preference to the embedded copy, as
/// upstream does).
pub const SYSTEM_PROMPTS_FILE: &str = "system_prompts.json";
/// `LoadSpec::components` key carrying the upsampler package directory. **Optional** (E5): the
/// upsampler is a co-requisite accounted separately from the image model and its text encoder;
/// a request with `enhance_prompt` on a generator loaded without it is a typed refusal
/// ([`UpsampleError::MissingUpsampler`]), never a silent pass-through.
pub const UPSAMPLER_COMPONENT: &str = "prompt_upsampler";
/// Files the package must carry (the weights, the Transformers config, the tokenizer with its
/// sidecar chat template, and the image-processor config the serving engine reads).
pub const UPSAMPLER_REQUIRED_FILES: [&str; 6] = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
];

// ---------------------------------------------------------------------------------------------
// Source constants (`pup.py`, `bbox.py`)
// ---------------------------------------------------------------------------------------------

/// `pup.SAMPLING["temperature"]` — greedy.
pub const TEMPERATURE: f32 = 0.0;
/// `pup.SAMPLING["top_p"]`.
pub const TOP_P: f32 = 0.95;
/// `pup.SAMPLING["top_k"]`.
pub const TOP_K: usize = 20;
/// `pup.SAMPLING["min_p"]` (no-op at `0.0`; the contract sampler has no min-p knob).
pub const MIN_P: f32 = 0.0;
/// Product admission bound for the COMPLETE effective caption and locked-annotation metadata,
/// in Unicode scalar values (`str::chars`), not UTF-8 bytes or an assumed token/character ratio.
/// Applied after reconciliation and before any image encoding/rendering; oversized locked rows
/// are rejected actionably, never truncated. Bypass keeps the caller's original prompt unchanged.
pub const MAX_EFFECTIVE_CAPTION_CHARS: usize = 16_000;

/// Validate text the producer will hand to the renderer/persistence consumer.
pub fn validate_caption_size(text: &str, field: &'static str) -> Result<()> {
    let chars = text.chars().count();
    if chars > MAX_EFFECTIVE_CAPTION_CHARS {
        return Err(UpsampleError::CaptionTooLong {
            field,
            chars,
            max: MAX_EFFECTIVE_CAPTION_CHARS,
        }
        .into());
    }
    Ok(())
}
/// `pup.PRESENCE_PENALTY["t2i"]`.
pub const PRESENCE_PENALTY_T2I: f32 = 1.5;
/// `pup.PRESENCE_PENALTY["i2i"]`.
pub const PRESENCE_PENALTY_I2I: f32 = 0.0;
/// `pup.ENGINE["max_model_len"]` — prompt plus generation must fit this window.
pub const MAX_MODEL_LEN: usize = 8192;
/// `pup.MAX_REFERENCES`.
pub const MAX_REFERENCES: usize = 10;
/// `pup.STOP_TOKENS` — both are EOS ids of the package's `generation_config.json`.
pub const STOP_TOKENS: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];
/// `pup.REFERENCE_MULTIPLE` — reference sides are multiples of this.
pub const REFERENCE_MULTIPLE: u32 = 32;
/// `pup.MAX_ASPECT_RATIO` — a reference beyond this is a `ValueError` (upsampling skipped).
pub const MAX_ASPECT_RATIO: f64 = 200.0;
/// `pup.JPEG_QUALITY` — the reference is re-encoded at this quality before the VLM sees it.
pub const JPEG_QUALITY: u8 = 92;
/// `pup.I2I_SCAFFOLD[False]` — the editing user text without locked boxes (`{instruction}` is
/// the user's prompt).
pub const I2I_SCAFFOLD_PLAIN: &str = "Reference images follow in order (the first image, the second image, ...).\nUser request: {instruction}\nOutput the caption.";
/// `pup.I2I_SCAFFOLD[True]` — the editing user text with locked boxes.
pub const I2I_SCAFFOLD_LOCKED: &str = "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\nUser editing instruction: {instruction}\nDecide the edit, imagine the target image, and output the annotation record.";
/// `bbox.SCALE` — the canvas the rows live on.
pub const CANVAS_SCALE: i64 = 1000;
/// The chat-template text one reference image renders to before the serving engine expands the
/// `<|image_pad|>` to the image's patch count (the Qwen3-VL processor's placeholder).
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
/// The `preprocessor_config.json` geometry the engine's Qwen image processor applies to every
/// reference: patch 16 × merge 2 = one token per 32×32, `shortest_edge` (min pixels) 65536,
/// `longest_edge` (max pixels) 16777216.
pub const VISION_TOKEN_SIDE: u32 = 32;
/// See [`VISION_TOKEN_SIDE`].
pub const VISION_MIN_PIXELS: u64 = 65_536;
/// See [`VISION_TOKEN_SIDE`].
pub const VISION_MAX_PIXELS: u64 = 16_777_216;

/// `pup.MAX_PIXELS_BY_REFS[n]` — the pixel budget of each reference when `n` are attached:
/// 1 MP for one, 724² for two, 512² for three to ten.
pub fn max_pixels_by_refs(n_refs: usize) -> Option<u64> {
    match n_refs {
        1 => Some(1024 * 1024),
        2 => Some(724 * 724),
        3..=MAX_REFERENCES => Some(512 * 512),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Modes and system prompts
// ---------------------------------------------------------------------------------------------

/// The four upstream modes — the system prompt key `bbox.plan` selects from the task (`t2i`
/// without references, `i2i` with) and whether boxes are locked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UpsampleMode {
    /// `t2i`: text to image.
    Text,
    /// `i2i`: editing with one to ten references.
    Editing,
    /// `t2i_bbox`: text to image with locked boxes.
    TextWithBoxes,
    /// `i2i_bbox`: editing with locked boxes.
    EditingWithBoxes,
}

impl UpsampleMode {
    pub const ALL: [UpsampleMode; 4] = [
        UpsampleMode::Text,
        UpsampleMode::Editing,
        UpsampleMode::TextWithBoxes,
        UpsampleMode::EditingWithBoxes,
    ];

    /// The `system_prompts.json` key (`bbox.plan`'s first return value).
    pub fn system_key(self) -> &'static str {
        match self {
            UpsampleMode::Text => "t2i",
            UpsampleMode::Editing => "i2i",
            UpsampleMode::TextWithBoxes => "t2i_bbox",
            UpsampleMode::EditingWithBoxes => "i2i_bbox",
        }
    }

    pub fn from_system_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.system_key() == key)
    }

    /// `pup.Upsampler.__call__`'s `task`: `i2i` with references, else `t2i`.
    pub fn task(self) -> &'static str {
        if self.uses_references() {
            "i2i"
        } else {
            "t2i"
        }
    }

    pub fn uses_references(self) -> bool {
        matches!(self, UpsampleMode::Editing | UpsampleMode::EditingWithBoxes)
    }

    pub fn uses_boxes(self) -> bool {
        matches!(
            self,
            UpsampleMode::TextWithBoxes | UpsampleMode::EditingWithBoxes
        )
    }

    /// `pup.PRESENCE_PENALTY[task]`.
    pub fn presence_penalty(self) -> f32 {
        if self.uses_references() {
            PRESENCE_PENALTY_I2I
        } else {
            PRESENCE_PENALTY_T2I
        }
    }

    /// The mode upstream derives from the inputs: references select the task, boxes (passed or
    /// detected in the prompt) select the `_bbox` variant.
    pub fn infer(n_references: usize, boxes: bool) -> Self {
        match (n_references > 0, boxes) {
            (false, false) => UpsampleMode::Text,
            (true, false) => UpsampleMode::Editing,
            (false, true) => UpsampleMode::TextWithBoxes,
            (true, true) => UpsampleMode::EditingWithBoxes,
        }
    }
}

impl fmt::Display for UpsampleMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.system_key())
    }
}

/// The four system prompts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemPrompts {
    pub t2i: String,
    pub i2i: String,
    pub t2i_bbox: String,
    pub i2i_bbox: String,
}

impl SystemPrompts {
    /// Parse a `system_prompts.json` (every key required, each a string).
    pub fn parse(json: &str) -> Result<Self> {
        let value =
            loads(json).map_err(|e| Error::Msg(format!("flux3 upsampler system prompts: {e}")))?;
        let key = |k: &str| -> Result<String> {
            value
                .get(k)
                .and_then(JsonValue::as_str)
                .map(str::to_string)
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "flux3 upsampler system prompts: missing or non-string key {k:?}"
                    ))
                })
        };
        Ok(Self {
            t2i: key("t2i")?,
            i2i: key("i2i")?,
            t2i_bbox: key("t2i_bbox")?,
            i2i_bbox: key("i2i_bbox")?,
        })
    }

    /// The embedded copy of the frozen prompts.
    pub fn builtin() -> &'static SystemPrompts {
        static BUILTIN: OnceLock<SystemPrompts> = OnceLock::new();
        BUILTIN.get_or_init(|| {
            SystemPrompts::parse(SYSTEM_PROMPTS_JSON).expect("the embedded prompts parse")
        })
    }

    /// Upstream's choice: the package's own [`SYSTEM_PROMPTS_FILE`] when it ships one, else the
    /// bundled prompts.
    pub fn from_package(dir: &Path) -> Result<Self> {
        let shipped = dir.join(SYSTEM_PROMPTS_FILE);
        if shipped.is_file() {
            let text = std::fs::read_to_string(&shipped)?;
            Self::parse(&text)
        } else {
            Ok(Self::builtin().clone())
        }
    }

    pub fn get(&self, mode: UpsampleMode) -> &str {
        match mode {
            UpsampleMode::Text => &self.t2i,
            UpsampleMode::Editing => &self.i2i,
            UpsampleMode::TextWithBoxes => &self.t2i_bbox,
            UpsampleMode::EditingWithBoxes => &self.i2i_bbox,
        }
    }
}

/// `pup.I2I_SCAFFOLD[locked].format(instruction=prompt)`.
pub fn i2i_scaffold(instruction: &str, locked: bool) -> String {
    let template = if locked {
        I2I_SCAFFOLD_LOCKED
    } else {
        I2I_SCAFFOLD_PLAIN
    };
    template.replace("{instruction}", instruction)
}

// ---------------------------------------------------------------------------------------------
// Python string helpers
// ---------------------------------------------------------------------------------------------

/// Python `str.isspace` for one character: Unicode `White_Space` plus the C0 separators
/// U+001C..U+001F, which Python counts and Rust does not.
pub fn py_is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python `str.strip()`.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_is_space)
}

/// Python `str.rstrip()`.
pub fn py_rstrip(s: &str) -> &str {
    s.trim_end_matches(py_is_space)
}

/// `bbox.ID.fullmatch`: `[A-Za-z][A-Za-z0-9_]*`.
pub fn is_row_id(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// ---------------------------------------------------------------------------------------------
// Boxes (`bbox.py`)
// ---------------------------------------------------------------------------------------------

/// One user-supplied box in the README's `boxes.json` shape, coordinates in 0–1 with the
/// top-left origin: `{"id", "desc", "bbox": [y_min, x_min, y_max, x_max]}` for a text-to-image
/// layout, or `{"id", "desc", "from": "ref_image_N" | "target", "src_bbox", "tgt_bbox"}` for
/// an edit (where it is in a reference, where it goes in the output, or both).
///
/// Absent, `null` and empty values are the same thing (Python truthiness in `bbox.to_canvas`):
/// an empty `id` becomes `obj_<i+1>`, an empty `desc` the empty string, an empty box list
/// "no box".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoxSpec {
    pub id: Option<String>,
    pub desc: Option<String>,
    pub bbox: Option<Vec<f64>>,
    pub from: Option<String>,
    pub src_bbox: Option<Vec<f64>>,
    pub tgt_bbox: Option<Vec<f64>>,
}

impl BoxSpec {
    /// Parse a `boxes.json` list.
    pub fn parse_list(json: &str) -> Result<Vec<BoxSpec>> {
        let value = loads(json)
            .map_err(|e| Error::Msg(format!("flux3 upsampler boxes: not a JSON list ({e})")))?;
        let items = value
            .as_array()
            .ok_or_else(|| Error::Msg("flux3 upsampler boxes: not a JSON list".into()))?;
        items
            .iter()
            .enumerate()
            .map(|(i, item)| Self::from_json(i, item))
            .collect()
    }

    /// One list element (`box` in `bbox.to_canvas`).
    pub fn from_json(index: usize, value: &JsonValue) -> Result<BoxSpec> {
        if !value.is_object() {
            return Err(UpsampleError::InvalidBox {
                index,
                reason: "a box is a JSON object".into(),
            }
            .into());
        }
        let text = |key: &str| -> Option<String> {
            value
                .get_non_null(key)
                .filter(|v| v.is_truthy())
                .map(JsonValue::py_str)
        };
        let coords = |key: &str| -> Result<Option<Vec<f64>>> {
            match value.get_non_null(key) {
                None => Ok(None),
                Some(v) if !v.is_truthy() => Ok(None),
                Some(JsonValue::Array(items)) => items
                    .iter()
                    .map(|v| {
                        v.as_number().ok_or_else(|| {
                            Error::from(UpsampleError::InvalidBox {
                                index,
                                reason: format!("{key} carries a non-numeric coordinate"),
                            })
                        })
                    })
                    .collect::<Result<Vec<f64>>>()
                    .map(Some),
                Some(_) => Err(UpsampleError::InvalidBox {
                    index,
                    reason: format!("{key} is not a list"),
                }
                .into()),
            }
        };
        Ok(BoxSpec {
            id: text("id"),
            desc: text("desc"),
            bbox: coords("bbox")?,
            from: text("from"),
            src_bbox: coords("src_bbox")?,
            tgt_bbox: coords("tgt_bbox")?,
        })
    }
}

/// `bbox.to_canvas`: 0–1 boxes → the upsampler's rows on the 0–1000 canvas (Python's
/// round-half-even `round`), as ordered JSON objects with upstream's key order.
pub fn to_canvas(boxes: &[BoxSpec]) -> Result<Vec<JsonValue>> {
    let grid = |index: usize, coords: Option<&Vec<f64>>| -> Result<JsonValue> {
        let Some(coords) = coords.filter(|c| !c.is_empty()) else {
            return Ok(JsonValue::Null);
        };
        if coords.len() != 4 {
            return Err(UpsampleError::InvalidBox {
                index,
                reason: format!("a box is [y_min, x_min, y_max, x_max], got {coords:?}"),
            }
            .into());
        }
        Ok(JsonValue::Array(
            coords
                .iter()
                .map(|v| {
                    JsonValue::Int(
                        (v.clamp(0.0, 1.0) * CANVAS_SCALE as f64).round_ties_even() as i64
                    )
                })
                .collect(),
        ))
    };
    let mut rows = Vec::with_capacity(boxes.len());
    for (i, b) in boxes.iter().enumerate() {
        let row_id = b.id.clone().unwrap_or_else(|| format!("obj_{}", i + 1));
        let desc = b.desc.clone().unwrap_or_default();
        if b.bbox.as_ref().is_some_and(|c| !c.is_empty()) {
            rows.push(JsonValue::Object(vec![
                ("id".into(), JsonValue::Str(row_id)),
                ("bbox".into(), grid(i, b.bbox.as_ref())?),
                ("desc".into(), JsonValue::Str(desc)),
            ]));
        } else if b.src_bbox.as_ref().is_some_and(|c| !c.is_empty())
            || b.tgt_bbox.as_ref().is_some_and(|c| !c.is_empty())
        {
            rows.push(JsonValue::Object(vec![
                ("id".into(), JsonValue::Str(row_id)),
                (
                    "from".into(),
                    JsonValue::Str(b.from.clone().unwrap_or_default()),
                ),
                ("src_bbox".into(), grid(i, b.src_bbox.as_ref())?),
                ("tgt_bbox".into(), grid(i, b.tgt_bbox.as_ref())?),
                ("desc".into(), JsonValue::Str(desc)),
            ]));
        } else {
            return Err(UpsampleError::InvalidBox {
                index: i,
                reason: format!("box {i} has neither bbox nor src_bbox/tgt_bbox"),
            }
            .into());
        }
    }
    Ok(rows)
}

/// `bbox.split`: `(caption, rows)` — the last `[` from which the rest parses as a JSON list.
/// Not a regex: descs may contain brackets and escaped quotes.
pub fn split(text: &str) -> (&str, Option<Vec<JsonValue>>) {
    for (i, c) in text.char_indices().rev() {
        if c != '[' {
            continue;
        }
        if let Ok(JsonValue::Array(rows)) = loads(&text[i..]) {
            return (py_rstrip(&text[..i]), Some(rows));
        }
    }
    (py_strip(text), None)
}

/// `bbox.valid_box`: a well-formed 0–1000 row box, returned with its original numbers; `None`
/// otherwise (not a 4-list, a bool or non-number, a degenerate or out-of-canvas box, or a box
/// whose coordinates all sit in 0–1 — those are the input grid, not the canvas).
pub fn valid_box(value: Option<&JsonValue>) -> Option<Vec<JsonValue>> {
    let items = value?.as_array()?;
    if items.len() != 4 {
        return None;
    }
    let mut nums = [0f64; 4];
    for (slot, v) in nums.iter_mut().zip(items) {
        let n = v.as_number()?;
        if !n.is_finite() {
            return None;
        }
        *slot = n;
    }
    let [y1, x1, y2, x2] = nums;
    let scale = CANVAS_SCALE as f64;
    if y1 < 0.0 || x1 < 0.0 || y2 <= y1 || x2 <= x1 || y2 > scale || x2 > scale {
        return None;
    }
    if nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max) <= 1.0 {
        return None;
    }
    Some(items.to_vec())
}

/// `bbox.detect`: `(prompt, rows)` when the prompt already carries boxes on the 0–1000 canvas,
/// else `None`. Two forms: `<caption> [rows]` (the DiT prompt format) and the upsampler's own
/// `ORIGINAL PROMPT: … LOCKED BOUNDING-BOX ANNOTATIONS …:\n[rows]` input.
pub fn detect(prompt: &str) -> Option<(String, Vec<JsonValue>)> {
    let (text, rows) = split(prompt);
    let rows = rows?;
    if rows.is_empty() {
        return None;
    }
    let all_rows = rows.iter().all(|r| {
        r.is_object()
            && r.get("id")
                .and_then(JsonValue::as_str)
                .is_some_and(is_row_id)
            && ["bbox", "src_bbox", "tgt_bbox"]
                .iter()
                .any(|k| valid_box(r.get(k)).is_some())
    });
    if !all_rows {
        return None;
    }
    Some((
        locked_original_prompt(text).unwrap_or(text).to_string(),
        rows,
    ))
}

/// The lazy group of `re.fullmatch(r"ORIGINAL PROMPT:\n(.*?)\n\nLOCKED BOUNDING-BOX ANNOTATIONS
/// .*", text, re.S)`: the text between the header and the first annotations line.
fn locked_original_prompt(text: &str) -> Option<&str> {
    let body = text.strip_prefix("ORIGINAL PROMPT:\n")?;
    let at = body.find("\n\nLOCKED BOUNDING-BOX ANNOTATIONS ")?;
    Some(&body[..at])
}

/// `bbox.is_finished`: an upsampler answer, not a request — its box table is cited in full as
/// `<id>`s in the caption.
pub fn is_finished(prompt: &str) -> bool {
    detect(prompt).is_some_and(|(caption, rows)| {
        rows.iter().all(|row| {
            row.get("id")
                .and_then(JsonValue::as_str)
                .is_some_and(|id| caption.contains(&format!("<{id}>")))
        })
    })
}

/// `bbox.upsampling_input`: the user text of the `_bbox` modes.
pub fn upsampling_input(prompt: &str, rows: &[JsonValue]) -> String {
    format!(
        "ORIGINAL PROMPT:\n{}\n\nLOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on \
         a 0-1000 canvas, with (0, 0) at the top-left). This is an authoritative JSON list of \
         dictionaries; preserve every supplied identity, provenance value, null, and coordinate \
         exactly:\n{}",
        py_strip(prompt),
        dumps(&JsonValue::Array(rows.to_vec()))
    )
}

/// `bbox.plan`: `(mode, user text, locked rows)` for the upsampler. Passed boxes win; otherwise
/// boxes found in the prompt switch it to the `_bbox` mode by themselves.
pub fn plan_text(
    prompt: &str,
    boxes: Option<&[BoxSpec]>,
    n_images: usize,
) -> Result<(UpsampleMode, String, Option<Vec<JsonValue>>)> {
    let (prompt, locked) = match boxes.filter(|b| !b.is_empty()) {
        Some(boxes) => (prompt.to_string(), to_canvas(boxes)?),
        None => match detect(prompt) {
            Some((found, locked)) => (found, locked),
            None => {
                return Ok((
                    UpsampleMode::infer(n_images, false),
                    prompt.to_string(),
                    None,
                ))
            }
        },
    };
    Ok((
        UpsampleMode::infer(n_images, true),
        upsampling_input(&prompt, &locked),
        Some(locked),
    ))
}

/// `generate.py`'s caption when the upsampler does not run: the prompt as typed, with a passed
/// box list appended on the 0–1000 canvas.
pub fn bypass_caption(prompt: &str, boxes: Option<&[BoxSpec]>) -> Result<String> {
    match boxes.filter(|b| !b.is_empty()) {
        Some(boxes) => Ok(format!(
            "{} {}",
            py_rstrip(prompt),
            dumps(&JsonValue::Array(to_canvas(boxes)?))
        )),
        None => Ok(prompt.to_string()),
    }
}

/// `bbox.valid_added`: a row the upsampler added, kept only if its id is cited as `<id>` and it
/// is well formed (normalized to upstream's key order).
pub fn valid_added(row: &JsonValue, caption: &str) -> Option<JsonValue> {
    if !row.is_object() {
        return None;
    }
    let row_id = row.get("id")?.as_str()?;
    let desc = row.get("desc")?.as_str()?;
    if !is_row_id(row_id) || !caption.contains(&format!("<{row_id}>")) || py_strip(desc).is_empty()
    {
        return None;
    }
    if let Some(bbox) = valid_box(row.get("bbox")) {
        return Some(JsonValue::Object(vec![
            ("id".into(), JsonValue::Str(row_id.into())),
            ("bbox".into(), JsonValue::Array(bbox)),
            ("desc".into(), JsonValue::Str(py_strip(desc).into())),
        ]));
    }
    let origin = row.get_non_null("from");
    let src = row.get_non_null("src_bbox");
    let tgt = row.get_non_null("tgt_bbox");
    let src_box = src.and_then(|v| valid_box(Some(v)));
    let tgt_box = tgt.and_then(|v| valid_box(Some(v)));
    let origin = origin
        .and_then(JsonValue::as_str)
        .filter(|o| is_row_origin(o))?;
    if (src.is_some() && src_box.is_none())
        || (tgt.is_some() && tgt_box.is_none())
        || (src_box.is_none() && tgt_box.is_none())
        || ((origin == "target") == src_box.is_some())
    {
        return None;
    }
    let boxed = |b: Option<Vec<JsonValue>>| b.map(JsonValue::Array).unwrap_or(JsonValue::Null);
    Some(JsonValue::Object(vec![
        ("id".into(), JsonValue::Str(row_id.into())),
        ("from".into(), JsonValue::Str(origin.into())),
        ("src_bbox".into(), boxed(src_box)),
        ("tgt_bbox".into(), boxed(tgt_box)),
        ("desc".into(), JsonValue::Str(py_strip(desc).into())),
    ]))
}

/// `bbox.iou` over two canvas boxes.
pub fn iou(a: &[JsonValue], b: &[JsonValue]) -> Result<f64> {
    let num = |v: &[JsonValue], i: usize| -> Result<f64> {
        v.get(i).and_then(JsonValue::as_number).ok_or_else(|| {
            UpsampleError::Reconcile(format!(
                "a locked box is not a numeric 4-list: {}",
                dumps(&JsonValue::Array(v.to_vec()))
            ))
            .into()
        })
    };
    let (a0, a1, a2, a3) = (num(a, 0)?, num(a, 1)?, num(a, 2)?, num(a, 3)?);
    let (b0, b1, b2, b3) = (num(b, 0)?, num(b, 1)?, num(b, 2)?, num(b, 3)?);
    let inter = (a2.min(b2) - a0.max(b0)).max(0.0) * (a3.min(b3) - a1.max(b1)).max(0.0);
    let union = (a2 - a0) * (a3 - a1) + (b2 - b0) * (b3 - b1) - inter;
    Ok(if union > 0.0 { inter / union } else { 0.0 })
}

/// `bbox.reconcile`: the upsampler's answer with the user's boxes re-imposed.
///
/// 1. A row re-planning a locked box under its own id (IoU ≥ 0.9, e.g. `balloon_1` for
///    `obj_1`) takes the locked id, in the rows and in the caption's `<id>` citations.
/// 2. Locked rows are kept verbatim, adopting the upsampler's desc; its own rows follow if
///    valid.
pub fn reconcile(text: &str, locked: &[JsonValue]) -> Result<String> {
    let (caption, rows) = split(text);
    let mut caption = caption.to_string();
    let mut rows = rows.unwrap_or_default();
    let lock_id = |lock: &JsonValue| -> Result<String> {
        lock.get("id")
            .and_then(JsonValue::as_str)
            .map(str::to_string)
            .ok_or_else(|| UpsampleError::Reconcile("a locked row has no string id".into()).into())
    };
    let locked_ids: Vec<String> = locked.iter().map(lock_id).collect::<Result<_>>()?;
    let row_ids: Vec<JsonValue> = rows
        .iter()
        .filter(|r| r.is_object())
        .map(|r| r.get("id").cloned().unwrap_or(JsonValue::Null))
        .collect();
    let is_locked_id = |v: Option<&JsonValue>| {
        v.and_then(JsonValue::as_str)
            .is_some_and(|s| locked_ids.iter().any(|l| l == s))
    };
    // Python truthiness chain: the first of bbox / tgt_bbox / src_bbox that is truthy.
    let first_box = |row: &JsonValue| -> Option<JsonValue> {
        ["bbox", "tgt_bbox", "src_bbox"]
            .iter()
            .filter_map(|k| row.get(k))
            .find(|v| v.is_truthy())
            .cloned()
    };
    let mut renamed: Vec<(JsonValue, String)> = Vec::new();
    for (lock, lock_id) in locked.iter().zip(&locked_ids) {
        let Some(lock_box) = first_box(lock) else {
            continue;
        };
        if row_ids.iter().any(|r| r.as_str() == Some(lock_id)) {
            continue;
        }
        let lock_box = lock_box
            .as_array()
            .map(<[JsonValue]>::to_vec)
            .ok_or_else(|| {
                Error::from(UpsampleError::Reconcile(format!(
                    "locked box of {lock_id:?} is not a list"
                )))
            })?;
        for row in &rows {
            if !row.is_object() {
                continue;
            }
            let rid = row.get("id");
            if renamed.iter().any(|(old, _)| Some(old) == rid) || is_locked_id(rid) {
                continue;
            }
            let Some(candidate) = first_box(row) else {
                continue;
            };
            let Some(bx) = valid_box(Some(&candidate)) else {
                continue;
            };
            if iou(&bx, &lock_box)? >= 0.9 {
                let rid = rid.cloned().ok_or_else(|| {
                    Error::from(UpsampleError::Reconcile(
                        "an upsampler row without an id re-plans a locked box".into(),
                    ))
                })?;
                renamed.push((rid, lock_id.clone()));
                break;
            }
        }
    }
    for row in rows.iter_mut() {
        if !row.is_object() {
            continue;
        }
        if let Some((_, new)) = renamed.iter().find(|(old, _)| Some(old) == row.get("id")) {
            row.set("id", JsonValue::Str(new.clone()));
        }
    }
    for (old, new) in &renamed {
        caption = caption.replace(&format!("<{}>", old.py_str()), &format!("<{new}>"));
    }
    // `by_id`: the last row per string id wins (dict comprehension).
    let desc_of = |id: &str| -> Option<&JsonValue> {
        rows.iter()
            .rev()
            .find(|r| r.is_object() && r.get("id").and_then(JsonValue::as_str) == Some(id))
            .and_then(|r| r.get("desc"))
    };
    let mut out: Vec<JsonValue> = Vec::with_capacity(locked.len());
    for (lock, lock_id) in locked.iter().zip(&locked_ids) {
        let desc = match desc_of(lock_id).and_then(JsonValue::as_str) {
            Some(d) if !py_strip(d).is_empty() => JsonValue::Str(py_strip(d).into()),
            _ => lock
                .get("desc")
                .cloned()
                .unwrap_or(JsonValue::Str(String::new())),
        };
        let mut row = lock.clone();
        row.set("desc", desc);
        out.push(row);
    }
    for row in &rows {
        if let Some(added) = valid_added(row, &caption) {
            let id = added
                .get("id")
                .and_then(JsonValue::as_str)
                .unwrap_or_default();
            if !out
                .iter()
                .any(|r| r.get("id").and_then(JsonValue::as_str) == Some(id))
            {
                out.push(added);
            }
        }
    }
    Ok(format!("{caption} {}", dumps(&JsonValue::Array(out))))
}

// ---------------------------------------------------------------------------------------------
// Reference images (`pup.reference_size`, `pup.policy_reference`)
// ---------------------------------------------------------------------------------------------

/// `pup.reference_size`: the size budget of one reference when `n_refs` are attached — sides
/// rounded to multiples of 32, then scaled under [`max_pixels_by_refs`] (two floor passes, as
/// upstream). A reference beyond [`MAX_ASPECT_RATIO`] is the `ValueError` that skips upsampling.
pub fn reference_size(width: u32, height: u32, n_refs: usize) -> Result<(u32, u32)> {
    if width == 0 || height == 0 {
        return Err(Error::Msg(format!(
            "flux3 upsampler: a reference image has a zero side ({width}x{height})"
        )));
    }
    let (w, h) = (f64::from(width), f64::from(height));
    if w.max(h) / w.min(h) > MAX_ASPECT_RATIO {
        return Err(UpsampleError::ReferenceAspectRatio { width, height }.into());
    }
    let m = f64::from(REFERENCE_MULTIPLE);
    let floor = |v: f64| m.max((v / m).floor() * m);
    let cap = max_pixels_by_refs(n_refs).ok_or(UpsampleError::TooManyReferences(n_refs))? as f64;
    let mut w_ = m.max((w / m).round_ties_even() * m);
    let mut h_ = m.max((h / m).round_ties_even() * m);
    if w_ * h_ > cap {
        let beta = (w * h / cap).sqrt();
        w_ = floor(w / beta);
        h_ = floor(h / beta);
    }
    if w_ * h_ > cap {
        if w_ == m {
            h_ = floor(cap / w_);
        } else {
            w_ = floor(cap / h_);
        }
    }
    Ok((w_ as u32, h_ as u32))
}

/// The Qwen3-VL processor's `smart_resize` for one image under the package's
/// `preprocessor_config.json`, and the merged token count it yields (`(h/32)·(w/32)`). Used
/// only to budget the prompt window; the serving engine's processor does the real resize.
pub fn vision_tokens(width: u32, height: u32) -> Result<usize> {
    let factor = f64::from(VISION_TOKEN_SIDE);
    let (w, h) = (f64::from(width), f64::from(height));
    if width == 0 || height == 0 || w.max(h) / w.min(h) > MAX_ASPECT_RATIO {
        return Err(Error::Msg(format!(
            "flux3 upsampler: reference {width}x{height} is outside the vision processor's \
             aspect limit"
        )));
    }
    let mut h_bar = factor.max((h / factor).round_ties_even() * factor);
    let mut w_bar = factor.max((w / factor).round_ties_even() * factor);
    if h_bar * w_bar > VISION_MAX_PIXELS as f64 {
        let beta = (h * w / VISION_MAX_PIXELS as f64).sqrt();
        h_bar = factor.max((h / beta / factor).floor() * factor);
        w_bar = factor.max((w / beta / factor).floor() * factor);
    } else if h_bar * w_bar < VISION_MIN_PIXELS as f64 {
        let beta = (VISION_MIN_PIXELS as f64 / (h * w)).sqrt();
        h_bar = (h * beta / factor).ceil() * factor;
        w_bar = (w * beta / factor).ceil() * factor;
    }
    Ok(((h_bar / factor) as usize) * ((w_bar / factor) as usize))
}

/// A reference image as the caller holds it. The upsampler's policy ([`policy_reference`])
/// starts from Pillow's `RGB` conversion of the opened file: an already-decoded buffer skips
/// the decode, an explicit alpha carrier is composited over white, and a path applies EXIF plus
/// the frozen ORIGINAL-mode policy. PNG RGB/L `tRNS` does not acquire alpha-composite semantics;
/// PNG16 gray saturates while color PNG16 uses its high bytes (as Pillow opens those modes).
#[derive(Clone, Debug, PartialEq)]
pub enum ReferenceImage {
    /// RGB8, row-major, `width * height * 3` bytes.
    Rgb8 {
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    },
    /// RGBA8 straight alpha, `width * height * 4` bytes; composited over white.
    Rgba8 {
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    },
    /// RGB `f32` in `[0, 1]`, row-major HWC, `width * height * 3` values — the decoded array
    /// the image model's own reference path holds. Quantized to RGB8 by rounding (upstream
    /// has no float path: Pillow images are 8-bit).
    F32Rgb {
        width: u32,
        height: u32,
        pixels: Vec<f32>,
    },
    /// Decoded from a file on use.
    Path(PathBuf),
}

impl ReferenceImage {
    /// The Pillow `RGB` image: `(width, height, rgb8)`.
    pub fn to_rgb8(&self) -> Result<(u32, u32, Vec<u8>)> {
        let check = |width: u32, height: u32, len: usize, channels: usize| -> Result<()> {
            let expected = (width as usize)
                .checked_mul(height as usize)
                .and_then(|n| n.checked_mul(channels));
            if expected != Some(len) {
                return Err(Error::Msg(format!(
                    "flux3 upsampler: reference buffer of {len} values is not {width}x{height}x{channels}"
                )));
            }
            Ok(())
        };
        match self {
            ReferenceImage::Rgb8 {
                width,
                height,
                pixels,
            } => {
                check(*width, *height, pixels.len(), 3)?;
                Ok((*width, *height, pixels.clone()))
            }
            ReferenceImage::Rgba8 {
                width,
                height,
                pixels,
            } => {
                check(*width, *height, pixels.len(), 4)?;
                Ok((*width, *height, composite_over_white(pixels)))
            }
            ReferenceImage::F32Rgb {
                width,
                height,
                pixels,
            } => {
                check(*width, *height, pixels.len(), 3)?;
                Ok((
                    *width,
                    *height,
                    pixels
                        .iter()
                        .map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
                        .collect(),
                ))
            }
            ReferenceImage::Path(path) => decode_reference_file(path),
        }
    }
}

/// Pillow `Image.alpha_composite(white, rgba).convert("RGB")` — `ImagingAlphaComposite` with an
/// opaque white destination, in its fixed-point arithmetic (`PRECISION_BITS = 7`,
/// `SHIFTFORDIV255`).
pub fn composite_over_white(rgba: &[u8]) -> Vec<u8> {
    const PRECISION_BITS: u32 = 7;
    fn shift_for_div_255(a: u32) -> u32 {
        ((a >> 8) + a) >> 8
    }
    let mut out = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        let src_a = u32::from(px[3]);
        if src_a == 255 {
            out.extend_from_slice(&px[..3]);
            continue;
        }
        if src_a == 0 {
            out.extend_from_slice(&[255, 255, 255]);
            continue;
        }
        // dst = (255, 255, 255, 255): blend = 255 * (255 - src_a); outa255 = src_a*255 + blend.
        let blend = 255 * (255 - src_a);
        let outa255 = src_a * 255 + blend;
        let coef1 = src_a * 255 * 255 * (1 << PRECISION_BITS) / outa255;
        let coef2 = 255 * (1 << PRECISION_BITS) - coef1;
        for &channel in &px[..3] {
            let tmp = u32::from(channel) * coef1 + 255 * coef2;
            out.push((shift_for_div_255(tmp + (0x80 << PRECISION_BITS)) >> PRECISION_BITS) as u8);
        }
    }
    out
}

fn decode_reference_file(path: &Path) -> Result<(u32, u32, Vec<u8>)> {
    use super::image_input::{decode_reference_file as decode_input, DecodedReference};
    // The shared decoder returns pixel data, NEVER another Path, so this delegation cannot
    // recurse through ReferenceImage::Path. Preserve the original Pillow mode policy first;
    // PNG EXPAND alpha is not evidence that upstream would composite it (RGB/L tRNS do not).
    match decode_input(path)? {
        DecodedReference::Rgb(image) => Ok((image.width, image.height, image.pixels)),
        DecodedReference::Rgba(image) => Ok((
            image.width,
            image.height,
            composite_over_white(&image.pixels),
        )),
    }
}

/// `pup.policy_reference`: RGB over white, Pillow `LANCZOS` to the size budget, JPEG q92 —
/// what the VLM is shown for one of `n_refs` references.
pub fn policy_reference(image: &ReferenceImage, n_refs: usize) -> Result<ImageRef> {
    let (width, height, rgb) = image.to_rgb8()?;
    let (tw, th) = reference_size(width, height, n_refs)?;
    let resized: Vec<u8> = if (tw, th) == (width, height) {
        rgb
    } else {
        gen_core::imageops::resize_lanczos_u8(
            &rgb,
            height as usize,
            width as usize,
            th as usize,
            tw as usize,
        )?
        .into_iter()
        .map(|v| v.round().clamp(0.0, 255.0) as u8)
        .collect()
    };
    let decoded = jpeg_round_trip(&resized, tw, th)?;
    ImageRef::new(tw, th, decoded).map_err(Error::Msg)
}

/// `image.save(JPEG, quality=92)` → `Image.open(...).convert("RGB")`: libjpeg's pixel effect,
/// reproduced in [`jpeg`] (the sides are multiples of 32 by [`reference_size`]).
fn jpeg_round_trip(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    jpeg::round_trip(rgb, width, height, JPEG_QUALITY)
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// The typed refusals and failures of the upsampling stage.
#[derive(Clone, Debug, PartialEq)]
pub enum UpsampleError {
    /// `enhance_prompt` was requested on a generator loaded without the
    /// [`UPSAMPLER_COMPONENT`].
    MissingUpsampler,
    /// The component directory is not a complete upsampler package.
    IncompletePackage { dir: PathBuf, missing: String },
    /// The component directory is some other model.
    NotTheUpsampler { dir: PathBuf, reason: String },
    /// More than [`MAX_REFERENCES`] references.
    TooManyReferences(usize),
    /// The explicit mode disagrees with the inputs (references and boxes).
    ModeMismatch {
        requested: UpsampleMode,
        inferred: UpsampleMode,
        references: usize,
        boxes: bool,
    },
    /// A malformed box (`bbox.to_canvas`'s `ValueError`).
    InvalidBox { index: usize, reason: String },
    /// A reference beyond [`MAX_ASPECT_RATIO`] — upstream's "upsampling skipped" fallback.
    ReferenceAspectRatio { width: u32, height: u32 },
    /// The rendered prompt leaves no room in the [`MAX_MODEL_LEN`] window.
    PromptTooLong { tokens: usize, max: usize },
    /// The model produced nothing (`RuntimeError("empty generation")`).
    EmptyGeneration,
    /// The complete post-reconcile output exceeds the renderer/persistence contract's character
    /// admission bound. This is a pre-image error, not a request to truncate authoritative rows.
    CaptionTooLong {
        field: &'static str,
        chars: usize,
        max: usize,
    },
    /// `bbox.reconcile` met an answer it cannot re-impose the locks on.
    Reconcile(String),
}

impl UpsampleError {
    /// The stable reason code of the fallback class (upstream catches the `ValueError` and sends
    /// the prompt as typed); `None` for the refusals.
    pub fn fallback_reason(&self) -> Option<&'static str> {
        match self {
            UpsampleError::ReferenceAspectRatio { .. } => Some("reference_aspect_ratio"),
            _ => None,
        }
    }
}

impl fmt::Display for UpsampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UpsampleError::MissingUpsampler => write!(
                f,
                "enhance_prompt needs the FLUX 3 prompt upsampler, which this generator was \
                 loaded without (provision {UPSAMPLER_REPO} @ {UPSAMPLER_REVISION} as the \
                 '{UPSAMPLER_COMPONENT}' component, or send the prompt as typed)"
            ),
            UpsampleError::IncompletePackage { dir, missing } => write!(
                f,
                "the '{UPSAMPLER_COMPONENT}' component at {} is missing {missing} (the package is \
                 {UPSAMPLER_REPO} @ {UPSAMPLER_REVISION})",
                dir.display()
            ),
            UpsampleError::NotTheUpsampler { dir, reason } => write!(
                f,
                "the '{UPSAMPLER_COMPONENT}' component at {} is not the FLUX 3 prompt upsampler: \
                 {reason}",
                dir.display()
            ),
            UpsampleError::TooManyReferences(n) => {
                write!(
                    f,
                    "the upsampler takes at most {MAX_REFERENCES} references (got {n})"
                )
            }
            UpsampleError::ModeMismatch {
                requested,
                inferred,
                references,
                boxes,
            } => write!(
                f,
                "upsample mode {requested} does not match the inputs ({references} reference(s), \
                 boxes: {boxes}) — upstream would run {inferred}"
            ),
            UpsampleError::InvalidBox { index, reason } => {
                write!(f, "box {index}: {reason}")
            }
            UpsampleError::ReferenceAspectRatio { width, height } => write!(
                f,
                "aspect ratio of {width}x{height} exceeds {MAX_ASPECT_RATIO}:1"
            ),
            UpsampleError::PromptTooLong { tokens, max } => write!(
                f,
                "the rendered upsampler prompt is {tokens} tokens, leaving no room to generate \
                 in the {max}-token window"
            ),
            UpsampleError::EmptyGeneration => f.write_str("empty generation"),
            UpsampleError::CaptionTooLong { field, chars, max } => write!(
                f, "{field} contains {chars} Unicode characters, exceeding the {max}-character limit; reduce the caption token budget or locked annotations (no rows were truncated)"
            ),
            UpsampleError::Reconcile(reason) => write!(f, "reconcile: {reason}"),
        }
    }
}

impl std::error::Error for UpsampleError {}

impl From<UpsampleError> for Error {
    fn from(e: UpsampleError) -> Self {
        let text = format!("flux3 prompt upsampler: {e}");
        match e {
            UpsampleError::MissingUpsampler
            | UpsampleError::NotTheUpsampler { .. }
            | UpsampleError::TooManyReferences(_)
            | UpsampleError::ModeMismatch { .. }
            | UpsampleError::InvalidBox { .. }
            | UpsampleError::ReferenceAspectRatio { .. }
            | UpsampleError::PromptTooLong { .. }
            | UpsampleError::CaptionTooLong { .. } => Error::Unsupported(text),
            UpsampleError::IncompletePackage { .. }
            | UpsampleError::EmptyGeneration
            | UpsampleError::Reconcile(_) => Error::Msg(text),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The package (prompts, tokenizer, chat template)
// ---------------------------------------------------------------------------------------------

/// The host-side view of the upsampler package: its system prompts, tokenizer and chat
/// template — enough to render a plan to the exact prompt text and to budget the window. The
/// weights are the serving engine's.
pub struct UpsamplePackage {
    dir: PathBuf,
    prompts: SystemPrompts,
    tokenizer: core_llm::Tokenizer,
    template: JinjaChatTemplate,
}

impl fmt::Debug for UpsamplePackage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpsamplePackage")
            .field("dir", &self.dir)
            .finish_non_exhaustive()
    }
}

impl UpsamplePackage {
    /// The upsampler directory a load spec names, if any.
    pub fn component_dir(spec: &LoadSpec) -> Result<Option<PathBuf>> {
        match spec.components.get(UPSAMPLER_COMPONENT) {
            None => Ok(None),
            Some(WeightsSource::Dir(dir)) => Ok(Some(dir.clone())),
            Some(WeightsSource::File(file)) => Err(Error::Msg(format!(
                "flux3 prompt upsampler: the '{UPSAMPLER_COMPONENT}' component must be the \
                 package directory, not the single file {}",
                file.display()
            ))),
        }
    }

    /// Existence and identity check: every [`UPSAMPLER_REQUIRED_FILES`] entry present and
    /// `config.json` naming the [`UPSAMPLER_MODEL_TYPE`] with a vision tower.
    pub fn check(dir: &Path) -> Result<()> {
        if !dir.is_dir() {
            return Err(UpsampleError::IncompletePackage {
                dir: dir.to_path_buf(),
                missing: "the directory itself".into(),
            }
            .into());
        }
        let missing: Vec<&str> = UPSAMPLER_REQUIRED_FILES
            .iter()
            .copied()
            .filter(|name| !dir.join(name).is_file())
            .collect();
        if !missing.is_empty() {
            return Err(UpsampleError::IncompletePackage {
                dir: dir.to_path_buf(),
                missing: missing.join(", "),
            }
            .into());
        }
        let config = std::fs::read_to_string(dir.join("config.json"))?;
        let config: serde_json::Value = serde_json::from_str(&config).map_err(|e| {
            Error::Msg(format!(
                "flux3 prompt upsampler: {}/config.json: {e}",
                dir.display()
            ))
        })?;
        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if model_type != UPSAMPLER_MODEL_TYPE {
            return Err(UpsampleError::NotTheUpsampler {
                dir: dir.to_path_buf(),
                reason: format!(
                    "config.json model_type is {model_type:?}, not {UPSAMPLER_MODEL_TYPE:?}"
                ),
            }
            .into());
        }
        if config.get("vision_config").is_none() {
            return Err(UpsampleError::NotTheUpsampler {
                dir: dir.to_path_buf(),
                reason: "config.json has no vision_config (the upsampler reads reference images)"
                    .into(),
            }
            .into());
        }
        Ok(())
    }

    /// Open a checked package directory.
    pub fn open(dir: &Path) -> Result<Self> {
        Self::check(dir)?;
        let prompts = SystemPrompts::from_package(dir)?;
        let tokenizer = core_llm::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::Msg(format!("flux3 prompt upsampler: {e}")))?;
        let source = std::fs::read_to_string(dir.join("chat_template.jinja"))?;
        let eos = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| {
                v.get("eos_token")
                    .and_then(|e| e.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_default();
        let template = JinjaChatTemplate::with_tokens(source, String::new(), eos);
        Ok(Self {
            dir: dir.to_path_buf(),
            prompts,
            tokenizer,
            template,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn prompts(&self) -> &SystemPrompts {
        &self.prompts
    }

    pub fn tokenizer(&self) -> &core_llm::Tokenizer {
        &self.tokenizer
    }

    /// The exact prompt text of a plan — `apply_chat_template(messages, tokenize=False,
    /// add_generation_prompt=True, enable_thinking=False)` with each reference rendered as
    /// [`IMAGE_PLACEHOLDER`] (what the serving engine substitutes before expanding it).
    pub fn render(&self, plan: &UpsamplePlan) -> Result<String> {
        let messages: Vec<Message> = plan
            .messages()
            .into_iter()
            .map(|m| Message {
                content: m
                    .content
                    .into_iter()
                    .map(|c| match c {
                        Content::Image(_) => Content::text(IMAGE_PLACEHOLDER),
                        other => other,
                    })
                    .collect(),
                ..m
            })
            .collect();
        self.template
            .render_with(
                &messages,
                &RenderOptions::generation().with_enable_thinking(Some(false)),
            )
            .map_err(|e| Error::Msg(format!("flux3 prompt upsampler: chat template: {e}")))
    }

    /// The prompt's token count as the engine will see it: the rendered text tokenized without
    /// special-token insertion (`tokenizer.encode(text, add_special_tokens=False)`), each
    /// single `<|image_pad|>` expanded to its reference's merged patch count.
    pub fn prompt_tokens(&self, plan: &UpsamplePlan) -> Result<usize> {
        let text = self.render(plan)?;
        let ids = self
            .tokenizer
            .encode(&text, false)
            .map_err(|e| Error::Msg(format!("flux3 prompt upsampler: tokenize: {e}")))?;
        let mut total = ids.len();
        for image in &plan.references {
            total += vision_tokens(image.width, image.height)?.saturating_sub(1);
        }
        Ok(total)
    }

    /// The decode budget of a plan: the request's `max_tokens` (default [`MAX_TOKENS`]) clipped
    /// to what the [`MAX_MODEL_LEN`] window leaves after the prompt; a prompt that leaves no
    /// room is [`UpsampleError::PromptTooLong`].
    pub fn budget(&self, plan: &UpsamplePlan, requested: Option<u32>) -> Result<u32> {
        let tokens = self.prompt_tokens(plan)?;
        let room = MAX_MODEL_LEN.saturating_sub(tokens);
        if room == 0 {
            return Err(UpsampleError::PromptTooLong {
                tokens,
                max: MAX_MODEL_LEN,
            }
            .into());
        }
        let want = requested.unwrap_or(MAX_TOKENS).max(1) as usize;
        Ok(want.min(room) as u32)
    }
}

// ---------------------------------------------------------------------------------------------
// The input contract and the stage
// ---------------------------------------------------------------------------------------------

/// What the generator hands the upsampling stage — the tensor-neutral contract the editing
/// (sc-26236) and layout (sc-26237) stories compose onto. `mode` is explicit (E3): it must agree
/// with the inputs, or the stage refuses rather than quietly running another mode.
#[derive(Clone, Debug, PartialEq)]
pub struct UpsamplerInput {
    /// The user's prompt as typed.
    pub prompt: String,
    /// Reference images in attachment order (empty for the text modes).
    pub references: Vec<ReferenceImage>,
    /// Locked boxes (the README `boxes.json` shape, 0–1 coordinates). `None` / empty: none
    /// passed — a table already in the prompt still locks (`bbox.detect`).
    pub boxes: Option<Vec<BoxSpec>>,
    /// The requested mode.
    pub mode: UpsampleMode,
    /// `--no-pup`: send the prompt as typed (boxes appended on the canvas).
    pub bypass: bool,
}

/// Per-request overrides from `GenerationRequest::enhance_*`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UpsampleOptions {
    /// Sampling temperature (`None` ⇒ upstream's greedy [`TEMPERATURE`]).
    pub temperature: Option<f32>,
    /// Decode budget (`None` ⇒ [`MAX_TOKENS`]); always clipped to the window.
    pub max_tokens: Option<u32>,
    /// RNG seed for a sampled (non-greedy) run.
    pub seed: Option<u64>,
}

impl UpsampleOptions {
    /// Native request transport: preserve absent values so frozen model defaults resolve at the
    /// producer, rather than substituting a legacy enhancer's default or silently clamping a cap.
    pub fn from_generation_request(request: &gen_core::GenerationRequest) -> Self {
        Self {
            temperature: request.enhance_temperature,
            max_tokens: request.enhance_max_tokens,
            seed: request.seed,
        }
    }
}

/// A planned run: the messages, the locked rows to re-impose and the sampling policy.
#[derive(Clone, Debug, PartialEq)]
pub struct UpsamplePlan {
    pub mode: UpsampleMode,
    /// The prompt as typed (the report's `original_prompt`).
    pub original_prompt: String,
    pub system_prompt: String,
    pub user_text: String,
    /// The locked canvas rows, when the mode is a `_bbox` one.
    pub locked: Option<Vec<JsonValue>>,
    /// The policy-preprocessed references, in order.
    pub references: Vec<ImageRef>,
    /// The decode budget after [`UpsamplePackage::budget`].
    pub max_new_tokens: u32,
}

impl UpsamplePlan {
    /// The chat turns: the mode's system prompt, then the user turn — the references (in
    /// order) followed by the scaffolded instruction for the editing modes, the user text alone
    /// for the text modes.
    pub fn messages(&self) -> Vec<Message> {
        let mut content: Vec<Content> = self
            .references
            .iter()
            .cloned()
            .map(Content::Image)
            .collect();
        content.push(Content::text(self.user_text.clone()));
        vec![
            Message::system(self.system_prompt.clone()),
            Message {
                role: Role::User,
                content,
                thinking: None,
                tool_calls: Vec::new(),
            },
        ]
    }

    /// `pup.SAMPLING` + the mode's presence penalty, with the request's temperature override.
    pub fn sampling(&self, options: &UpsampleOptions) -> Sampling {
        Sampling {
            temperature: options.temperature.unwrap_or(TEMPERATURE),
            top_p: TOP_P,
            top_k: TOP_K,
            presence_penalty: self.mode.presence_penalty(),
            presence_penalty_generated_only: true,
            repetition_penalty: 1.0,
            repetition_context: 0,
        }
    }

    /// The engine request: thinking off, the stop tokens, the budgeted decode length.
    pub fn request(
        &self,
        options: &UpsampleOptions,
        cancel: core_llm::CancelFlag,
    ) -> TextLlmRequest {
        TextLlmRequest {
            messages: self.messages(),
            sampling: self.sampling(options),
            max_new_tokens: self.max_new_tokens,
            seed: options.seed,
            thinking: ThinkingMode::Disabled,
            stop: STOP_TOKENS.iter().map(|s| s.to_string()).collect(),
            cancel,
            ..TextLlmRequest::default()
        }
    }

    /// The answer as the image model receives it: stripped, the locks re-imposed; an empty
    /// answer is [`UpsampleError::EmptyGeneration`].
    pub fn finish(&self, raw: &str) -> Result<UpsampleOutcome> {
        let caption = py_strip(raw);
        if caption.is_empty() {
            return Err(UpsampleError::EmptyGeneration.into());
        }
        let caption = match &self.locked {
            Some(locked) => reconcile(caption, locked)?,
            None => caption.to_string(),
        };
        // This complete caption (including re-imposed rows) must be admissible BEFORE the image
        // encoder/render starts, not rejected later when its already-rendered asset is persisted.
        validate_caption_size(&caption, "effective caption")?;
        let locked_annotations = self
            .locked
            .as_ref()
            .map(|rows| dumps(&JsonValue::Array(rows.clone())));
        if let Some(rows) = &locked_annotations {
            validate_caption_size(rows, "locked annotations")?;
        }
        // Enhanced means the enhancer ran and its answer was accepted. An already-rich caption
        // may legitimately be returned unchanged; that is not a failed/fallback LLM execution.
        let mut report =
            PromptEnhancementReport::enhanced(self.original_prompt.clone(), caption.clone())
                .with_mode(self.mode.system_key());
        if let Some(rows) = locked_annotations {
            report = report.with_locked_annotations(rows);
        }
        Ok(UpsampleOutcome { caption, report })
    }
}

/// The stage's result: the effective caption and its provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpsampleOutcome {
    /// The caption the image model encodes.
    pub caption: String,
    pub report: PromptEnhancementReport,
}

/// What [`decide`] resolved an input to.
#[derive(Clone, Debug, PartialEq)]
pub enum UpsampleDecision {
    /// The upsampler does not run; the caption is final.
    Bypass(UpsampleOutcome),
    /// Run the plan, then [`UpsamplePlan::finish`].
    Run(UpsamplePlan),
}

/// Resolve an input: validate it, decide between the bypass paths and a run, preprocess the
/// references and budget the window. Mirrors `generate.py` + `pup.Upsampler.__call__` up to
/// the engine call.
pub fn decide(
    input: &UpsamplerInput,
    package: &UpsamplePackage,
    options: &UpsampleOptions,
) -> Result<UpsampleDecision> {
    let n_refs = input.references.len();
    if n_refs > MAX_REFERENCES {
        return Err(UpsampleError::TooManyReferences(n_refs).into());
    }
    let boxes = input.boxes.as_deref().filter(|b| !b.is_empty());
    let (inferred, user_text, locked) = plan_text(&input.prompt, boxes, n_refs)?;
    if inferred != input.mode {
        return Err(UpsampleError::ModeMismatch {
            requested: input.mode,
            inferred,
            references: n_refs,
            boxes: locked.is_some(),
        }
        .into());
    }
    let bypass = |outcome: PromptEnhancementOutcome,
                  reason: Option<&str>|
     -> Result<UpsampleDecision> {
        let caption = bypass_caption(&input.prompt, boxes)?;
        let mut report = PromptEnhancementReport {
            outcome,
            original_prompt: input.prompt.clone(),
            effective_prompt: caption.clone(),
            fallback_reason: reason.map(str::to_string),
            mode: None,
            locked_annotations: None,
        };
        if let Some(boxes) = boxes {
            report = report.with_locked_annotations(dumps(&JsonValue::Array(to_canvas(boxes)?)));
        }
        Ok(UpsampleDecision::Bypass(UpsampleOutcome {
            caption,
            report,
        }))
    };
    if input.bypass {
        return bypass(PromptEnhancementOutcome::Absent, None);
    }
    // A finished upsampler answer (its box table fully cited as <id>s) goes to the DiT as it is.
    if boxes.is_none() && is_finished(&input.prompt) {
        return bypass(PromptEnhancementOutcome::Fallback, Some("finished_answer"));
    }
    let mut references = Vec::with_capacity(n_refs);
    for image in &input.references {
        match policy_reference(image, n_refs) {
            Ok(reference) => references.push(reference),
            Err(Error::Unsupported(text)) if text.contains("aspect ratio of") => {
                // `except ValueError: upsampling skipped` — the prompt as typed.
                return bypass(
                    PromptEnhancementOutcome::Fallback,
                    Some("reference_aspect_ratio"),
                );
            }
            Err(e) => return Err(e),
        }
    }
    let user_text = if inferred.uses_references() {
        i2i_scaffold(&user_text, locked.is_some())
    } else {
        user_text
    };
    let mut plan = UpsamplePlan {
        mode: inferred,
        original_prompt: input.prompt.clone(),
        system_prompt: package.prompts().get(inferred).to_string(),
        user_text,
        locked,
        references,
        max_new_tokens: MAX_TOKENS,
    };
    plan.max_new_tokens = package.budget(&plan, options.max_tokens)?;
    Ok(UpsampleDecision::Run(plan))
}

/// A loaded upsampler: the package view plus a backend that runs a plan. The MLX and Candle
/// provider crates implement this over their `TextLlm` through [`run_text_llm`].
pub trait PromptUpsampler {
    fn package(&self) -> &UpsamplePackage;

    /// Run the plan and return the model's raw answer text.
    fn generate(
        &self,
        plan: &UpsamplePlan,
        options: &UpsampleOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<String>;
}

/// Drive a plan through a [`TextLlm`]: the generator and engine share one cancellation atomic
/// (including during prefill before the first token), and every content token is reported as
/// [`Progress::Upsampling`].
pub fn run_text_llm(
    llm: &dyn TextLlm,
    plan: &UpsamplePlan,
    options: &UpsampleOptions,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<String> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    let engine_cancel = core_llm::CancelFlag::from_arc(cancel.as_arc().clone());
    let request = plan.request(options, engine_cancel.clone());
    llm.validate(&request)
        .map_err(|e| Error::Unsupported(format!("flux3 prompt upsampler: {e}")))?;
    let max_tokens = plan.max_new_tokens;
    on_progress(Progress::Upsampling {
        tokens: 0,
        max_tokens,
    });
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    let mut tokens = 0u32;
    let output = llm.generate(&request, &mut |event| {
        if let StreamEvent::Token {
            channel: Channel::Content,
            ..
        } = &event
        {
            tokens = tokens.saturating_add(1);
            on_progress(Progress::Upsampling { tokens, max_tokens });
        }
        if cancel.is_cancelled() {
            engine_cancel.cancel();
        }
    });
    let output = match output {
        Ok(output) => output,
        Err(core_llm::Error::Canceled) => return Err(Error::Canceled),
        Err(e) => return Err(Error::Msg(format!("flux3 prompt upsampler: {e}"))),
    };
    if cancel.is_cancelled() || output.finish_reason == Some(FinishReason::Cancelled) {
        return Err(Error::Canceled);
    }
    Ok(output.text)
}

/// The whole pre-generation step: decide, run (when an upsampler is loaded), finish. The
/// effective caption and its [`PromptEnhancementReport`] come back; the generator encodes the
/// caption and emits the report.
pub fn upsample(
    upsampler: Option<&dyn PromptUpsampler>,
    input: &UpsamplerInput,
    options: &UpsampleOptions,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<UpsampleOutcome> {
    let Some(upsampler) = upsampler else {
        if input.bypass {
            // No upsampler and none asked for: the identity path, which never needs the package.
            let boxes = input.boxes.as_deref().filter(|b| !b.is_empty());
            let caption = bypass_caption(&input.prompt, boxes)?;
            let mut report = PromptEnhancementReport {
                outcome: PromptEnhancementOutcome::Absent,
                original_prompt: input.prompt.clone(),
                effective_prompt: caption.clone(),
                fallback_reason: None,
                mode: None,
                locked_annotations: None,
            };
            if let Some(boxes) = boxes {
                report =
                    report.with_locked_annotations(dumps(&JsonValue::Array(to_canvas(boxes)?)));
            }
            return Ok(UpsampleOutcome { caption, report });
        }
        return Err(UpsampleError::MissingUpsampler.into());
    };
    match decide(input, upsampler.package(), options)? {
        UpsampleDecision::Bypass(outcome) => Ok(outcome),
        UpsampleDecision::Run(plan) => {
            let raw = upsampler.generate(&plan, options, cancel, on_progress)?;
            plan.finish(&raw)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(entries: &[(&str, JsonValue)]) -> JsonValue {
        JsonValue::Object(
            entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        )
    }

    fn ints(v: &[i64]) -> JsonValue {
        JsonValue::Array(v.iter().map(|i| JsonValue::Int(*i)).collect())
    }

    #[test]
    fn embedded_prompts_are_the_pinned_file() {
        use sha2::{Digest, Sha256};
        let digest = format!("{:x}", Sha256::digest(SYSTEM_PROMPTS_JSON.as_bytes()));
        assert_eq!(digest, SYSTEM_PROMPTS_SHA256);
        let prompts = SystemPrompts::builtin();
        assert_eq!(prompts.t2i.len(), 3404);
        assert_eq!(prompts.i2i.len(), 2677);
        assert_eq!(prompts.t2i_bbox.len(), 3302);
        assert_eq!(prompts.i2i_bbox.len(), 3034);
        for mode in UpsampleMode::ALL {
            assert!(prompts.get(mode).ends_with("."), "{mode}");
            assert_eq!(UpsampleMode::from_system_key(mode.system_key()), Some(mode));
        }
        assert!(SystemPrompts::parse(r#"{"t2i": "a"}"#).is_err());
    }

    #[test]
    fn modes_follow_the_task_and_box_switches() {
        assert_eq!(UpsampleMode::infer(0, false), UpsampleMode::Text);
        assert_eq!(UpsampleMode::infer(3, false), UpsampleMode::Editing);
        assert_eq!(UpsampleMode::infer(0, true), UpsampleMode::TextWithBoxes);
        assert_eq!(UpsampleMode::infer(1, true), UpsampleMode::EditingWithBoxes);
        assert_eq!(UpsampleMode::Text.presence_penalty(), 1.5);
        assert_eq!(UpsampleMode::TextWithBoxes.presence_penalty(), 1.5);
        assert_eq!(UpsampleMode::Editing.presence_penalty(), 0.0);
        assert_eq!(UpsampleMode::EditingWithBoxes.task(), "i2i");
        assert_eq!(UpsampleMode::TextWithBoxes.task(), "t2i");
        assert_eq!(max_pixels_by_refs(0), None);
        assert_eq!(max_pixels_by_refs(1), Some(1_048_576));
        assert_eq!(max_pixels_by_refs(2), Some(524_176));
        assert_eq!(max_pixels_by_refs(10), Some(262_144));
        assert_eq!(max_pixels_by_refs(11), None);
    }

    #[test]
    fn scaffolds_substitute_the_instruction_verbatim() {
        let s = i2i_scaffold("add a {hat} \\n", false);
        assert_eq!(
            s,
            "Reference images follow in order (the first image, the second image, ...).\nUser \
             request: add a {hat} \\n\nOutput the caption."
        );
        assert!(i2i_scaffold("x", true).starts_with("Reference images follow in attachment order: ref_image_0, ref_image_1, ...\nUser editing instruction: x\n"));
    }

    #[test]
    fn to_canvas_rounds_half_to_even_and_keeps_upstream_key_order() {
        let boxes = BoxSpec::parse_list(
            r#"[{"id": "obj_1", "desc": "a red balloon", "bbox": [0.0825, 0.0835, 0.5, 1.5]},
                {"desc": null, "from": "target", "src_bbox": null, "tgt_bbox": [0.55, 0.62, 0.85, 0.85]},
                {"id": "", "bbox": [-0.2, 0.0, 0.0005, 0.0015]}]"#,
        )
        .unwrap();
        let rows = to_canvas(&boxes).unwrap();
        assert_eq!(
            dumps(&JsonValue::Array(rows)),
            r#"[{"id": "obj_1", "bbox": [82, 84, 500, 1000], "desc": "a red balloon"}, {"id": "obj_2", "from": "target", "src_bbox": null, "tgt_bbox": [550, 620, 850, 850], "desc": ""}, {"id": "obj_3", "bbox": [0, 0, 0, 2], "desc": ""}]"#
        );
        let bad = BoxSpec {
            bbox: Some(vec![0.1, 0.2]),
            ..Default::default()
        };
        assert!(
            matches!(to_canvas(&[bad]), Err(Error::Unsupported(m)) if m.contains("[y_min, x_min, y_max, x_max]"))
        );
        assert!(
            matches!(to_canvas(&[BoxSpec::default()]), Err(Error::Unsupported(m)) if m.contains("neither bbox"))
        );
        assert!(BoxSpec::parse_list(r#"[{"bbox": ["0.1", 0, 1, 1]}]"#).is_err());
        assert!(BoxSpec::parse_list(r#"[1]"#).is_err());
    }

    #[test]
    fn split_finds_the_last_parsing_list_and_strips() {
        let (caption, rows) = split("a sign reading \"[SALE]\" <sign_1>  [{\"id\": \"sign_1\", \"bbox\": [1, 2, 3, 4], \"desc\": \"x\"}]\n");
        assert_eq!(caption, "a sign reading \"[SALE]\" <sign_1>");
        assert_eq!(rows.unwrap().len(), 1);
        // Two lists: the LAST `[` that parses wins, the earlier one stays in the caption.
        let (caption, rows) = split("x [1] y [2]");
        assert_eq!((caption, rows), ("x [1] y", Some(vec![JsonValue::Int(2)])));
    }

    #[test]
    fn split_handles_a_bare_bracket_list_and_none() {
        // "[SALE]" is not JSON (bare word) → not a list; the caption keeps it, stripped.
        assert_eq!(split("  plain [SALE]  "), ("plain [SALE]", None));
        // A list that is not a box table is still a list for `split`; `detect` rejects it.
        let (c, rows) = split("ends with [1, 2]");
        assert_eq!(
            (c, rows),
            (
                "ends with",
                Some(vec![JsonValue::Int(1), JsonValue::Int(2)])
            )
        );
        assert_eq!(detect("ends with [1, 2]"), None);
        assert_eq!(split("\u{1f}x\u{1f}"), ("x", None));
    }

    #[test]
    fn detect_requires_ids_and_canvas_boxes_and_unwraps_the_locked_input() {
        let rows = "[{\"id\": \"obj_1\", \"bbox\": [80, 60, 480, 380], \"desc\": \"b\"}]";
        let (prompt, found) = detect(&format!("a balloon <obj_1> {rows}")).unwrap();
        assert_eq!(prompt, "a balloon <obj_1>");
        assert_eq!(found.len(), 1);
        // 0–1 coordinates are the input grid, not the canvas.
        assert_eq!(
            detect("a [{\"id\": \"obj_1\", \"bbox\": [0.1, 0.1, 0.5, 0.5]}]"),
            None
        );
        // A bad id.
        assert_eq!(
            detect("a [{\"id\": \"1obj\", \"bbox\": [80, 60, 480, 380]}]"),
            None
        );
        // The upsampler's own input form unwraps to the original prompt.
        let input = upsampling_input("  two balloons  ", loads(rows).unwrap().as_array().unwrap());
        let (prompt, found) = detect(&input).unwrap();
        assert_eq!(prompt, "two balloons");
        assert_eq!(found.len(), 1);
        assert!(!is_finished(&input));
        assert!(is_finished(&format!("a balloon <obj_1> {rows}")));
        assert!(!is_finished(&format!("a balloon {rows}")));
        assert_eq!(detect("[]"), None);
    }

    #[test]
    fn plan_text_selects_the_four_modes() {
        let boxes =
            BoxSpec::parse_list(r#"[{"id": "obj_1", "desc": "d", "bbox": [0.1, 0.2, 0.3, 0.4]}]"#)
                .unwrap();
        let (mode, text, locked) = plan_text("p", None, 0).unwrap();
        assert_eq!(
            (mode, text.as_str(), locked.is_none()),
            (UpsampleMode::Text, "p", true)
        );
        let (mode, text, _) = plan_text("p", None, 2).unwrap();
        assert_eq!((mode, text.as_str()), (UpsampleMode::Editing, "p"));
        let (mode, text, locked) = plan_text(" p ", Some(&boxes), 0).unwrap();
        assert_eq!(mode, UpsampleMode::TextWithBoxes);
        assert_eq!(
            dumps(&JsonValue::Array(locked.unwrap())),
            r#"[{"id": "obj_1", "bbox": [100, 200, 300, 400], "desc": "d"}]"#
        );
        assert!(text.starts_with("ORIGINAL PROMPT:\np\n\nLOCKED BOUNDING-BOX ANNOTATIONS ("));
        assert!(text.ends_with(
            ":\n[{\"id\": \"obj_1\", \"bbox\": [100, 200, 300, 400], \"desc\": \"d\"}]"
        ));
        let (mode, _, _) = plan_text("p", Some(&boxes), 1).unwrap();
        assert_eq!(mode, UpsampleMode::EditingWithBoxes);
        // An empty box list is "no boxes".
        let (mode, _, _) = plan_text("p", Some(&[]), 0).unwrap();
        assert_eq!(mode, UpsampleMode::Text);
        // Boxes in the prompt switch the mode by themselves.
        let (mode, text, _) = plan_text(
            "a <obj_1> [{\"id\": \"obj_1\", \"bbox\": [80, 60, 480, 380]}]",
            None,
            0,
        )
        .unwrap();
        assert_eq!(mode, UpsampleMode::TextWithBoxes);
        assert!(text.starts_with("ORIGINAL PROMPT:\na <obj_1>\n\n"));
        assert_eq!(
            bypass_caption(" p \n", Some(&boxes)).unwrap(),
            " p [{\"id\": \"obj_1\", \"bbox\": [100, 200, 300, 400], \"desc\": \"d\"}]"
        );
        assert_eq!(bypass_caption(" p \n", None).unwrap(), " p \n");
    }

    #[test]
    fn valid_box_and_valid_added_follow_bbox_py() {
        assert_eq!(
            valid_box(Some(&ints(&[0, 0, 1000, 1000]))).unwrap().len(),
            4
        );
        assert_eq!(valid_box(Some(&ints(&[0, 0, 1, 1]))), None, "all in 0-1");
        assert_eq!(valid_box(Some(&ints(&[10, 0, 10, 50]))), None, "degenerate");
        assert_eq!(valid_box(Some(&ints(&[0, 0, 1001, 5]))), None, "off canvas");
        assert_eq!(valid_box(Some(&ints(&[0, 0, 5]))), None);
        assert_eq!(
            valid_box(Some(&JsonValue::Array(vec![
                JsonValue::Bool(true),
                JsonValue::Int(0),
                JsonValue::Int(5),
                JsonValue::Int(5)
            ]))),
            None
        );
        let kept = valid_box(Some(&JsonValue::Array(vec![
            JsonValue::Float(0.0),
            JsonValue::Int(0),
            JsonValue::Float(5.5),
            JsonValue::Int(5),
        ])))
        .unwrap();
        assert_eq!(dumps(&JsonValue::Array(kept)), "[0.0, 0, 5.5, 5]");

        let caption = "a <dog_1> and <cup_2>";
        let row = obj(&[
            ("id", JsonValue::Str("dog_1".into())),
            ("bbox", ints(&[1, 2, 3, 4])),
            ("desc", JsonValue::Str("  a dog ".into())),
            ("extra", JsonValue::Int(1)),
        ]);
        assert_eq!(
            dumps(&valid_added(&row, caption).unwrap()),
            r#"{"id": "dog_1", "bbox": [1, 2, 3, 4], "desc": "a dog"}"#
        );
        let uncited = obj(&[
            ("id", JsonValue::Str("cat_1".into())),
            ("bbox", ints(&[1, 2, 3, 4])),
            ("desc", JsonValue::Str("c".into())),
        ]);
        assert_eq!(valid_added(&uncited, caption), None);
        let edit = obj(&[
            ("id", JsonValue::Str("cup_2".into())),
            ("desc", JsonValue::Str("c".into())),
            ("from", JsonValue::Str("ref_image_0".into())),
            ("src_bbox", ints(&[1, 2, 3, 4])),
            ("tgt_bbox", JsonValue::Null),
        ]);
        assert_eq!(
            dumps(&valid_added(&edit, caption).unwrap()),
            r#"{"id": "cup_2", "from": "ref_image_0", "src_bbox": [1, 2, 3, 4], "tgt_bbox": null, "desc": "c"}"#
        );
        // target rows carry no src box; a ref row must carry one.
        let bad_target = obj(&[
            ("id", JsonValue::Str("cup_2".into())),
            ("desc", JsonValue::Str("c".into())),
            ("from", JsonValue::Str("target".into())),
            ("src_bbox", ints(&[1, 2, 3, 4])),
        ]);
        assert_eq!(valid_added(&bad_target, caption), None);
        let bad_ref = obj(&[
            ("id", JsonValue::Str("cup_2".into())),
            ("desc", JsonValue::Str("c".into())),
            ("from", JsonValue::Str("ref_image_1".into())),
            ("tgt_bbox", ints(&[1, 2, 3, 4])),
        ]);
        assert_eq!(valid_added(&bad_ref, caption), None);
        let bad_src = obj(&[
            ("id", JsonValue::Str("cup_2".into())),
            ("desc", JsonValue::Str("c".into())),
            ("from", JsonValue::Str("ref_image_1".into())),
            ("src_bbox", ints(&[1, 2])),
            ("tgt_bbox", ints(&[1, 2, 3, 4])),
        ]);
        assert_eq!(valid_added(&bad_src, caption), None);
        assert!(is_row_origin("ref_image_12") && is_row_origin("target"));
        assert!(
            !is_row_origin("ref_image_")
                && !is_row_origin("ref_image_1x")
                && !is_row_origin("Target")
        );
        assert!(is_row_id("En_Text_1") && !is_row_id("_x") && !is_row_id("") && !is_row_id("a-b"));
    }

    #[test]
    fn reconcile_renames_adopts_descs_and_appends_valid_rows() {
        let locked = loads(r#"[{"id": "obj_1", "bbox": [80, 60, 480, 380], "desc": "a red balloon"}, {"id": "obj_2", "bbox": [200, 620, 420, 800], "desc": "a blue balloon"}]"#).unwrap().as_array().unwrap().to_vec();
        let answer = "Full-color photograph of a red balloon <balloon_1> and a blue balloon <obj_2> over a sea <sea_1>. [{\"id\": \"balloon_1\", \"bbox\": [82, 60, 480, 380], \"desc\": \" A red balloon with stripes \"}, {\"id\": \"obj_2\", \"bbox\": [1, 1, 2, 2], \"desc\": \"\"}, {\"id\": \"sea_1\", \"bbox\": [600, 0, 1000, 1000], \"desc\": \"grey sea\"}, {\"id\": \"ghost_1\", \"bbox\": [0, 0, 10, 10], \"desc\": \"uncited\"}, 7]";
        let out = reconcile(answer, &locked).unwrap();
        assert_eq!(
            out,
            "Full-color photograph of a red balloon <obj_1> and a blue balloon <obj_2> over a sea <sea_1>. [{\"id\": \"obj_1\", \"bbox\": [80, 60, 480, 380], \"desc\": \"A red balloon with stripes\"}, {\"id\": \"obj_2\", \"bbox\": [200, 620, 420, 800], \"desc\": \"a blue balloon\"}, {\"id\": \"sea_1\", \"bbox\": [600, 0, 1000, 1000], \"desc\": \"grey sea\"}]"
        );
        // No table at all: the locks are appended with their own descs.
        assert_eq!(
            reconcile("  just prose  ", &locked).unwrap(),
            format!("just prose {}", dumps(&JsonValue::Array(locked.clone())))
        );
        // A locked row from a user prompt keeps its arbitrary keys and order; `desc` is added.
        let odd =
            loads(r#"[{"z": 1, "id": "obj_1", "src_bbox": [1, 2, 3, 4], "from": "ref_image_0"}]"#)
                .unwrap()
                .as_array()
                .unwrap()
                .to_vec();
        assert_eq!(reconcile("x <obj_1> [{\"id\": \"obj_1\", \"desc\": \"d\"}]", &odd).unwrap(), "x <obj_1> [{\"z\": 1, \"id\": \"obj_1\", \"src_bbox\": [1, 2, 3, 4], \"from\": \"ref_image_0\", \"desc\": \"d\"}]");
        // A locked row with a non-string desc keeps it when the model gives none.
        let typed = loads(r#"[{"id": "obj_1", "bbox": [1, 2, 3, 4], "desc": null}]"#)
            .unwrap()
            .as_array()
            .unwrap()
            .to_vec();
        assert_eq!(
            reconcile("x", &typed).unwrap(),
            "x [{\"id\": \"obj_1\", \"bbox\": [1, 2, 3, 4], \"desc\": null}]"
        );
        let bx = |v: &[i64]| -> Vec<JsonValue> { v.iter().map(|i| JsonValue::Int(*i)).collect() };
        assert!((iou(&bx(&[0, 0, 10, 10]), &bx(&[0, 0, 10, 10])).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(
            iou(&bx(&[0, 0, 10, 10]), &bx(&[20, 20, 30, 30])).unwrap(),
            0.0
        );
        assert!(
            (iou(&bx(&[0, 0, 10, 10]), &bx(&[5, 0, 15, 10])).unwrap() - 50.0 / 150.0).abs() < 1e-12
        );
    }

    #[test]
    fn reference_size_follows_the_budget_table() {
        assert_eq!(reference_size(1024, 1024, 1).unwrap(), (1024, 1024));
        assert_eq!(reference_size(1000, 1000, 1).unwrap(), (992, 992));
        assert_eq!(reference_size(2000, 1000, 1).unwrap(), (1440, 704));
        assert_eq!(reference_size(1024, 1024, 2).unwrap(), (704, 704));
        assert_eq!(reference_size(1024, 1024, 3).unwrap(), (512, 512));
        assert_eq!(reference_size(10, 10, 1).unwrap(), (32, 32));
        assert_eq!(
            reference_size(48, 48, 1).unwrap(),
            (64, 64),
            "48/32 = 1.5 rounds half to even: 2"
        );
        assert_eq!(
            reference_size(80, 80, 1).unwrap(),
            (64, 64),
            "80/32 = 2.5 rounds half to even: 2"
        );
        assert_eq!(
            reference_size(112, 112, 1).unwrap(),
            (128, 128),
            "112/32 = 3.5 rounds half to even: 4"
        );
        assert_eq!(reference_size(4000, 32, 10).unwrap(), (4000 / 32 * 32, 32));
        assert_eq!(
            reference_size(16000, 80, 1).unwrap(),
            (16000, 64),
            "under the 1 MP budget: no scaling"
        );
        assert_eq!(
            reference_size(16000, 100, 1).unwrap(),
            (12928, 64),
            "one floor pass under beta"
        );
        assert!(
            matches!(reference_size(201, 1, 1), Err(Error::Unsupported(m)) if m.contains("aspect ratio of 201x1"))
        );
        assert!(reference_size(0, 10, 1).is_err());
        assert!(
            matches!(reference_size(10, 10, 11), Err(Error::Unsupported(m)) if m.contains("at most 10"))
        );
    }

    #[test]
    fn vision_tokens_follow_smart_resize() {
        assert_eq!(vision_tokens(1024, 1024).unwrap(), 1024);
        assert_eq!(
            vision_tokens(32, 32).unwrap(),
            64,
            "upscaled to the 65536-pixel floor (256x256)"
        );
        assert_eq!(vision_tokens(512, 512).unwrap(), 256);
        assert_eq!(vision_tokens(704, 704).unwrap(), 484);
        assert_eq!(vision_tokens(1440, 704).unwrap(), 45 * 22);
        assert!(vision_tokens(0, 5).is_err());
    }

    #[test]
    fn composite_over_white_matches_pillow_fixed_point() {
        // Opaque and transparent are exact; a half-alpha black pixel lands on 128 (Pillow:
        // alpha_composite(white, (0,0,0,128)) == (128,128,128)).
        assert_eq!(composite_over_white(&[10, 20, 30, 255]), vec![10, 20, 30]);
        assert_eq!(composite_over_white(&[10, 20, 30, 0]), vec![255, 255, 255]);
        assert_eq!(composite_over_white(&[0, 0, 0, 128]), vec![127, 127, 127]);
        assert_eq!(composite_over_white(&[0, 0, 0, 1]), vec![254, 254, 254]);
        assert_eq!(composite_over_white(&[255, 0, 0, 200]), vec![255, 55, 55]);
    }

    #[test]
    fn errors_map_to_the_typed_classes() {
        assert!(
            matches!(Error::from(UpsampleError::MissingUpsampler), Error::Unsupported(m) if m.contains(UPSAMPLER_COMPONENT))
        );
        assert!(
            matches!(Error::from(UpsampleError::EmptyGeneration), Error::Msg(m) if m.ends_with("empty generation"))
        );
        assert_eq!(
            UpsampleError::ReferenceAspectRatio {
                width: 1,
                height: 300
            }
            .fallback_reason(),
            Some("reference_aspect_ratio")
        );
        assert_eq!(UpsampleError::TooManyReferences(11).fallback_reason(), None);
    }

    fn finish_plan(prompt: &str, locked: Option<Vec<JsonValue>>) -> UpsamplePlan {
        UpsamplePlan {
            mode: if locked.is_some() {
                UpsampleMode::TextWithBoxes
            } else {
                UpsampleMode::Text
            },
            original_prompt: prompt.to_owned(),
            system_prompt: String::new(),
            user_text: String::new(),
            locked,
            references: vec![],
            max_new_tokens: MAX_TOKENS,
        }
    }

    #[test]
    fn accepted_unchanged_caption_is_successful_not_a_failed_fallback() {
        let out = finish_plan("a rich caption", None)
            .finish("a rich caption")
            .unwrap();
        assert_eq!(out.report.outcome, PromptEnhancementOutcome::Enhanced);
        assert_eq!(out.report.original_prompt, out.report.effective_prompt);
        assert_eq!(out.report.mode.as_deref(), Some("t2i"));
        assert_eq!(out.report.fallback_reason, None);
    }

    #[test]
    fn complete_caption_admission_counts_unicode_scalars_not_bytes_or_tokens() {
        let plan = finish_plan("original", None);
        let edge = "界".repeat(MAX_EFFECTIVE_CAPTION_CHARS);
        assert_eq!(edge.len(), MAX_EFFECTIVE_CAPTION_CHARS * 3);
        assert_eq!(plan.finish(&edge).unwrap().caption, edge);
        let over = format!("界{edge}");
        assert!(
            matches!(plan.finish(&over), Err(Error::Unsupported(message)) if message.contains("effective caption") && message.contains("16001"))
        );
    }

    #[test]
    fn admission_covers_complete_reconciled_rows_and_original_locked_metadata() {
        let mut locked = loads(r#"[{"id":"obj_1","bbox":[10,20,30,40],"desc":"locked"}]"#)
            .unwrap()
            .as_array()
            .unwrap()
            .to_vec();
        locked[0].set(
            "desc",
            JsonValue::Str("界".repeat(MAX_EFFECTIVE_CAPTION_CHARS)),
        );
        let plan = finish_plan("original", Some(locked));
        assert!(
            matches!(plan.finish("scene <obj_1>"), Err(Error::Unsupported(message)) if message.contains("effective caption"))
        );
        // A concise generated desc shrinks the rendered table, but the original authoritative
        // metadata is still carried verbatim and must fit BEFORE any renderer is called.
        let short_answer = r#"scene <obj_1> [{"id":"obj_1","bbox":[10,20,30,40],"desc":"short"}]"#;
        assert!(
            matches!(plan.finish(short_answer), Err(Error::Unsupported(message)) if message.contains("locked annotations"))
        );
    }

    #[test]
    fn f32_and_rgba_references_become_pillow_rgb() {
        let f = ReferenceImage::F32Rgb {
            width: 1,
            height: 1,
            pixels: vec![0.0, 0.5, 1.0],
        };
        assert_eq!(f.to_rgb8().unwrap(), (1, 1, vec![0, 128, 255]));
        let a = ReferenceImage::Rgba8 {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0, 0],
        };
        assert_eq!(a.to_rgb8().unwrap(), (1, 1, vec![255, 255, 255]));
        let bad = ReferenceImage::Rgb8 {
            width: 2,
            height: 1,
            pixels: vec![0; 3],
        };
        assert!(bad.to_rgb8().is_err());
    }

    #[test]
    fn policy_reference_hits_the_budget_and_survives_the_jpeg_round_trip() {
        let (w, h) = (70u32, 50u32);
        let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let r = policy_reference(
            &ReferenceImage::Rgb8 {
                width: w,
                height: h,
                pixels: pixels.clone(),
            },
            1,
        )
        .unwrap();
        assert_eq!((r.width, r.height), (64, 64));
        assert_eq!(r.pixels.len(), 64 * 64 * 3);
        // A flat image is JPEG-stable to within a couple of levels.
        let flat = policy_reference(
            &ReferenceImage::Rgb8 {
                width: 64,
                height: 32,
                pixels: vec![200; 64 * 32 * 3],
            },
            2,
        )
        .unwrap();
        assert_eq!((flat.width, flat.height), (64, 32));
        assert!(flat.pixels.iter().all(|&p| (195..=205).contains(&p)));
    }
}
