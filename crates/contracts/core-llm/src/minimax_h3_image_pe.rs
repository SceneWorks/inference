//! MiniMax-H3-Image-PE — the prompt expander for MiniMax-H3-Image (SW-7099, feature SW-F363).
//!
//! `MiniMaxAI/MiniMax-H3-Image-PE` is a fine-tune of Qwen3.8-27B (`qwen3_5`, 64 layers, three Gated
//! DeltaNet layers per full-attention layer, Qwen3-VL vision tower). It runs on the ordinary
//! `Architecture::Qwen35` lane of both backends; this module is the tensor-free adapter around
//! that lane — which files hold the prompts, how a request is assembled, and what the reply means.
//! It loads no weights.
//!
//! # Prompts are read from the model's own files, never copied
//!
//! The system and task prompts are MiniMax Materials under the MiniMax-H3-Image Community License.
//! They are read at run time from the **upstream** `MiniMaxAI/MiniMax-H3-Image-PE` snapshot at
//! [`REVISION`] — its `prompt/` directory — and each is checked against the SHA-256 the snapshot's
//! `prompt/manifest.json` lists, and the manifest itself against [`MANIFEST_SHA256`]. A converted
//! tier (bf16 / Q8 / Q4, local or published) never carries `prompt/`: the weights load from the
//! tier, the prompts from the upstream snapshot, as the Qwen-Image rewriter never redistributes
//! Qwen's prompts. Nothing licensed is vendored here, tests included: the tests build synthetic
//! prompt trees.
//!
//! # The two workflows (model card, revision [`REVISION`])
//!
//! - **Text-to-image**: a system turn (`prompt/t2i/system_prompt.txt`, trailing newlines removed)
//!   and a user turn (`prompt/t2i/task_prompt.txt` with the short prompt substituted by Python
//!   `str.format`). Reply: `{"expanded_caption", "aspect_ratio"}`, the ratio one of the enum in
//!   `prompt/t2i/request_params.json`.
//! - **Edit**: one user turn of interleaved parts — the edit task prompt, the reference count, one
//!   `<Picture N>` label (with the reference's reduced source ratio) before each image, then the
//!   instruction (JSON-quoted) and the target-ratio selection. Reply: `{"expanded_caption",
//!   "target_aspect_ratio"}`, the ratio a positive-integer `W:H`, equal to an explicit selection.
//!
//! Both use the published sampling recipe from `request_params.json` (temperature 1.0, top-p 0.95,
//! top-k 20, 4096 new tokens covering thinking and answer, `reasoning_effort` xhigh, thinking and
//! `preserve_thinking` on). The card ships no edit-specific parameters; the Diffusers blocks it
//! recommends reuse the T2I ones for edit, and so does this adapter.
//!
//! The card serves the expander with a reasoning parser and XGrammar (`--reasoning-parser qwen3
//! --grammar-backend xgrammar`, `response_format` = the reply schema): the thinking trace is free,
//! the answer after `</think>` is grammar-constrained, and the client (`scripts/expand_prompt.py`)
//! `json.loads` the answer with no retry, fallback or extraction. This adapter mirrors that: every
//! request carries [`Constraint::Json`], which both backends apply only after the reasoning block
//! closes, so the answer is always well-formed JSON (an unconstrained run produced prose or fenced
//! answers, SW-7099 run 42954971396, candle-bf16 `t2i-0` and mlx-q4 `t2i-3`). The schema itself
//! (exact keys, ratio enum) is not a decode constraint here; [`parse_t2i`] / [`parse_edit`] check it
//! on the finished answer, and a reply that is not exactly the schema is refused, never repaired.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::constraint::Constraint;
use crate::message::{Content, ImageRef, Message, Role};
use crate::request::{ReasoningEffort, Sampling, TextLlmRequest, ThinkingMode};

/// Catalog id of the prompt expander (its own model, not a variant of the image model).
pub const MODEL_ID: &str = "minimax_h3_image_pe";
/// Hugging Face repository of the published NVFP4 weights and prompts.
pub const REPO: &str = "MiniMaxAI/MiniMax-H3-Image-PE";
/// The revision this adapter was written against (read 2026-11-10).
pub const REVISION: &str = "35603d4b63949afa12cb098227fffefedf081afa";
/// SHA-256 of `prompt/manifest.json` at [`REVISION`].
pub const MANIFEST_SHA256: &str =
    "6b0969322669d9713e33148d36c401f73b18939337ea2384d90dc457430b6a5b";
/// The most references the edit workflow accepts (MiniMax-H3-Image's edit cap).
pub const MAX_REFERENCES: usize = 16;

const MANIFEST: &str = "prompt/manifest.json";
const T2I_SYSTEM: &str = "prompt/t2i/system_prompt.txt";
const T2I_TASK: &str = "prompt/t2i/task_prompt.txt";
const T2I_PARAMS: &str = "prompt/t2i/request_params.json";
const EDIT_TASK: &str = "prompt/edit/task_prompt.txt";

/// The published request recipe (`prompt/t2i/request_params.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct RequestParams {
    pub sampling: Sampling,
    pub max_new_tokens: u32,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub enable_thinking: bool,
    pub preserve_thinking: Option<bool>,
    /// The T2I reply's allowed `aspect_ratio` values, in the schema's order.
    pub aspect_ratios: Vec<String>,
}

/// The verified prompt files of one snapshot.
#[derive(Clone, Debug)]
pub struct PromptFiles {
    t2i_system: String,
    t2i_task: String,
    edit_task: String,
    params: RequestParams,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn read(root: &Path, rel: &str) -> Result<Vec<u8>, String> {
    let path: PathBuf = root.join(rel);
    std::fs::read(&path).map_err(|e| format!("{REPO}: cannot read {}: {e}", path.display()))
}

impl PromptFiles {
    /// Read and verify the prompt files under `root` (a snapshot directory holding `prompt/`):
    /// the manifest must be the pinned revision's, and every file must match its manifest digest.
    pub fn load_verified(root: &Path) -> Result<Self, String> {
        Self::load_with_manifest_digest(root, MANIFEST_SHA256)
    }

    /// [`Self::load_verified`] against an explicit manifest digest (tests use a synthetic tree).
    pub fn load_with_manifest_digest(root: &Path, manifest_sha256: &str) -> Result<Self, String> {
        let manifest_bytes = read(root, MANIFEST)?;
        let digest = sha256_hex(&manifest_bytes);
        if digest != manifest_sha256 {
            return Err(format!(
                "{REPO}: {MANIFEST} has SHA-256 {digest}, expected {manifest_sha256} (revision \
                 {REVISION}); the prompts changed upstream or the download is corrupt"
            ));
        }
        let manifest: Value = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| format!("{REPO}: {MANIFEST} is not JSON: {e}"))?;
        let verified = |rel: &str| -> Result<String, String> {
            let expected = ["t2i", "edit"]
                .iter()
                .find_map(|mode| {
                    manifest
                        .get(mode)?
                        .get("files")?
                        .get(rel)?
                        .get("sha256")?
                        .as_str()
                })
                .ok_or_else(|| format!("{REPO}: {MANIFEST} lists no digest for {rel}"))?;
            let bytes = read(root, rel)?;
            let actual = sha256_hex(&bytes);
            if actual != expected {
                return Err(format!(
                    "{REPO}: {rel} has SHA-256 {actual}, but {MANIFEST} lists {expected}"
                ));
            }
            String::from_utf8(bytes).map_err(|_| format!("{REPO}: {rel} is not UTF-8"))
        };
        let params_text = verified(T2I_PARAMS)?;
        let params = parse_request_params(&params_text)?;
        Ok(Self {
            t2i_system: verified(T2I_SYSTEM)?.trim_end_matches('\n').to_string(),
            t2i_task: verified(T2I_TASK)?,
            edit_task: verified(EDIT_TASK)?,
            params,
        })
    }

    /// The published request recipe.
    pub fn params(&self) -> &RequestParams {
        &self.params
    }
}

/// Parse `request_params.json` into the backend-neutral request knobs.
pub fn parse_request_params(text: &str) -> Result<RequestParams, String> {
    let v: Value =
        serde_json::from_str(text).map_err(|e| format!("{REPO}: {T2I_PARAMS} is not JSON: {e}"))?;
    let num = |k: &str| -> Result<f64, String> {
        v.get(k)
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("{REPO}: {T2I_PARAMS} has no numeric `{k}`"))
    };
    let effort = match v.get("reasoning_effort").and_then(Value::as_str) {
        None => None,
        Some("xhigh") => Some(ReasoningEffort::XHigh),
        Some("medium") => Some(ReasoningEffort::Medium),
        Some("low") => Some(ReasoningEffort::Low),
        Some(other) => return Err(format!("{REPO}: unknown reasoning_effort {other:?}")),
    };
    let kwargs = v.get("chat_template_kwargs");
    let kwarg = |k: &str| kwargs.and_then(|o| o.get(k)).and_then(Value::as_bool);
    let ratios = v
        .pointer("/response_format/json_schema/schema/properties/aspect_ratio/enum")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{REPO}: {T2I_PARAMS} has no aspect_ratio enum"))?
        .iter()
        .map(|r| r.as_str().map(str::to_string))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| format!("{REPO}: {T2I_PARAMS} aspect_ratio enum is not all strings"))?;
    let min_p = v.get("min_p").and_then(Value::as_f64).unwrap_or(0.0);
    if min_p != 0.0 {
        return Err(format!("{REPO}: min_p {min_p} is not supported"));
    }
    Ok(RequestParams {
        sampling: Sampling {
            temperature: num("temperature")? as f32,
            top_p: num("top_p")? as f32,
            top_k: num("top_k")? as usize,
            presence_penalty: v
                .get("presence_penalty")
                .and_then(Value::as_f64)
                .unwrap_or(0.0) as f32,
            presence_penalty_generated_only: false,
            repetition_penalty: v
                .get("repetition_penalty")
                .and_then(Value::as_f64)
                .unwrap_or(1.0) as f32,
            repetition_context: 0,
        },
        max_new_tokens: num("max_tokens")? as u32,
        reasoning_effort: effort,
        enable_thinking: kwarg("enable_thinking").unwrap_or(true),
        preserve_thinking: kwarg("preserve_thinking"),
        aspect_ratios: ratios,
    })
}

/// Python `str.format(arg)` for a template whose only replacement field is `{}` / `{0}`;
/// `{{` / `}}` are literal braces. Any other field is refused (the template changed shape).
pub fn python_format_one(template: &str, arg: &str) -> Result<String, String> {
    let mut out = String::with_capacity(template.len() + arg.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                out.push('}');
            }
            '{' => {
                let mut field = String::new();
                loop {
                    match chars.next() {
                        Some('}') => break,
                        Some(ch) => field.push(ch),
                        None => return Err("unterminated replacement field".into()),
                    }
                }
                if !(field.is_empty() || field == "0") {
                    return Err(format!("unsupported replacement field {{{field}}}"));
                }
                out.push_str(arg);
            }
            '}' => return Err("single '}' in format string".into()),
            other => out.push(other),
        }
    }
    Ok(out)
}

fn request(messages: Vec<Message>, params: &RequestParams) -> TextLlmRequest {
    let mut req = TextLlmRequest::new(messages, params.max_new_tokens);
    req.sampling = params.sampling;
    req.thinking = if params.enable_thinking {
        ThinkingMode::Enabled
    } else {
        ThinkingMode::Disabled
    };
    req.reasoning_effort = params.reasoning_effort;
    req.preserve_thinking = params.preserve_thinking;
    // Upstream constrains the answer with XGrammar after the reasoning parser splits the thinking
    // off (module docs); the backends apply this constraint only after `</think>`.
    req.constraint = Some(Constraint::Json);
    req
}

/// The text-to-image expansion request for a short `prompt`.
pub fn t2i_request(files: &PromptFiles, prompt: &str) -> Result<TextLlmRequest, String> {
    if prompt.trim().is_empty() {
        return Err("prompt must be a non-empty string".into());
    }
    let user = python_format_one(&files.t2i_task, prompt)
        .map_err(|e| format!("{REPO}: {T2I_TASK}: {e}"))?;
    Ok(request(
        vec![
            Message::text(Role::System, files.t2i_system.clone()),
            Message::text(Role::User, user),
        ],
        &files.params,
    ))
}

/// One edit reference: the prepared image and its source display ratio, if known.
#[derive(Clone, Debug, PartialEq)]
pub struct EditReference {
    pub image: ImageRef,
    /// The original display ratio as a reduced `W:H`, `None` when unknown.
    pub aspect_ratio: Option<String>,
}

impl EditReference {
    /// A reference whose source ratio is the image's own pixel ratio, reduced.
    pub fn from_image(image: ImageRef) -> Self {
        let ratio = reduced_ratio(image.width, image.height);
        Self {
            image,
            aspect_ratio: ratio,
        }
    }
}

/// The edit workflow's target-ratio selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AspectSelection {
    /// Let the model choose (`adaptive`).
    Adaptive,
    /// An explicit positive-integer `W:H`, which the reply must preserve exactly.
    Explicit(String),
}

impl AspectSelection {
    fn as_str(&self) -> &str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Explicit(r) => r,
        }
    }
}

/// `w:h` reduced by their GCD; `None` for a zero side.
pub fn reduced_ratio(w: u32, h: u32) -> Option<String> {
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    if w == 0 || h == 0 {
        return None;
    }
    let g = gcd(w, h);
    Some(format!("{}:{}", w / g, h / g))
}

/// Whether `s` is a positive-integer `W:H`.
pub fn is_ratio(s: &str) -> bool {
    let Some((w, h)) = s.split_once(':') else {
        return false;
    };
    let positive = |p: &str| {
        !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) && p.parse::<u64>().unwrap_or(0) > 0
    };
    positive(w) && positive(h)
}

/// The edit expansion request: `instruction` refers to the references as `<Picture N>` in order.
pub fn edit_request(
    files: &PromptFiles,
    instruction: &str,
    references: &[EditReference],
    aspect: &AspectSelection,
) -> Result<TextLlmRequest, String> {
    if instruction.trim().is_empty() {
        return Err("edit instruction must be a non-empty string".into());
    }
    if references.is_empty() || references.len() > MAX_REFERENCES {
        return Err(format!(
            "edit expansion takes 1..={MAX_REFERENCES} references, got {}",
            references.len()
        ));
    }
    if let AspectSelection::Explicit(r) = aspect {
        if !is_ratio(r) {
            return Err(format!(
                "target aspect ratio {r:?} is not a positive-integer W:H"
            ));
        }
    }
    let count = references.len();
    let mut content = vec![
        Content::text(files.edit_task.clone()),
        Content::text(format!(
            "Reference Image Count:\n{count} input image(s), labeled <Picture 1> through \
             <Picture {count}>."
        )),
    ];
    for (i, reference) in references.iter().enumerate() {
        let metadata = match &reference.aspect_ratio {
            Some(r) => format!("aspect_ratio {r}"),
            None => "aspect_ratio unavailable".to_string(),
        };
        content.push(Content::text(format!(
            "Reference Media:\n- <Picture {}> [{metadata}]",
            i + 1
        )));
        content.push(Content::Image(reference.image.clone()));
    }
    let quoted = serde_json::to_string(instruction).map_err(|e| e.to_string())?;
    content.push(Content::text(format!(
        "\nUser Editing Instruction (JSON string):\n{quoted}\n\nUser Aspect Ratio Selection:\n{}\n\n\
         Apply the instructions above to the supplied inputs and reference images. Output only \
         the required two-field JSON object.",
        aspect.as_str()
    )));
    let message = Message {
        role: Role::User,
        content,
        thinking: None,
        tool_calls: Vec::new(),
    };
    Ok(request(vec![message], &files.params))
}

/// A parsed expansion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expansion {
    /// The expanded prompt (trimmed, non-empty).
    pub expanded_caption: String,
    /// The proposed `W:H`: the T2I `aspect_ratio` or the edit `target_aspect_ratio`.
    pub aspect_ratio: String,
}

fn exact_object(text: &str, keys: [&str; 2]) -> Result<Map<String, Value>, String> {
    let value: Value = serde_json::from_str(text.trim())
        .map_err(|e| format!("expansion reply is not JSON: {e}"))?;
    let Value::Object(map) = value else {
        return Err("expansion reply is not a JSON object".into());
    };
    let mut got: Vec<&str> = map.keys().map(String::as_str).collect();
    got.sort_unstable();
    let mut want = keys.to_vec();
    want.sort_unstable();
    if got != want {
        return Err(format!(
            "expansion reply must have exactly {want:?}, got {got:?}"
        ));
    }
    Ok(map)
}

fn caption(map: &Map<String, Value>) -> Result<String, String> {
    match map
        .get("expanded_caption")
        .and_then(Value::as_str)
        .map(str::trim)
    {
        Some(c) if !c.is_empty() => Ok(c.to_string()),
        _ => Err("expanded_caption must be a non-empty string".into()),
    }
}

/// Parse a T2I answer (the reply text after the thinking block).
pub fn parse_t2i(answer: &str, allowed_ratios: &[String]) -> Result<Expansion, String> {
    let map = exact_object(answer, ["expanded_caption", "aspect_ratio"])?;
    let ratio = map
        .get("aspect_ratio")
        .and_then(Value::as_str)
        .ok_or("aspect_ratio must be a string")?;
    if !allowed_ratios.iter().any(|r| r == ratio) {
        return Err(format!("unsupported aspect ratio {ratio:?}"));
    }
    Ok(Expansion {
        expanded_caption: caption(&map)?,
        aspect_ratio: ratio.to_string(),
    })
}

/// Parse an edit answer; an explicit selection must come back unchanged.
pub fn parse_edit(answer: &str, aspect: &AspectSelection) -> Result<Expansion, String> {
    let map = exact_object(answer, ["expanded_caption", "target_aspect_ratio"])?;
    let ratio = map
        .get("target_aspect_ratio")
        .and_then(Value::as_str)
        .ok_or("target_aspect_ratio must be a string")?;
    if !is_ratio(ratio) {
        return Err(format!(
            "target_aspect_ratio {ratio:?} is not a positive-integer W:H"
        ));
    }
    if let AspectSelection::Explicit(selected) = aspect {
        if selected != ratio {
            return Err(format!(
                "target_aspect_ratio {ratio:?} does not preserve the selected {selected:?}"
            ));
        }
    }
    Ok(Expansion {
        expanded_caption: caption(&map)?,
        aspect_ratio: ratio.to_string(),
    })
}

/// The fixed quality set every converted tier is compared on against the NVFP4 original
/// (AC2), and the comparison metric.
///
/// The prompts are this repository's own text (never MiniMax's). Edit references are drawn
/// procedurally, so the set needs no image files.
///
/// **Metric.** Each tier runs the set with the **published** sampling recipe (temperature 1.0,
/// top-p 0.95, top-k 20, 4096 new tokens), seeded per case ([`quality::QUALITY_SEED`]), as the
/// NVFP4 reference does. Greedy decoding is not used: the card publishes a sampling recipe only,
/// the Qwen3.8 base card lists the same values for thinking mode, and a greedy thinking trace
/// can run out the 4096-token budget before it answers (SW-7099 run 42885269431, `edit-0`). Per
/// case it records whether the reply parses, whether its aspect ratio equals the reference's, and
/// the chrF score ([`quality::chrf`], character 6-gram F-score, β = 2) of its `expanded_caption`
/// against the reference's. chrF is used because the caption is the only thing the image model
/// consumes, it tolerates the rewordings two sampled runs make, and it needs no second model to
/// judge. A token-by-token comparison is not used: a 27B reasoning model's sampled thinking
/// diverges within a few hundred tokens under any quantization, which says nothing about the
/// caption it settles on.
pub mod quality {
    use super::{reduced_ratio, AspectSelection, EditReference};
    use crate::message::ImageRef;

    /// Short T2I prompts.
    pub const T2I_PROMPTS: [&str; 6] = [
        "a red bicycle leaning against a brick wall",
        "a lighthouse on a rocky coast at dusk, waves breaking",
        "portrait of an elderly fisherman mending a net, natural light",
        "a cozy reading nook with a green armchair and a tall bookshelf",
        "a poster that says SUMMER MARKET in bold letters above a basket of fruit",
        "two foxes playing in fresh snow under pine trees",
    ];

    /// One edit case: the instruction, the procedural references and the ratio selection.
    pub struct EditCase {
        pub instruction: &'static str,
        pub references: Vec<EditReference>,
        pub aspect: AspectSelection,
    }

    /// A `w×h` image filled with `background` and a centred `shape` rectangle of `fill`.
    pub fn procedural_image(w: u32, h: u32, background: [u8; 3], fill: [u8; 3]) -> ImageRef {
        let mut pixels = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let inside = x >= w / 4 && x < 3 * w / 4 && y >= h / 4 && y < 3 * h / 4;
                pixels.extend_from_slice(if inside { &fill } else { &background });
            }
        }
        ImageRef::new(w, h, pixels).expect("procedural image is sized exactly")
    }

    fn reference(w: u32, h: u32, background: [u8; 3], fill: [u8; 3]) -> EditReference {
        let image = procedural_image(w, h, background, fill);
        EditReference {
            aspect_ratio: reduced_ratio(w, h),
            image,
        }
    }

    /// The edit cases: one single-reference and one two-reference edit.
    pub fn edit_cases() -> Vec<EditCase> {
        vec![
            EditCase {
                instruction: "Turn the red square in <Picture 1> into a glossy red apple on a \
                              wooden table.",
                references: vec![reference(384, 256, [240, 240, 240], [200, 30, 30])],
                aspect: AspectSelection::Adaptive,
            },
            EditCase {
                instruction: "Place the blue box from <Picture 2> next to the yellow box from \
                              <Picture 1> on a sandy beach.",
                references: vec![
                    reference(256, 256, [250, 250, 250], [230, 200, 20]),
                    reference(256, 320, [20, 20, 20], [30, 60, 210]),
                ],
                aspect: AspectSelection::Explicit("16:9".into()),
            },
        ]
    }

    /// Character n-gram F-score (chrF, Popović 2015) of `hypothesis` against `reference`:
    /// whitespace removed, n = 1..=6 averaged precision and recall, β = 2, scaled to `0..=100`.
    pub fn chrf(hypothesis: &str, reference: &str) -> f64 {
        use std::collections::HashMap;
        const MAX_N: usize = 6;
        const BETA2: f64 = 4.0;
        let strip = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<Vec<_>>();
        let (h, r) = (strip(hypothesis), strip(reference));
        fn grams(s: &[char], n: usize) -> HashMap<&[char], usize> {
            let mut m: HashMap<&[char], usize> = HashMap::new();
            if s.len() >= n {
                for w in s.windows(n) {
                    *m.entry(w).or_default() += 1;
                }
            }
            m
        }
        let (mut p_sum, mut r_sum, mut orders) = (0.0, 0.0, 0usize);
        for n in 1..=MAX_N {
            let (hg, rg) = (grams(&h, n), grams(&r, n));
            let (h_total, r_total): (usize, usize) = (hg.values().sum(), rg.values().sum());
            if h_total == 0 || r_total == 0 {
                continue;
            }
            let matched: usize = hg
                .iter()
                .map(|(g, c)| (*c).min(rg.get(g).copied().unwrap_or(0)))
                .sum();
            p_sum += matched as f64 / h_total as f64;
            r_sum += matched as f64 / r_total as f64;
            orders += 1;
        }
        if orders == 0 {
            return if h == r { 100.0 } else { 0.0 };
        }
        let (p, r) = (p_sum / orders as f64, r_sum / orders as f64);
        if p + r == 0.0 {
            return 0.0;
        }
        100.0 * (1.0 + BETA2) * p * r / (BETA2 * p + r)
    }

    /// One case's comparison against the NVFP4 reference.
    #[derive(Clone, Debug, PartialEq)]
    pub struct CaseScore {
        pub parsed: bool,
        pub ratio_matches: bool,
        pub chrf: f64,
    }

    /// Score one case: an unparsed reply scores 0 chrF and no ratio match.
    pub fn score(tier: Option<&super::Expansion>, reference: &super::Expansion) -> CaseScore {
        match tier {
            None => CaseScore {
                parsed: false,
                ratio_matches: false,
                chrf: 0.0,
            },
            Some(t) => CaseScore {
                parsed: true,
                ratio_matches: t.aspect_ratio == reference.aspect_ratio,
                chrf: chrf(&t.expanded_caption, &reference.expanded_caption),
            },
        }
    }

    /// The tier's stated quality: parse rate, ratio agreement and mean chrF over the set.
    #[derive(Clone, Debug, PartialEq)]
    pub struct TierQuality {
        pub cases: usize,
        pub parse_rate: f64,
        pub ratio_agreement: f64,
        pub mean_chrf: f64,
    }

    /// One case as a tier ran it.
    #[derive(Clone, Debug, PartialEq)]
    pub struct CaseRun {
        /// Stable case id (`t2i-N` / `edit-N`).
        pub id: String,
        /// The parsed expansion, or `None` when the reply broke the schema.
        pub expansion: Option<super::Expansion>,
        /// Why the reply was refused, when it was.
        pub error: Option<String>,
        /// Tokens generated (thinking and answer).
        pub generated_tokens: u32,
        /// Why generation stopped (`stop`, `length`, …), when the backend says.
        pub finish_reason: Option<String>,
        /// The last [`THINKING_TAIL_CHARS`] characters of the reasoning trace, kept as evidence
        /// for a reply that broke the schema (a looping or unfinished trace shows here).
        pub thinking_tail: Option<String>,
        /// The first [`ANSWER_HEAD_CHARS`] characters of the raw answer (the text after the
        /// thinking block), kept as evidence for a reply that broke the schema.
        pub answer_head: Option<String>,
    }

    /// Base seed of the quality set; case `i` of the run uses `QUALITY_SEED + i`, so a rerun on
    /// one backend and tier reproduces its samples.
    pub const QUALITY_SEED: u64 = 0x5EED_7099;

    /// How much of a failed reply's reasoning trace the evidence keeps.
    pub const THINKING_TAIL_CHARS: usize = 800;

    /// How much of a failed reply's raw answer the evidence keeps.
    pub const ANSWER_HEAD_CHARS: usize = 800;

    /// What every quality-set request constrains its answer to, as the evidence states it.
    pub const ANSWER_CONSTRAINT: &str =
        "json: well-formed JSON after </think> (upstream: XGrammar response_format); \
         schema checked by parse_t2i/parse_edit";

    /// Judge a finished reply: the parsed expansion, or why it is refused. A reply that stopped on
    /// the token budget is named as a truncation — in the reasoning trace (no answer text at all)
    /// or in the answer — never reported as the JSON error an empty or cut answer would produce.
    pub fn judge_reply(
        answer: &str,
        finish: Option<crate::FinishReason>,
        max_new_tokens: u32,
        parse: &dyn Fn(&str) -> Result<super::Expansion, String>,
    ) -> Result<super::Expansion, String> {
        match parse(answer) {
            Ok(expansion) => Ok(expansion),
            Err(_) if finish == Some(crate::FinishReason::Length) && answer.trim().is_empty() => {
                Err(format!(
                    "truncated: the {max_new_tokens}-token budget ran out inside the reasoning \
                     trace, before any answer"
                ))
            }
            Err(why) if finish == Some(crate::FinishReason::Length) => Err(format!(
                "truncated: the {max_new_tokens}-token budget ran out inside the answer ({why})"
            )),
            Err(why) => Err(why),
        }
    }

    /// Run the whole set on `llm` with the published recipe, seeded per case
    /// ([`QUALITY_SEED`]), as both the NVFP4 reference and each tier do. A generation error fails
    /// the run; a reply that breaks the schema (truncations included, [`judge_reply`]) is
    /// recorded as unparsed.
    pub fn run_set(
        llm: &dyn crate::TextLlm,
        files: &super::PromptFiles,
    ) -> crate::Result<Vec<CaseRun>> {
        let mut runs = Vec::new();
        let ratios = files.params().aspect_ratios.clone();
        let mut run = |id: String,
                       mut req: crate::TextLlmRequest,
                       parse: &dyn Fn(&str) -> Result<super::Expansion, String>|
         -> crate::Result<()> {
            req.seed = Some(QUALITY_SEED + runs.len() as u64);
            let out = llm.complete(&req)?;
            let judged = judge_reply(&out.text, out.finish_reason, req.max_new_tokens, parse);
            let (expansion, error) = match judged {
                Ok(e) => (Some(e), None),
                Err(e) => (None, Some(e)),
            };
            let thinking_tail = match (&error, &out.thinking) {
                (Some(_), Some(t)) => {
                    let skip = t.chars().count().saturating_sub(THINKING_TAIL_CHARS);
                    Some(t.chars().skip(skip).collect())
                }
                _ => None,
            };
            let answer_head = error
                .as_ref()
                .map(|_| out.text.chars().take(ANSWER_HEAD_CHARS).collect());
            runs.push(CaseRun {
                id,
                expansion,
                error,
                generated_tokens: out.usage.generated_tokens,
                finish_reason: out.finish_reason.map(|f| format!("{f:?}").to_lowercase()),
                thinking_tail,
                answer_head,
            });
            Ok(())
        };
        for (i, prompt) in T2I_PROMPTS.iter().enumerate() {
            let req = super::t2i_request(files, prompt).map_err(crate::Error::InvalidRequest)?;
            run(format!("t2i-{i}"), req, &|t| super::parse_t2i(t, &ratios))?;
        }
        for (i, case) in edit_cases().iter().enumerate() {
            let req = super::edit_request(files, case.instruction, &case.references, &case.aspect)
                .map_err(crate::Error::InvalidRequest)?;
            let aspect = case.aspect.clone();
            run(format!("edit-{i}"), req, &|t| super::parse_edit(t, &aspect))?;
        }
        Ok(runs)
    }

    /// A run as the JSON document the evidence keeps (and the reference file other tiers read).
    pub fn runs_to_json(label: &str, runs: &[CaseRun]) -> serde_json::Value {
        let cases: Vec<serde_json::Value> = runs
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "expanded_caption": r.expansion.as_ref().map(|e| &e.expanded_caption),
                    "aspect_ratio": r.expansion.as_ref().map(|e| &e.aspect_ratio),
                    "error": r.error,
                    "generated_tokens": r.generated_tokens,
                    "finish_reason": r.finish_reason,
                    "thinking_tail": r.thinking_tail,
                    "answer_head": r.answer_head,
                })
            })
            .collect();
        serde_json::json!({
            "label": label,
            "revision": super::REVISION,
            "answer_constraint": ANSWER_CONSTRAINT,
            "cases": cases,
        })
    }

    /// Read the reference expansions back out of [`runs_to_json`]'s document, by case id.
    pub fn reference_from_json(
        doc: &serde_json::Value,
    ) -> Result<std::collections::BTreeMap<String, super::Expansion>, String> {
        let cases = doc
            .get("cases")
            .and_then(serde_json::Value::as_array)
            .ok_or("reference document has no cases")?;
        let mut out = std::collections::BTreeMap::new();
        for case in cases {
            let id = case.get("id").and_then(serde_json::Value::as_str);
            let caption = case
                .get("expanded_caption")
                .and_then(serde_json::Value::as_str);
            let ratio = case.get("aspect_ratio").and_then(serde_json::Value::as_str);
            match (id, caption, ratio) {
                (Some(id), Some(c), Some(r)) => {
                    out.insert(
                        id.to_string(),
                        super::Expansion {
                            expanded_caption: c.to_string(),
                            aspect_ratio: r.to_string(),
                        },
                    );
                }
                _ => return Err(format!("reference case {case} did not parse")),
            }
        }
        Ok(out)
    }

    /// Score `runs` against the reference: per case and summarized. A case the reference lacks
    /// is an error — the two runs did not cover the same set.
    pub fn compare(
        runs: &[CaseRun],
        reference: &std::collections::BTreeMap<String, super::Expansion>,
    ) -> Result<(Vec<(String, CaseScore)>, TierQuality), String> {
        let mut scores = Vec::new();
        for run in runs {
            let r = reference
                .get(&run.id)
                .ok_or_else(|| format!("reference has no case {}", run.id))?;
            scores.push((run.id.clone(), score(run.expansion.as_ref(), r)));
        }
        let summary = summarize(&scores.iter().map(|(_, s)| s.clone()).collect::<Vec<_>>());
        Ok((scores, summary))
    }

    /// Aggregate per-case scores.
    pub fn summarize(scores: &[CaseScore]) -> TierQuality {
        let n = scores.len().max(1) as f64;
        TierQuality {
            cases: scores.len(),
            parse_rate: scores.iter().filter(|s| s.parsed).count() as f64 / n,
            ratio_agreement: scores.iter().filter(|s| s.ratio_matches).count() as f64 / n,
            mean_chrf: scores.iter().map(|s| s.chrf).sum::<f64>() / n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic prompt tree (no MiniMax text) plus the digest of its manifest.
    fn synthetic_tree() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let files = [
            (T2I_SYSTEM, "SYSTEM RULES\n\n".to_string()),
            (T2I_TASK, "Expand {{carefully}}: {}".to_string()),
            (EDIT_TASK, "EDIT TASK".to_string()),
            (
                T2I_PARAMS,
                serde_json::json!({
                    "reasoning_effort": "xhigh", "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                    "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true},
                    "response_format": {"json_schema": {"schema": {"properties": {
                        "aspect_ratio": {"enum": ["1:1", "16:9"]}}}}},
                    "max_tokens": 4096
                })
                .to_string(),
            ),
        ];
        let mut manifest = serde_json::json!({"t2i": {"files": {}}, "edit": {"files": {}}});
        for (rel, text) in &files {
            let path = dir.path().join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, text).unwrap();
            let mode = if *rel == EDIT_TASK { "edit" } else { "t2i" };
            manifest[mode]["files"][*rel] =
                serde_json::json!({"sha256": sha256_hex(text.as_bytes())});
        }
        let manifest_text = manifest.to_string();
        std::fs::write(dir.path().join(MANIFEST), &manifest_text).unwrap();
        let digest = sha256_hex(manifest_text.as_bytes());
        (dir, digest)
    }

    #[test]
    fn verified_prompts_build_the_published_t2i_request() {
        let (dir, digest) = synthetic_tree();
        let files = PromptFiles::load_with_manifest_digest(dir.path(), &digest).unwrap();
        let req = t2i_request(&files, "a cat").unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0], Message::text(Role::System, "SYSTEM RULES"));
        assert_eq!(
            req.messages[1],
            Message::text(Role::User, "Expand {carefully}: a cat")
        );
        assert_eq!(req.max_new_tokens, 4096);
        assert_eq!((req.sampling.temperature, req.sampling.top_k), (1.0, 20));
        assert_eq!(req.thinking, ThinkingMode::Enabled);
        assert_eq!(req.reasoning_effort, Some(ReasoningEffort::XHigh));
        assert_eq!(req.preserve_thinking, Some(true));
        assert_eq!(req.constraint, Some(Constraint::Json));
        assert_eq!(files.params().aspect_ratios, ["1:1", "16:9"]);
    }

    #[test]
    fn a_manifest_or_prompt_that_drifted_is_refused() {
        let (dir, digest) = synthetic_tree();
        let wrong = PromptFiles::load_with_manifest_digest(dir.path(), &"0".repeat(64));
        assert!(wrong.unwrap_err().contains(MANIFEST));
        std::fs::write(dir.path().join(T2I_TASK), "tampered {}").unwrap();
        let err = PromptFiles::load_with_manifest_digest(dir.path(), &digest).unwrap_err();
        assert!(err.contains(T2I_TASK), "{err}");
    }

    #[test]
    fn the_pinned_manifest_digest_refuses_a_synthetic_tree() {
        let (dir, _) = synthetic_tree();
        assert!(PromptFiles::load_verified(dir.path()).is_err());
    }

    #[test]
    fn the_edit_message_interleaves_labels_and_images_in_order() {
        let (dir, digest) = synthetic_tree();
        let files = PromptFiles::load_with_manifest_digest(dir.path(), &digest).unwrap();
        let cases = quality::edit_cases();
        let case = &cases[1];
        let req = edit_request(&files, case.instruction, &case.references, &case.aspect).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.constraint, Some(Constraint::Json));
        let parts = &req.messages[0].content;
        assert_eq!(parts.len(), 2 + 2 * 2 + 1);
        assert_eq!(parts[0], Content::text("EDIT TASK"));
        assert_eq!(
            parts[1],
            Content::text(
                "Reference Image Count:\n2 input image(s), labeled <Picture 1> through <Picture 2>."
            )
        );
        assert_eq!(
            parts[2],
            Content::text("Reference Media:\n- <Picture 1> [aspect_ratio 1:1]")
        );
        assert!(parts[3].is_image());
        assert_eq!(
            parts[4],
            Content::text("Reference Media:\n- <Picture 2> [aspect_ratio 4:5]")
        );
        let Content::Text(last) = &parts[6] else {
            panic!("the instruction part is text")
        };
        assert!(last.contains("\"Place the blue box"), "{last}");
        assert!(
            last.contains("User Aspect Ratio Selection:\n16:9\n"),
            "{last}"
        );
    }

    #[test]
    fn edit_requests_refuse_bad_inputs() {
        let (dir, digest) = synthetic_tree();
        let files = PromptFiles::load_with_manifest_digest(dir.path(), &digest).unwrap();
        let one = quality::edit_cases().remove(0).references;
        assert!(edit_request(&files, "x", &[], &AspectSelection::Adaptive).is_err());
        let many = vec![one[0].clone(); MAX_REFERENCES + 1];
        assert!(edit_request(&files, "x", &many, &AspectSelection::Adaptive).is_err());
        let bad = AspectSelection::Explicit("wide".into());
        assert!(edit_request(&files, "x", &one, &bad).is_err());
    }

    #[test]
    fn python_format_matches_str_format() {
        assert_eq!(python_format_one("a {} b", "X").unwrap(), "a X b");
        assert_eq!(python_format_one("{0}{{}}", "X").unwrap(), "X{}");
        assert!(python_format_one("{name}", "X").is_err());
        assert!(python_format_one("}", "X").is_err());
    }

    #[test]
    fn replies_must_match_the_schema_exactly() {
        let ratios = vec!["1:1".to_string(), "3:2".to_string()];
        let ok = parse_t2i(
            r#"{"expanded_caption":" A cat. ","aspect_ratio":"3:2"}"#,
            &ratios,
        );
        assert_eq!(
            ok.unwrap(),
            Expansion {
                expanded_caption: "A cat.".into(),
                aspect_ratio: "3:2".into()
            }
        );
        for bad in [
            r#"{"expanded_caption":"A","aspect_ratio":"7:5"}"#,
            r#"{"expanded_caption":"","aspect_ratio":"1:1"}"#,
            r#"{"expanded_caption":"A","aspect_ratio":"1:1","x":1}"#,
            "not json",
        ] {
            assert!(parse_t2i(bad, &ratios).is_err(), "{bad}");
        }
        let edit = r#"{"expanded_caption":"B","target_aspect_ratio":"16:9"}"#;
        assert!(parse_edit(edit, &AspectSelection::Adaptive).is_ok());
        assert!(parse_edit(edit, &AspectSelection::Explicit("16:9".into())).is_ok());
        assert!(parse_edit(edit, &AspectSelection::Explicit("1:1".into())).is_err());
        let zero = r#"{"expanded_caption":"B","target_aspect_ratio":"0:9"}"#;
        assert!(parse_edit(zero, &AspectSelection::Adaptive).is_err());
    }

    #[test]
    fn ratios_reduce_and_validate() {
        assert_eq!(reduced_ratio(1920, 1080).as_deref(), Some("16:9"));
        assert_eq!(reduced_ratio(0, 5), None);
        assert!(is_ratio("21:9") && !is_ratio("1.5:1") && !is_ratio("0:1") && !is_ratio("16"));
    }

    #[test]
    fn chrf_is_100_for_identity_and_falls_with_divergence() {
        use quality::chrf;
        let a = "A red bicycle leans against a weathered brick fence.";
        assert!((chrf(a, a) - 100.0).abs() < 1e-9);
        let near = chrf("A red bicycle leans on an old brick fence.", a);
        let far = chrf("Two foxes play in fresh snow.", a);
        assert!(near > far && near < 100.0 && far >= 0.0, "{near} {far}");
        assert_eq!(chrf("", a), 0.0);
    }

    #[test]
    fn quality_summary_counts_unparsed_cases_as_zero() {
        use quality::{score, summarize};
        let reference = Expansion {
            expanded_caption: "A lighthouse at dusk.".into(),
            aspect_ratio: "3:2".into(),
        };
        let scores = [score(Some(&reference), &reference), score(None, &reference)];
        let q = summarize(&scores);
        assert_eq!((q.cases, q.parse_rate, q.ratio_agreement), (2, 0.5, 0.5));
        assert!((q.mean_chrf - 50.0).abs() < 1e-9);
    }

    #[test]
    fn a_reference_document_round_trips_and_scores_a_tier() {
        use quality::{compare, reference_from_json, runs_to_json, CaseRun};
        let e = |c: &str, r: &str| Expansion {
            expanded_caption: c.into(),
            aspect_ratio: r.into(),
        };
        let reference = vec![
            CaseRun {
                id: "t2i-0".into(),
                expansion: Some(e("A red bicycle.", "3:2")),
                error: None,
                generated_tokens: 9,
                finish_reason: Some("stop".into()),
                thinking_tail: None,
                answer_head: None,
            },
            CaseRun {
                id: "edit-0".into(),
                expansion: Some(e("An apple.", "16:9")),
                error: None,
                generated_tokens: 7,
                finish_reason: Some("stop".into()),
                thinking_tail: None,
                answer_head: None,
            },
        ];
        let doc = runs_to_json("nvfp4", &reference);
        let back = reference_from_json(&doc).unwrap();
        assert_eq!(back["t2i-0"], e("A red bicycle.", "3:2"));
        let tier = vec![
            CaseRun {
                expansion: Some(e("A red bicycle.", "1:1")),
                ..reference[0].clone()
            },
            CaseRun {
                expansion: None,
                error: Some("not JSON".into()),
                ..reference[1].clone()
            },
        ];
        let (scores, summary) = compare(&tier, &back).unwrap();
        assert_eq!(scores.len(), 2);
        assert_eq!((summary.parse_rate, summary.ratio_agreement), (0.5, 0.0));
        assert!((summary.mean_chrf - 50.0).abs() < 1e-9);
        let missing = vec![CaseRun {
            id: "t2i-9".into(),
            ..reference[0].clone()
        }];
        assert!(compare(&missing, &back).is_err());
    }

    #[test]
    fn a_reply_cut_by_the_budget_is_a_truncation_not_a_json_error() {
        use crate::FinishReason;
        use quality::judge_reply;
        let ratios = vec!["1:1".to_string()];
        let parse = |t: &str| parse_t2i(t, &ratios);
        let thinking_only = judge_reply("", Some(FinishReason::Length), 4096, &parse).unwrap_err();
        assert!(
            thinking_only.starts_with("truncated:") && thinking_only.contains("reasoning trace"),
            "{thinking_only}"
        );
        let cut = judge_reply(
            r#"{"expanded_caption":"A ca"#,
            Some(FinishReason::Length),
            4096,
            &parse,
        )
        .unwrap_err();
        assert!(
            cut.starts_with("truncated:") && cut.contains("inside the answer"),
            "{cut}"
        );
        let stopped = judge_reply("", Some(FinishReason::Stop), 4096, &parse).unwrap_err();
        assert!(stopped.contains("not JSON"), "{stopped}");
        let ok = r#"{"expanded_caption":"A cat.","aspect_ratio":"1:1"}"#;
        assert!(judge_reply(ok, Some(FinishReason::Length), 4096, &parse).is_ok());
    }

    #[test]
    fn every_quality_request_is_json_constrained_and_a_refused_answer_is_kept() {
        use crate::output::{StreamEvent, TextLlmOutput};
        use crate::{TextLlm, TextLlmDescriptor};
        // Answers every request with prose, as an unconstrained run did (run 42954971396).
        struct Prose(std::cell::RefCell<Vec<Option<Constraint>>>);
        impl TextLlm for Prose {
            fn descriptor(&self) -> &TextLlmDescriptor {
                unreachable!("run_set never reads the descriptor")
            }
            fn validate(&self, _: &TextLlmRequest) -> crate::Result<()> {
                Ok(())
            }
            fn generate(
                &self,
                req: &TextLlmRequest,
                _: &mut dyn FnMut(StreamEvent),
            ) -> crate::Result<TextLlmOutput> {
                self.0.borrow_mut().push(req.constraint);
                Ok(TextLlmOutput {
                    text: "Here is the caption: a red bicycle.".into(),
                    thinking: Some("plan".into()),
                    finish_reason: Some(crate::FinishReason::Stop),
                    ..Default::default()
                })
            }
        }
        let (dir, digest) = synthetic_tree();
        let files = PromptFiles::load_with_manifest_digest(dir.path(), &digest).unwrap();
        let llm = Prose(Default::default());
        let runs = quality::run_set(&llm, &files).unwrap();
        let seen = llm.0.borrow();
        assert_eq!(seen.len(), runs.len());
        assert!(
            seen.iter().all(|c| *c == Some(Constraint::Json)),
            "{seen:?}"
        );
        for run in &runs {
            assert!(run.expansion.is_none());
            assert!(
                run.error.as_deref().unwrap().contains("not JSON"),
                "{run:?}"
            );
            assert_eq!(
                run.answer_head.as_deref(),
                Some("Here is the caption: a red bicycle.")
            );
            assert_eq!(run.thinking_tail.as_deref(), Some("plan"));
        }
        let doc = quality::runs_to_json("candle-bf16", &runs);
        assert_eq!(doc["answer_constraint"], quality::ANSWER_CONSTRAINT);
        assert_eq!(
            doc["cases"][0]["answer_head"],
            "Here is the caption: a red bicycle."
        );
    }

    #[test]
    fn procedural_references_carry_their_reduced_ratio() {
        let cases = quality::edit_cases();
        assert_eq!(cases[0].references[0].aspect_ratio.as_deref(), Some("3:2"));
        let r = EditReference::from_image(quality::procedural_image(64, 48, [0; 3], [9; 3]));
        assert_eq!(r.aspect_ratio.as_deref(), Some("4:3"));
    }
}
