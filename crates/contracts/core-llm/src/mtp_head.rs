//! Companion multi-token-prediction heads (epic sc-24432, story sc-24444): the backend-neutral
//! contract a standalone Qwen3.8 MTP proposal head is checked against before a backend builds it.
//!
//! A companion head (e.g. `EigenLabs/Qwen3.8-27B-MTP-4bit`) is a directory holding a
//! `config.json` with `model_type` [`COMPANION_MTP_MODEL_TYPE`] and the predictor layer's
//! safetensors — no embeddings and no LM head: it borrows the target's. Both backends read the
//! head's config through [`read_companion_mtp_config`], project it and the target's config onto
//! [`CompanionMtpGeometry`] and refuse the head when [`CompanionMtpGeometry::mismatches`] is
//! non-empty, then check every stored tensor against
//! [`matrices`](CompanionMtpGeometry::matrices) / [`norms`](CompanionMtpGeometry::norms) — one
//! check, so the two backends accept and refuse exactly the same heads (E8).

use std::path::Path;

use crate::error::{Error, Result};

/// The `model_type` of a standalone Qwen3.8 MTP proposal head: one predictor layer, no
/// embeddings, no LM head.
pub const COMPANION_MTP_MODEL_TYPE: &str = "qwen3_5_mtp";

/// Why a companion head is refused for a sparse-MoE target — the same words on both backends
/// (E8). A MoE target's *native* head runs (sc-24438); a companion head carries a dense predictor
/// FFN the MoE body has no path to pair with.
pub const COMPANION_MTP_MOE_REFUSAL: &str =
    "a MoE target has no companion MTP predictor path on this architecture";

/// How every companion-head load fallback ends (E2: the target still loaded).
pub const COMPANION_MTP_DROPPED: &str = "the model loaded without a companion head";

/// Why a companion head is refused for a target that carries its own native predictor.
pub const COMPANION_MTP_ALREADY_NATIVE: &str = "the target already carries its own MTP predictor";

/// Why a companion head is refused for a target outside the Qwen3.5/3.8 family, which has no
/// predictor path to attach it to (the family follows, named by the backend).
pub const COMPANION_MTP_FAMILY_REFUSAL: &str =
    "mtp_head: companion MTP heads attach to Qwen3.5/3.8-family (qwen3_5, Prism) targets only";

/// The load fallback for a companion head at `head` that did not attach, for `reason` (leading
/// with `mtp_head:`) — the one wording both backends push onto
/// [`LoadReport::fallbacks`](crate::LoadReport::fallbacks) (E2, E8).
pub fn companion_head_fallback(reason: &str, head: &Path) -> String {
    format!("{reason} (`{}`); {COMPANION_MTP_DROPPED}", head.display())
}

/// What a native (in-checkpoint) MTP head's configuration and tensors settle to at load (epic
/// sc-24432 E2) — [`native_mtp_plan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeMtp {
    /// The config declares no predictor layer and the snapshot stores no `mtp.*` tensor.
    Absent,
    /// Build the one-layer predictor. A tensor the build needs and the snapshot lacks — a
    /// **partial** `mtp.*` set — still fails the load: that is a corrupt or mis-assembled
    /// snapshot (integrity), not an optional accelerator that is simply absent.
    Build,
    /// Load the target plain: the head is configured but cannot run here. The reason leads with
    /// `mtp:` and belongs in [`LoadReport::fallbacks`](crate::LoadReport::fallbacks); MTP is not
    /// advertised.
    Fallback(String),
}

/// Settle a Qwen3.5/3.8 target's native MTP head from its config — `layers` predictor layers
/// (`mtp_num_hidden_layers`), `dedicated_embeddings` (`mtp_use_dedicated_embeddings`) — and
/// whether the snapshot stores any `mtp.*` tensor (`stores_mtp`). The one rule both backends
/// load by (E8):
///
/// * no layer declared and no `mtp.*` tensor: [`NativeMtp::Absent`];
/// * `mtp.*` tensors under a config that declares no layer: `Err` — a contradiction between the
///   config and the snapshot, refused rather than guessed at;
/// * a declared head the snapshot stores **no** tensor of (a text-only re-host that dropped it),
///   or a variant this runtime does not run (more than one predictor layer, dedicated MTP
///   embeddings): [`NativeMtp::Fallback`] — the model loads and decodes without it (E2: an
///   optional accelerator never fails a load);
/// * otherwise [`NativeMtp::Build`].
pub fn native_mtp_plan(
    layers: usize,
    dedicated_embeddings: bool,
    stores_mtp: bool,
) -> Result<NativeMtp> {
    let why = match (layers, stores_mtp) {
        (0, false) => return Ok(NativeMtp::Absent),
        (0, true) => {
            return Err(Error::Load(
                "qwen3_5 snapshot carries `mtp.*` tensors but its config disables MTP \
                 (mtp_num_hidden_layers 0)"
                    .into(),
            ))
        }
        (_, false) => format!(
            "the config declares {layers} MTP predictor layer(s) but the snapshot stores no \
             `mtp.*` tensor"
        ),
        (1, true) if !dedicated_embeddings => return Ok(NativeMtp::Build),
        (1, true) => "the head declares dedicated MTP embeddings; this runtime's predictor \
                      shares the target's"
            .into(),
        (_, true) => {
            format!("the head declares {layers} predictor layers; this runtime runs exactly one")
        }
    };
    Ok(NativeMtp::Fallback(format!(
        "mtp: {why}; the model loaded without its MTP head"
    )))
}

/// A companion head's safetensors payload under `dir` — the model-agnostic half of a backend's
/// companion-head pricing (E7, story sc-24444): `dir` must be a directory holding at least one
/// `.safetensors` byte. The backend adds what its build allocates beside the payload.
pub fn companion_head_payload_bytes(dir: &Path) -> Result<u64> {
    if !dir.is_dir() {
        return Err(Error::Load(format!(
            "companion MTP head `{}` is not a directory",
            dir.display()
        )));
    }
    let payload = crate::checkpoint_payload_bytes(dir)?;
    if payload == 0 {
        return Err(Error::Load(format!(
            "companion MTP head `{}` holds no .safetensors",
            dir.display()
        )));
    }
    Ok(payload)
}

/// Admit a companion head of `head_bytes` (the backend's pricing of `head`) on top of what the
/// load already admitted — the target and any admitted draft (E7) — in every allocation domain
/// `(name, admitted, available)` the backend loads into (MLX's one unified-memory domain; Candle's
/// host and, on CUDA, device), returning the head and its bytes. `Err` is the named load fallback
/// (E2, leading `mtp_head:`): the head could not be priced, or the admitted bytes plus the head's
/// exceed a domain's budget. One rule and one wording on both backends (E8).
pub fn admit_companion_head(
    head: &Path,
    head_bytes: Result<u64>,
    domains: impl IntoIterator<Item = (&'static str, u64, u64)>,
) -> std::result::Result<(&Path, u64), String> {
    let head_bytes = head_bytes.map_err(|e| format!("mtp_head: {e}; {COMPANION_MTP_DROPPED}"))?;
    for (domain, target_bytes, available) in domains {
        let total = target_bytes
            .checked_add(head_bytes)
            .ok_or_else(|| format!("mtp_head: load admission overflow; {COMPANION_MTP_DROPPED}"))?;
        crate::admit_request_memory(total, available).map_err(|e| {
            format!(
                "mtp_head: refused by load admission: the head's {head_bytes} bytes on top of \
                 the target's {target_bytes} exceed the {domain} budget ({e}); \
                 {COMPANION_MTP_DROPPED} (`{}`)",
                head.display()
            )
        })?;
    }
    Ok((head, head_bytes))
}

/// Read a companion head's `config.json` from `dir` and require `model_type`
/// [`COMPANION_MTP_MODEL_TYPE`]. The returned value is the backend's to parse (its Qwen3.5
/// config reader resolves `text_config` nesting and the partial-rotary width).
pub fn read_companion_mtp_config(dir: &Path) -> Result<serde_json::Value> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| Error::Load(format!("read {}: {e}", path.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Load(format!("parse {}: {e}", path.display())))?;
    let model_type = value.get("model_type").and_then(|v| v.as_str());
    if model_type != Some(COMPANION_MTP_MODEL_TYPE) {
        return Err(Error::Load(format!(
            "companion MTP head `model_type` must be `{COMPANION_MTP_MODEL_TYPE}`, got {}",
            model_type.unwrap_or("none")
        )));
    }
    Ok(value)
}

/// Everything the one-layer Qwen3.8 predictor computes with, as a head's or a target's config
/// declares it. The predictor runs inside the target's residual stream, RoPE and vocabulary, so
/// a head is usable only when its geometry equals the target's.
#[derive(Clone, Debug, PartialEq)]
pub struct CompanionMtpGeometry {
    pub hidden_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    /// The only guard against a head trained for another tokenizer.
    pub vocab_size: i32,
    /// The partial-rotary width (`head_dim · partial_rotary_factor`).
    pub rotary_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
    /// The head's own predictor-layer count (a target's is irrelevant: it may have none).
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    /// Whether the config declares a sparse-MoE FFN.
    pub moe: bool,
}

impl CompanionMtpGeometry {
    /// Refuse a head whose geometry disagrees with `target`'s, naming every
    /// [`mismatch`](Self::mismatches) — the message both backends return.
    pub fn check_against(&self, target: &Self) -> std::result::Result<(), String> {
        let mismatches = self.mismatches(target);
        if mismatches.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "companion MTP head geometry does not match the target: {}",
                mismatches.join("; ")
            ))
        }
    }

    /// Check every stored tensor a head of this geometry must carry under `prefix` (see
    /// [`companion_mtp_prefix`]) before anything is built: each [`matrix`](Self::matrices)'s
    /// logical `[out, in]` as the backend reads it (`matrix(key)`, `None` when missing or not a
    /// matrix — a quantized matrix's logical width, not its packed one) and each
    /// [`norm`](Self::norms) vector's shape (`vector(key)`). `Err` names every disagreement, in
    /// the one message both backends return (E8).
    pub fn check_tensors(
        &self,
        prefix: &str,
        matrix: impl Fn(&str) -> Option<[usize; 2]>,
        vector: impl Fn(&str) -> Option<Vec<usize>>,
    ) -> std::result::Result<(), String> {
        let mut wrong = Vec::new();
        for (name, expected) in self.matrices() {
            match matrix(&format!("{prefix}{name}.weight")) {
                Some(actual) if actual == expected => {}
                Some(actual) => wrong.push(format!(
                    "`{name}` is {actual:?}, the target needs {expected:?}"
                )),
                None => wrong.push(format!("`{name}` is missing or not a matrix")),
            }
        }
        for (name, expected) in self.norms() {
            match vector(&format!("{prefix}{name}.weight")) {
                Some(shape) if shape == [expected] => {}
                Some(shape) => wrong.push(format!(
                    "`{name}` is {shape:?}, the target needs [{expected}]"
                )),
                None => wrong.push(format!("`{name}` is missing")),
            }
        }
        if wrong.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "companion MTP head tensors do not match the target: {}",
                wrong.join("; ")
            ))
        }
    }

    /// Every disagreement between this head and `target`, each named with both values — not just
    /// the first. Empty when the head fits.
    pub fn mismatches(&self, target: &Self) -> Vec<String> {
        let mut out = Vec::new();
        for (name, head, target) in [
            ("hidden_size", self.hidden_size, target.hidden_size),
            (
                "num_attention_heads",
                self.num_attention_heads,
                target.num_attention_heads,
            ),
            (
                "num_key_value_heads",
                self.num_key_value_heads,
                target.num_key_value_heads,
            ),
            ("head_dim", self.head_dim, target.head_dim),
            (
                "intermediate_size",
                self.intermediate_size,
                target.intermediate_size,
            ),
            ("vocab_size", self.vocab_size, target.vocab_size),
            ("rotary_dim", self.rotary_dim, target.rotary_dim),
        ] {
            if head != target {
                out.push(format!("{name} {head} != target {target}"));
            }
        }
        if self.mtp_num_hidden_layers != 1 {
            out.push(format!(
                "mtp_num_hidden_layers {} (this runtime runs exactly one predictor layer)",
                self.mtp_num_hidden_layers
            ));
        }
        for (name, head, target) in [
            ("rms_norm_eps", self.rms_norm_eps, target.rms_norm_eps),
            ("rope_theta", self.rope_theta, target.rope_theta),
        ] {
            if head != target {
                out.push(format!("{name} {head} != target {target}"));
            }
        }
        if self.mrope_section != target.mrope_section {
            out.push(format!(
                "mrope_section {:?} != target {:?}",
                self.mrope_section, target.mrope_section
            ));
        }
        if self.mtp_use_dedicated_embeddings {
            out.push(
                "mtp_use_dedicated_embeddings is true (the head must share the target's)".into(),
            );
        }
        if self.moe {
            out.push("the head declares a MoE predictor".into());
        }
        out
    }

    /// The `[out, in]` of every projection a dense predictor layer stores, by bare tensor stem
    /// (a backend reads `{prefix}{stem}.weight`, plus its `.scales` / `.biases` when quantized).
    pub fn matrices(&self) -> [(&'static str, [usize; 2]); 8] {
        let d = |v: i32| v.max(0) as usize;
        let (h, heads, kv, hd, inter) = (
            d(self.hidden_size),
            d(self.num_attention_heads),
            d(self.num_key_value_heads),
            d(self.head_dim),
            d(self.intermediate_size),
        );
        [
            ("fc", [h, 2 * h]),
            // The gated attention's query projection carries the output gate too (2×).
            ("layers.0.self_attn.q_proj", [2 * heads * hd, h]),
            ("layers.0.self_attn.k_proj", [kv * hd, h]),
            ("layers.0.self_attn.v_proj", [kv * hd, h]),
            ("layers.0.self_attn.o_proj", [h, heads * hd]),
            ("layers.0.mlp.gate_proj", [inter, h]),
            ("layers.0.mlp.up_proj", [inter, h]),
            ("layers.0.mlp.down_proj", [h, inter]),
        ]
    }

    /// The width of every RMSNorm vector the predictor stores, by bare tensor stem. Their values
    /// follow the zero-centred Qwen3.8 checkpoint convention: the runtime applies `1 + w`.
    pub fn norms(&self) -> [(&'static str, usize); 7] {
        let (h, hd) = (
            self.hidden_size.max(0) as usize,
            self.head_dim.max(0) as usize,
        );
        [
            ("pre_fc_norm_embedding", h),
            ("pre_fc_norm_hidden", h),
            ("norm", h),
            ("layers.0.input_layernorm", h),
            ("layers.0.post_attention_layernorm", h),
            ("layers.0.self_attn.q_norm", hd),
            ("layers.0.self_attn.k_norm", hd),
        ]
    }
}

/// The tensor-name prefix of a head's stored tensors: bare (`fc.weight`, the published layout) or
/// `mtp.`-prefixed. `contains` answers whether the head stores a tensor name.
pub fn companion_mtp_prefix(contains: impl Fn(&str) -> bool) -> Result<&'static str> {
    match (contains("fc.weight"), contains("mtp.fc.weight")) {
        (true, false) => Ok(""),
        (false, true) => Ok("mtp."),
        (true, true) => Err(Error::Load(
            "companion MTP head has both bare and `mtp.`-prefixed tensors".into(),
        )),
        (false, false) => Err(Error::Load(
            "companion MTP head stores no `fc.weight`".into(),
        )),
    }
}

/// Refuse a companion head that stores tensors its predictor does not read (`unused`, any
/// order): every tensor in a head must be consumed, on both backends (E8).
pub fn check_companion_unused(mut unused: Vec<String>) -> std::result::Result<(), String> {
    if unused.is_empty() {
        return Ok(());
    }
    unused.sort_unstable();
    Err(format!(
        "companion MTP head carries tensors the predictor does not use: {}",
        unused.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> CompanionMtpGeometry {
        CompanionMtpGeometry {
            hidden_size: 128,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 64,
            intermediate_size: 256,
            vocab_size: 64,
            rotary_dim: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 1e7,
            mrope_section: [11, 11, 10],
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            moe: false,
        }
    }

    fn head() -> CompanionMtpGeometry {
        CompanionMtpGeometry {
            mtp_num_hidden_layers: 1,
            ..target()
        }
    }

    /// Every field the predictor computes with is checked and named with both values; a head
    /// matching the target (its own layer count aside) has no mismatch.
    #[test]
    fn every_geometry_field_is_checked_and_named() {
        assert!(head().mismatches(&target()).is_empty());
        type Edit = fn(&mut CompanionMtpGeometry);
        let cases: [(Edit, &str); 13] = [
            (|g| g.hidden_size = 256, "hidden_size 256 != target 128"),
            (
                |g| g.num_attention_heads = 4,
                "num_attention_heads 4 != target 2",
            ),
            (
                |g| g.num_key_value_heads = 2,
                "num_key_value_heads 2 != target 1",
            ),
            (|g| g.head_dim = 32, "head_dim 32 != target 64"),
            (
                |g| g.intermediate_size = 128,
                "intermediate_size 128 != target 256",
            ),
            (|g| g.vocab_size = 65, "vocab_size 65 != target 64"),
            (|g| g.rotary_dim = 64, "rotary_dim 64 != target 32"),
            (
                |g| g.rms_norm_eps = 1e-5,
                "rms_norm_eps 0.00001 != target 0.000001",
            ),
            (
                |g| g.rope_theta = 1e6,
                "rope_theta 1000000 != target 10000000",
            ),
            (
                |g| g.mrope_section = [8, 12, 12],
                "mrope_section [8, 12, 12] != target [11, 11, 10]",
            ),
            (|g| g.mtp_num_hidden_layers = 2, "mtp_num_hidden_layers 2"),
            (
                |g| g.mtp_use_dedicated_embeddings = true,
                "mtp_use_dedicated_embeddings is true",
            ),
            (|g| g.moe = true, "the head declares a MoE predictor"),
        ];
        for (mutate, named) in cases {
            let mut h = head();
            mutate(&mut h);
            let found = h.mismatches(&target());
            assert_eq!(found.len(), 1, "{found:?}");
            assert!(found[0].contains(named), "{found:?} lacks {named}");
        }
        // Several at once are all named.
        let mut h = head();
        h.vocab_size = 65;
        h.rope_theta = 1e6;
        assert_eq!(h.mismatches(&target()).len(), 2);
    }

    #[test]
    fn a_head_config_must_declare_the_companion_model_type() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path();
        std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5_mtp"}"#).unwrap();
        assert!(read_companion_mtp_config(dir).is_ok());
        std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5"}"#).unwrap();
        let err = read_companion_mtp_config(dir).unwrap_err().to_string();
        assert!(err.contains("must be `qwen3_5_mtp`, got qwen3_5"), "{err}");
        assert!(read_companion_mtp_config(&dir.join("absent")).is_err());
    }

    #[test]
    fn the_tensor_prefix_is_bare_or_mtp_never_both() {
        assert_eq!(companion_mtp_prefix(|k| k == "fc.weight").unwrap(), "");
        assert_eq!(
            companion_mtp_prefix(|k| k == "mtp.fc.weight").unwrap(),
            "mtp."
        );
        assert!(companion_mtp_prefix(|_| true).is_err());
        assert!(companion_mtp_prefix(|_| false).is_err());
    }

    /// E2: a configured native head the snapshot does not carry, or a variant this runtime does
    /// not run, settles to a named `mtp:` load fallback; a complete one builds; tensors under a
    /// config that disables MTP stay a refused contradiction.
    #[test]
    fn a_native_mtp_head_settles_to_build_absent_or_a_named_fallback() {
        assert_eq!(native_mtp_plan(0, false, false).unwrap(), NativeMtp::Absent);
        assert_eq!(native_mtp_plan(1, false, true).unwrap(), NativeMtp::Build);
        let err = native_mtp_plan(0, false, true).unwrap_err().to_string();
        assert!(err.contains("config disables MTP"), "{err}");
        for (layers, dedicated, stores, named) in [
            (1, false, false, "stores no `mtp.*` tensor"),
            (2, false, false, "stores no `mtp.*` tensor"),
            (2, false, true, "declares 2 predictor layers"),
            (1, true, true, "dedicated MTP embeddings"),
        ] {
            let NativeMtp::Fallback(why) = native_mtp_plan(layers, dedicated, stores).unwrap()
            else {
                panic!("({layers}, {dedicated}, {stores}) must fall back");
            };
            assert!(why.starts_with("mtp: "), "{why}");
            assert!(why.contains(named), "{why} lacks {named}");
            assert!(
                why.ends_with("the model loaded without its MTP head"),
                "{why}"
            );
        }
    }

    /// E7/E8: the companion head is admitted in every domain the backend names, and a refusal
    /// names the domain and leads with `mtp_head:`.
    #[test]
    fn companion_head_admission_names_the_refusing_domain() {
        let head = Path::new("/heads/h");
        assert_eq!(
            admit_companion_head(head, Ok(10), [("host", 50, 100), ("device", 80, 90)]),
            Ok((head, 10))
        );
        let refused = admit_companion_head(head, Ok(10), [("host", 50, 100), ("device", 85, 90)])
            .unwrap_err();
        assert!(
            refused.starts_with("mtp_head: refused by load admission")
                && refused.contains("device budget")
                && refused.contains(COMPANION_MTP_DROPPED),
            "{refused}"
        );
        let unpriced = admit_companion_head(
            head,
            Err(Error::Load("bad".into())),
            [("unified memory", 0, 1)],
        )
        .unwrap_err();
        assert_eq!(
            unpriced,
            format!(
                "mtp_head: {}; {COMPANION_MTP_DROPPED}",
                Error::Load("bad".into())
            )
        );
        let root = tempfile::tempdir().unwrap();
        let err = companion_head_payload_bytes(root.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("holds no .safetensors"), "{err}");
    }

    /// E8: the stored-tensor check names every wrong or missing tensor, and every unused one.
    #[test]
    fn the_tensor_check_names_every_wrong_or_missing_tensor() {
        let g = head();
        let matrices: std::collections::HashMap<String, [usize; 2]> = g
            .matrices()
            .into_iter()
            .map(|(name, shape)| (format!("mtp.{name}.weight"), shape))
            .collect();
        let norms: std::collections::HashMap<String, Vec<usize>> = g
            .norms()
            .into_iter()
            .map(|(name, width)| (format!("mtp.{name}.weight"), vec![width]))
            .collect();
        g.check_tensors(
            "mtp.",
            |k| matrices.get(k).copied(),
            |k| norms.get(k).cloned(),
        )
        .unwrap();
        let err = g
            .check_tensors(
                "mtp.",
                |k| {
                    (k != "mtp.fc.weight")
                        .then(|| matrices.get(k).copied())
                        .flatten()
                },
                |k| {
                    if k == "mtp.norm.weight" {
                        Some(vec![3])
                    } else {
                        norms.get(k).cloned()
                    }
                },
            )
            .unwrap_err();
        assert!(
            err.contains("`fc` is missing or not a matrix")
                && err.contains("`norm` is [3], the target needs [128]"),
            "{err}"
        );
        assert!(check_companion_unused(Vec::new()).is_ok());
        assert_eq!(
            check_companion_unused(vec!["b".into(), "a".into()]).unwrap_err(),
            "companion MTP head carries tensors the predictor does not use: a, b"
        );
        assert!(g.check_against(&target()).is_ok());
        let mut wide = head();
        wide.hidden_size = 256;
        assert!(wide
            .check_against(&target())
            .unwrap_err()
            .starts_with("companion MTP head geometry does not match the target: hidden_size"));
    }
}
