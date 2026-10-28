//! Third-party LyCORIS **LoHa** files: detection, parsing and install.
//!
//! Split out of `loader.rs` (sc-27427); every item is re-exported there, so
//! `mlx_gen::adapters::loader::…` paths are unchanged.

use super::*;

// ---- Third-party LyCORIS LoHa (sc-3643) ----------------------------------------------------------

/// Third-party LoHa factor suffixes (the two Hadamard low-rank pairs + optional tucker factors).
/// (gen-core.)
pub(super) use gen_core::weightsmeta::LOHA_TP_SUFFIXES;

/// `true` if any key is a LoHa factor (`*.hada_w…`) — how a third-party LyCORIS LoHa (kohya /
/// ai-toolkit / lycoris-lib) is recognized. Mutually exclusive with [`is_lokr_keys`] (`lokr_*`).
pub fn is_loha_keys(w: &Weights) -> bool {
    wmeta::keys_contain_loha(w.keys())
}

/// One module's third-party LoHa factors — two low-rank Hadamard pairs, optional tucker `t1`/`t2`,
/// and an optional per-module `.alpha` (rank/scale derived per module, like [`ThirdPartyLokr`]).
#[derive(Default)]
pub struct ThirdPartyLoha {
    pub(super) w1_a: Option<Array>,
    pub(super) w1_b: Option<Array>,
    pub(super) w2_a: Option<Array>,
    pub(super) w2_b: Option<Array>,
    pub(super) t1: Option<Array>,
    pub(super) t2: Option<Array>,
    pub(super) alpha: Option<f32>,
}

/// Exact selected third-party LoHa rank gate, shared by runtime and header facts. Other factors
/// are validated by the actual reconstruction operation; this gate must not restrict them anew.
pub fn thirdparty_loha_rank(w1_b: Option<&[i32]>) -> Result<Option<f32>> {
    let Some(shape) = w1_b else {
        return Ok(None);
    };
    if shape.len() < 2 {
        return Err(Error::Msg(format!(
            "loha adapter: hada_w1_b must be 2-D, got {}-D shape {:?}",
            shape.len(),
            shape
        )));
    }
    let rank = shape[0] as f32;
    if rank <= 0.0 {
        return Err(Error::Msg(
            "loha adapter: hada_w1_b has zero rank (zero leading dim)".into(),
        ));
    }
    Ok(Some(rank))
}

impl ThirdPartyLoha {
    /// rank (`lora_dim`) = `hada_w1_b.shape[0]` (lycoris stores `hada_w1_b` as `[lora_dim, …]` in
    /// both the tucker and non-tucker layouts).
    ///
    /// F-010: a 1-D/0-D factor panics on the shape index; a zero leading dim → `scale = alpha/0` =
    /// inf/NaN baked into the reconstructed delta. `Result` so both are rejected (F-002 LoRA twin).
    pub(super) fn rank(&self) -> Result<Option<f32>> {
        thirdparty_loha_rank(self.w1_b.as_ref().map(Array::shape))
    }

    /// LyCORIS `scale = alpha / lora_dim` (alpha defaulting to `lora_dim`). LoHa is always decomposed
    /// (no both-full case), so — unlike LoKr — there is no forced-1 branch.
    pub(super) fn scale(&self) -> Result<f32> {
        match self.rank()? {
            None => Ok(1.0),
            Some(r) => Ok(self.alpha.unwrap_or(r) / r),
        }
    }

    /// Reconstruct this module's LoHa `ΔW` (lycoris per-module scale baked in) at `out_dtype`. `pub`
    /// for the merge-path providers (sc-3671).
    pub fn delta(&self, base_shape: &[i32], out_dtype: Dtype) -> Result<Array> {
        let (w1_a, w1_b, w2_a, w2_b) = match (&self.w1_a, &self.w1_b, &self.w2_a, &self.w2_b) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return Err("LoHa: a hada_w1/w2 a/b factor is missing".into()),
        };
        reconstruct_loha_delta(
            self.scale()?,
            base_shape,
            w1_a,
            w1_b,
            w2_a,
            w2_b,
            self.t1.as_ref(),
            self.t2.as_ref(),
            out_dtype,
        )
    }
}

/// Group a third-party LoHa file's tensors by raw module key (the part before `.hada_*`/`.alpha`).
/// `pub` for the merge-path providers (sc-3671).
pub fn parse_loha_thirdparty(w: &Weights) -> Result<BTreeMap<String, ThirdPartyLoha>> {
    let mut groups: BTreeMap<String, ThirdPartyLoha> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        if let Some(raw) = key.strip_suffix(".alpha") {
            if let Some(a) = scalar_alpha(w.require(&key)?)? {
                groups.entry(raw.to_string()).or_default().alpha = Some(a);
            }
            continue;
        }
        for suffix in LOHA_TP_SUFFIXES {
            if let Some(raw) = key.strip_suffix(suffix) {
                let g = groups.entry(raw.to_string()).or_default();
                let t = w.require(&key)?.clone();
                match &suffix[1..] {
                    "hada_w1_a" => g.w1_a = Some(t),
                    "hada_w1_b" => g.w1_b = Some(t),
                    "hada_w2_a" => g.w2_a = Some(t),
                    "hada_w2_b" => g.w2_b = Some(t),
                    "hada_t1" => g.t1 = Some(t),
                    "hada_t2" => g.t2 = Some(t),
                    _ => {}
                }
                break;
            }
        }
    }
    Ok(groups)
}

/// Install a third-party LyCORIS **LoHa** file onto `host`. Reconstructs each module's Hadamard delta
/// and stacks it as an [`Adapter::Lokr`] residual (the reconstructed `ΔW` applies through the same
/// `scale · x·ΔWᵀ` forward path — no distinct adapter variant needed). Module-key resolution
/// (flattened-prefixed → dotted via `kohya_table`) and unmatched-path surfacing mirror
/// [`apply_lokr_thirdparty`].
///
/// The Hadamard product has **no deferred (unmaterialized) form** — unlike LoKr's Kronecker vec-trick —
/// so the factors builder below is unconditionally `None`: there is nothing to build. On a packed Q4/Q8
/// base, `install_lycoris_groups` therefore falls back to the materialized `[out,in]` delta. That is
/// CORRECT (the residual `scale · x·ΔWᵀ` never reads the base, so it rides over `quantized_matmul`
/// unchanged) but holds a dense tensor per target. Refusing the trade instead — sc-10051's call for
/// quantized Wan — would break every third-party LoHa on the DEFAULT tier of every MLX model, because
/// adapters install *after* `.quantize()` and the worker defaults `spec.quantize` to `Some(Q8)`.
/// Whether a packed base should refuse a multi-GB delta is sc-10678, not this path's call to make.
pub fn apply_loha_thirdparty(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
) -> Result<ApplyReport> {
    let table = kohya_table(&host.adaptable_paths());
    let groups = parse_loha_thirdparty(w)?;
    // Same residual dtype as the LoKr paths (the host's, bf16 by default — PARITY-BF16, sc-2609).
    let dtype = host.lycoris_dtype();
    let groups = groups.iter().map(|(raw, g)| {
        (
            raw.as_str(),
            move |base: &[i32]| g.delta(base, dtype),
            // Hadamard has no Kronecker structure to defer — never deferrable, on any base.
            |_strength: f32, _base: &[i32]| -> Result<Option<LokrFactors>> { Ok(None) },
        )
    });
    install_lycoris_groups(
        host,
        groups,
        scale,
        LycorisKeyResolution::Thirdparty(&table),
    )
}
