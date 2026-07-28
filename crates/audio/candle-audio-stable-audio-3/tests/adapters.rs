//! Weight-free gates for the Stable Audio 3 adapter family (sc-14550).
//!
//! # No real adapter exists, and this suite does not pretend otherwise
//!
//! There is **no Stable Audio 3 adapter artifact on this machine**, of any of the eight types, and
//! no published community one — `sc-15347` tracks obtaining them. Every adapter this file uses is
//! *synthesized here*, written to disk in the format [`candle_audio_stable_audio_3::adapters`]
//! declares, and read back through the shipped loader. That exercises the real load → validate →
//! plan → fold path with no weights, which is enough to gate the math, the ordering, the filters,
//! the formats and the refusals. It is **not** a real-weight claim and none is made.
//!
//! # The correctly-signed probes
//!
//! sc-14548's blocker was a set of floors that all got *better* when the feature broke. The trap
//! here is the obvious one: "output with an adapter differs from the base". A **misapplied**
//! adapter differs too — usually more. So every gate below collapses to exact equality or exact
//! inequality rather than to a threshold:
//!
//! * `scale == 0.0` ⇒ the adapted weight is **bit-identical** to the base. Exactly zero difference;
//!   no threshold can hide a fold that ran.
//! * adapter A vs adapter B on the same base ⇒ **differ**, byte-wise.
//! * `[A, B]` vs `[B, A]` for a non-commuting type ⇒ **differ**, byte-wise; for classic LoRA ⇒
//!   **identical**, byte-wise. Both directions asserted, so the ordering machinery is proven to be
//!   measuring order rather than noise.
//! * DoRA rows ⇒ every row norm equals `|magnitude_r[i]|` **exactly** (to F32 rounding). A wrong
//!   axis, a missing normalization or a dropped magnitude each break it outright.
//! * `-xs` on a diagonal base with a known spectrum ⇒ the delta equals the core exactly, because
//!   `U` and `V` are the identity up to the sign rule. Sign drift is visible as a sign, not a
//!   distance.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use candle_audio_stable_audio_3::adapters::{
    adaptable_targets, apply_adapter_ops, expand_bracket_ranges, is_adaptable_target, load_adapter,
    plan_for, resolve_adapter_type, validate_spec_shape, AdapterBackend, AdapterPlan,
    AdapterSource, AdapterType, LoadedAdapter, ADAPTABLE_PREFIXES,
};
use candle_audio_stable_audio_3::model;
use candle_audio_stable_audio_3::weights::safetensors_shapes;

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::gen_core::{AdapterKind, AdapterSpec};

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

/// A representative slice of the **real** `stable_audio_3_small_music` root key inventory, copied
/// verbatim from the pinned snapshot's safetensors header.
///
/// Weight-free by construction — these are names and shapes, not tensors — but real enough that a
/// prefix or suffix rule that only works on invented keys fails here. The Conv1d entries are the
/// checkpoint's only 3-D DiT weights and the conditioner entry is its only adaptable one.
fn real_key_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut map = BTreeMap::new();
    for (key, shape) in [
        // DiT — Linear.
        ("model.model.to_cond_embed.0.weight", vec![1024, 768]),
        (
            "model.model.transformer.layers.0.cross_attn.to_q.weight",
            vec![1024, 1024],
        ),
        (
            "model.model.transformer.layers.0.cross_attn.to_kv.weight",
            vec![2048, 1024],
        ),
        (
            "model.model.transformer.layers.1.cross_attn.to_q.weight",
            vec![1024, 1024],
        ),
        (
            "model.model.transformer.layers.0.ff.ff.0.proj.weight",
            vec![8192, 1024],
        ),
        // DiT — Conv1d.
        ("model.model.preprocess_conv.weight", vec![256, 256, 1]),
        ("model.model.postprocess_conv.weight", vec![256, 256, 1]),
        // DiT — 1-D norm gain, never a target.
        ("model.model.transformer.layers.0.ff_norm.gamma", vec![1024]),
        // The ONE tensor in the real header that is 2-D, under an allowed prefix, and does *not*
        // end in `.weight` — verified against the pinned small-music safetensors header, where it
        // is the sole such tensor out of 685. It is the DiT's learned memory-token table, not a
        // projection, so adapting it is meaningless; it is here because without it the `.weight`
        // suffix rule in `is_adaptable_target` is unobservable (every other non-`.weight` key in
        // this fixture is 1-D and would be rejected on rank alone) and deleting that rule is green.
        ("model.model.transformer.memory_tokens", vec![64, 1024]),
        // Conditioner — the learned `seconds_total` NumberConditioner Linear, and a 1-D embedding.
        (
            "conditioner.conditioners.seconds_total.embedder.embedding.1.weight",
            vec![768, 256],
        ),
        (
            "conditioner.conditioners.seconds_total.embedder.embedding.1.bias",
            vec![768],
        ),
        (
            "conditioner.conditioners.prompt.padding_embedding",
            vec![768],
        ),
        // SAME — in the same file, and never a target.
        (
            "pretransform.model.encoder.layers.0.mapping.weight_v",
            vec![128, 64, 1],
        ),
        ("pretransform.model.bottleneck.bias", vec![1, 256, 1]),
    ] {
        map.insert(key.to_string(), shape);
    }
    map
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sa3-adapters-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn f32_tensor(values: Vec<f32>, shape: (usize, usize)) -> Tensor {
    Tensor::from_vec(values, shape, &Device::Cpu).expect("tensor")
}

/// A deterministic, non-degenerate fill. Distinct from a constant so a transposed or mis-strided
/// application is visible rather than symmetric.
fn fill(rows: usize, cols: usize, seed: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| {
            let i = (index / cols) as f32;
            let j = (index % cols) as f32;
            ((i * 1.7 + j * 0.31 + seed) as f64).sin() as f32 + 0.05 * (i - j) + 0.11 * seed
        })
        .collect()
}

/// target key (with `.weight`) → factor name → (shape, values)
type RecipeModules = BTreeMap<String, BTreeMap<String, (Vec<usize>, Vec<f32>)>>;

/// Describes one synthetic adapter before it is written to disk.
struct AdapterRecipe {
    kind: &'static str,
    rank: usize,
    alpha: f32,
    include: Option<String>,
    exclude: Option<String>,
    modules: RecipeModules,
}

impl AdapterRecipe {
    fn new(kind: &'static str, rank: usize, alpha: f32) -> Self {
        Self {
            kind,
            rank,
            alpha,
            include: None,
            exclude: None,
            modules: BTreeMap::new(),
        }
    }

    fn factor(mut self, target: &str, factor: &str, shape: Vec<usize>, values: Vec<f32>) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            values.len(),
            "recipe factor {factor} for {target} has {} values for shape {shape:?}",
            values.len()
        );
        self.modules
            .entry(target.to_string())
            .or_default()
            .insert(factor.to_string(), (shape, values));
        self
    }

    fn include(mut self, pattern: &str) -> Self {
        self.include = Some(pattern.to_string());
        self
    }

    fn exclude(mut self, pattern: &str) -> Self {
        self.exclude = Some(pattern.to_string());
        self
    }

    /// Write the native single-file layout: `"{target-without-.weight}.0.{factor}"`.
    fn write_native(&self, path: &Path) {
        let mut tensors: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
        let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        for (target, factors) in &self.modules {
            let stem = target
                .strip_suffix(".weight")
                .expect("recipe targets end in .weight");
            for (factor, (shape, values)) in factors {
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>();
                owned.push((format!("{stem}.0.{factor}"), shape.clone(), bytes));
            }
        }
        for (name, shape, bytes) in &owned {
            tensors.push((
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .expect("view"),
            ));
        }
        let mut metadata = HashMap::new();
        metadata.insert("adapter_type".to_string(), self.kind.to_string());
        metadata.insert("rank".to_string(), self.rank.to_string());
        metadata.insert("alpha".to_string(), self.alpha.to_string());
        if let Some(include) = &self.include {
            metadata.insert("include".to_string(), include.clone());
        }
        if let Some(exclude) = &self.exclude {
            metadata.insert("exclude".to_string(), exclude.clone());
        }
        safetensors::serialize_to_file(tensors, &Some(metadata), path).expect("write adapter");
    }

    /// Write the PEFT directory layout: `adapter_config.json` plus
    /// `base_model.model.{target}.{factor}.weight`.
    fn write_peft(&self, dir: &Path) {
        std::fs::create_dir_all(dir).expect("peft dir");
        let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        for (target, factors) in &self.modules {
            let stem = target.strip_suffix(".weight").expect("target");
            for (factor, (shape, values)) in factors {
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>();
                owned.push((
                    format!("base_model.model.{stem}.{factor}.weight"),
                    shape.clone(),
                    bytes,
                ));
            }
        }
        let tensors: Vec<(String, safetensors::tensor::TensorView<'_>)> = owned
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.clone(),
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        shape.clone(),
                        bytes,
                    )
                    .expect("view"),
                )
            })
            .collect();
        safetensors::serialize_to_file(tensors, &None, &dir.join("adapter_model.safetensors"))
            .expect("write peft weights");
        let mut config = serde_json::Map::new();
        config.insert("r".into(), serde_json::json!(self.rank));
        config.insert("lora_alpha".into(), serde_json::json!(self.alpha));
        config.insert("sa3_adapter_type".into(), serde_json::json!(self.kind));
        if let Some(include) = &self.include {
            config.insert(
                "target_modules".into(),
                serde_json::json!(include.split(',').map(str::trim).collect::<Vec<_>>()),
            );
        }
        std::fs::write(
            dir.join("adapter_config.json"),
            serde_json::to_string_pretty(&serde_json::Value::Object(config)).unwrap(),
        )
        .expect("write peft config");
    }
}

/// A rank-`r` recipe for `target` shaped `[out, in]`, carrying whatever factors `kind` needs.
fn recipe_for(
    kind: AdapterType,
    target: &str,
    out_features: usize,
    in_features: usize,
    rank: usize,
    alpha: f32,
    seed: f32,
) -> AdapterRecipe {
    let mut recipe = AdapterRecipe::new(kind.as_str(), rank, alpha);
    if kind.is_xs() {
        recipe = recipe.factor(
            target,
            "lora_M",
            vec![rank, rank],
            fill(rank, rank, seed + 3.0),
        );
    } else {
        recipe = recipe
            .factor(
                target,
                "lora_A",
                vec![rank, in_features],
                fill(rank, in_features, seed),
            )
            .factor(
                target,
                "lora_B",
                vec![out_features, rank],
                fill(out_features, rank, seed + 1.0),
            );
    }
    if kind.uses_row_magnitude() {
        recipe = recipe.factor(
            target,
            "magnitude_r",
            vec![out_features],
            (0..out_features)
                .map(|i| 0.5 + 0.25 * (i as f32) + seed * 0.01)
                .collect(),
        );
    }
    if kind.uses_col_magnitude() {
        recipe = recipe.factor(
            target,
            "magnitude_c",
            vec![in_features],
            (0..in_features)
                .map(|j| 0.7 + 0.15 * (j as f32) + seed * 0.01)
                .collect(),
        );
    }
    recipe
}

fn spec(path: &Path, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path.to_path_buf(), scale, AdapterKind::Lora)
}

/// A checkpoint-shaped target map for one small square Linear plus a distractor.
fn small_targets(
    target: &str,
    out_features: usize,
    in_features: usize,
) -> BTreeMap<String, Vec<usize>> {
    let mut map = BTreeMap::new();
    map.insert(target.to_string(), vec![out_features, in_features]);
    map.insert(
        "model.model.transformer.layers.9.ff.ff.2.weight".to_string(),
        vec![out_features, in_features],
    );
    map
}

fn rows(tensor: &Tensor) -> Vec<Vec<f32>> {
    tensor.to_vec2::<f32>().expect("2-D f32")
}

const TARGET: &str = "model.model.transformer.layers.0.cross_attn.to_q.weight";

// ---------------------------------------------------------------------------------------------
// 1. The family is complete, and each type is distinguishable from every other
// ---------------------------------------------------------------------------------------------

/// Every one of the eight types loads, plans and folds, and **no two produce the same weight**.
///
/// The pairwise-distinctness half is what makes this more than a smoke test: eight types that all
/// silently fell through to plain LoRA would each "work". The comparison is exact inequality on the
/// raw F32 bytes, so it cannot be satisfied by a near-miss.
#[test]
fn all_eight_types_load_fold_and_produce_pairwise_distinct_weights() {
    let dir = scratch("family");
    let device = Device::Cpu;
    let (out_features, in_features, rank) = (6usize, 5usize, 3usize);
    let base = f32_tensor(
        fill(out_features, in_features, 0.0),
        (out_features, in_features),
    );
    let targets = small_targets(TARGET, out_features, in_features);

    let mut folded: Vec<(AdapterType, Vec<Vec<f32>>)> = Vec::new();
    for kind in AdapterType::ALL {
        let path = dir.join(format!("{}.safetensors", kind.as_str()));
        recipe_for(kind, TARGET, out_features, in_features, rank, 2.0, 1.0).write_native(&path);

        let loaded = load_adapter(&spec(&path, 1.0), &device)
            .unwrap_or_else(|error| panic!("{}: {error}", kind.as_str()));
        assert_eq!(
            loaded.kind,
            kind,
            "{} round-tripped as {:?}",
            kind.as_str(),
            loaded.kind
        );

        let plan = plan_for(&[spec(&path, 1.0)], &targets, &device)
            .unwrap_or_else(|error| panic!("{}: {error}", kind.as_str()));
        assert_eq!(plan.op_count(), 1);
        let ops = plan.ops_for(TARGET).expect("planned target");
        let adapted = apply_adapter_ops(&base, ops).expect("fold");
        assert_eq!(adapted.dims(), base.dims());
        folded.push((kind, rows(&adapted)));
    }
    assert_eq!(folded.len(), 8, "the family is eight types");

    for (left_index, (left_kind, left)) in folded.iter().enumerate() {
        for (right_kind, right) in folded.iter().skip(left_index + 1) {
            assert_ne!(
                left,
                right,
                "{} and {} produced identical weights; one of them is not implemented",
                left_kind.as_str(),
                right_kind.as_str()
            );
        }
        assert_ne!(
            left,
            &rows(&base),
            "{} left the base weight unchanged",
            left_kind.as_str()
        );
    }
}

/// The legacy `dora` alias resolves by sniffing the magnitude tensor, defaulting to rows.
///
/// Three cases, and the third is the discriminating one: a file carrying **only** `magnitude_c`
/// must resolve to columns. A resolver hard-wired to rows passes the first two and fails that one.
#[test]
fn the_legacy_dora_alias_resolves_by_magnitude_shape_defaulting_to_rows() {
    assert_eq!(
        resolve_adapter_type("dora", true, false).unwrap(),
        AdapterType::DoraRows
    );
    assert_eq!(
        resolve_adapter_type("dora", false, false).unwrap(),
        AdapterType::DoraRows,
        "an alias with no magnitude at all defaults to rows, upstream's training default"
    );
    assert_eq!(
        resolve_adapter_type("dora", false, true).unwrap(),
        AdapterType::DoraCols
    );
    assert_eq!(
        resolve_adapter_type("dora", true, true).unwrap(),
        AdapterType::DoraRows,
        "an ambiguous alias falls back to rows rather than guessing columns"
    );
    assert_eq!(
        resolve_adapter_type("dora-xs", false, true).unwrap(),
        AdapterType::DoraColsXs
    );
    assert_eq!(
        resolve_adapter_type("DORA-Rows", false, false).unwrap(),
        AdapterType::DoraRows,
        "type names are case-insensitive"
    );
    assert!(resolve_adapter_type("lokr", false, false).is_err());
    assert!(resolve_adapter_type("", false, false).is_err());

    // An end-to-end alias file: written as `dora`, loaded as `dora-cols` purely from its tensors.
    let dir = scratch("alias");
    let (out_features, in_features, rank) = (4usize, 4usize, 2usize);
    let path = dir.join("alias.safetensors");
    AdapterRecipe::new("dora", rank, 1.0)
        .factor(
            TARGET,
            "lora_A",
            vec![rank, in_features],
            fill(rank, in_features, 1.0),
        )
        .factor(
            TARGET,
            "lora_B",
            vec![out_features, rank],
            fill(out_features, rank, 2.0),
        )
        .factor(
            TARGET,
            "magnitude_c",
            vec![in_features],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .write_native(&path);
    let loaded = load_adapter(&spec(&path, 1.0), &Device::Cpu).expect("alias");
    assert_eq!(loaded.kind, AdapterType::DoraCols);

    // The other side of the sniff, end-to-end through a file: an alias carrying BOTH magnitudes is
    // ambiguous and must fall back to rows. Without this the `has_row` half of the sniff is never
    // observed through a real file — every alias file above carries `magnitude_c` alone, so the
    // row accumulator can be stuck at `false` and the loader still answers correctly.
    let both = dir.join("alias-both.safetensors");
    AdapterRecipe::new("dora", rank, 1.0)
        .factor(
            TARGET,
            "lora_A",
            vec![rank, in_features],
            fill(rank, in_features, 1.0),
        )
        .factor(
            TARGET,
            "lora_B",
            vec![out_features, rank],
            fill(out_features, rank, 2.0),
        )
        .factor(
            TARGET,
            "magnitude_r",
            vec![out_features],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .factor(
            TARGET,
            "magnitude_c",
            vec![in_features],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .write_native(&both);
    let loaded = load_adapter(&spec(&both, 1.0), &Device::Cpu).expect("ambiguous alias");
    assert_eq!(
        loaded.kind,
        AdapterType::DoraRows,
        "an alias file carrying both magnitudes must resolve to rows, not columns"
    );
}

/// `AdapterPlan::build` refuses a module-less adapter handed to it directly.
///
/// `build` is public and takes caller-constructed [`LoadedAdapter`]s, which is the only way its
/// zero-matched-targets refusal can fire: on every shipped path `load_adapter` has already rejected
/// a module-less file at `finish_modules`, and any module that fails to match returns earlier and
/// more specifically. Without this case that refusal is dead code that reads as live — deleting it
/// is green — so this is what keeps the public entry point's contract honest.
#[test]
fn a_module_less_adapter_is_refused_by_build() {
    let targets = small_targets(TARGET, 4, 4);
    let empty = LoadedAdapter {
        path: PathBuf::from("/synthetic/module-less.safetensors"),
        kind: AdapterType::Lora,
        rank: 2,
        alpha: 1.0,
        scale: 1.0,
        include: Vec::new(),
        exclude: Vec::new(),
        modules: BTreeMap::new(),
        spec_index: 0,
    };
    let error = AdapterPlan::build(&[empty], &targets)
        .expect_err("a module-less adapter must not produce a plan");
    assert!(
        error.to_string().contains("no_target_matched"),
        "unexpected refusal: {error}"
    );
}

// ---------------------------------------------------------------------------------------------
// 2. The math, gated by properties that a wrong axis or a dropped factor cannot satisfy
// ---------------------------------------------------------------------------------------------

/// Classic LoRA's delta is exactly `(alpha/rank) * scale * (B @ A)`.
///
/// The oracle here is a hand-rolled `B @ A` over tiny integers, computed independently of the
/// shipped path — not a call back into it.
#[test]
fn classic_lora_folds_exactly_alpha_over_rank_times_scale_times_b_at_a() {
    let dir = scratch("lora-math");
    let (out_features, in_features, rank) = (2usize, 3usize, 1usize);
    let path = dir.join("lora.safetensors");
    // A = [[1, 2, 3]], B = [[2], [-1]]  ⇒  B@A = [[2, 4, 6], [-1, -2, -3]]
    AdapterRecipe::new("lora", rank, 4.0)
        .factor(TARGET, "lora_A", vec![1, 3], vec![1.0, 2.0, 3.0])
        .factor(TARGET, "lora_B", vec![2, 1], vec![2.0, -1.0])
        .write_native(&path);

    let base = f32_tensor(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], (2, 3));
    let targets = small_targets(TARGET, out_features, in_features);
    // alpha/rank = 4, scale = 0.5 ⇒ effective 2.0
    let plan = plan_for(&[spec(&path, 0.5)], &targets, &Device::Cpu).expect("plan");
    let adapted = apply_adapter_ops(&base, plan.ops_for(TARGET).unwrap()).expect("fold");

    let expected = [
        [10.0 + 2.0 * 2.0, 20.0 + 2.0 * 4.0, 30.0 + 2.0 * 6.0],
        [40.0 - 2.0 * 1.0, 50.0 - 2.0 * 2.0, 60.0 - 2.0 * 3.0],
    ];
    for (row, (got, want)) in rows(&adapted).iter().zip(expected.iter()).enumerate() {
        for (column, (got, want)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "[{row}][{column}]: {got} vs {want}"
            );
        }
    }
}

/// Classic LoRA's delta is **exactly linear in the requested strength**: `‖δ(s)‖ == s·‖δ(1)‖`.
///
/// The single-point check above pins `alpha/rank · s` at `s = 0.5` only, which a scale applied to
/// the wrong factor, squared, or clamped can still satisfy at that one value. This sweeps
/// `{0.25, 0.5, 1, 2}` — including `s > 1`, where a clamp to `[0, 1]` shows up — and the relation is
/// analytically exact for the classic family because `s` multiplies `B @ A` and nothing else. That
/// exactness is the point: this is not a monotonicity check with a tolerance to tune, it is an
/// equality, so it cannot be passed by a fold that merely trends the right way.
///
/// Deliberately **not** extended to DoRA/BoRA: there `s` participates inside the normalization, so
/// the delta is genuinely non-linear in `s` and no such identity holds. That asymmetry is the reason
/// the linearity gate is stated for classic LoRA specifically.
#[test]
fn classic_lora_delta_norm_is_exactly_linear_in_the_requested_strength() {
    let dir = scratch("lora-linearity");
    let (out_features, in_features, rank) = (4usize, 5usize, 2usize);
    let path = dir.join("lora.safetensors");
    AdapterRecipe::new("lora", rank, 3.0)
        .factor(
            TARGET,
            "lora_A",
            vec![rank, in_features],
            fill(rank, in_features, 1.0),
        )
        .factor(
            TARGET,
            "lora_B",
            vec![out_features, rank],
            fill(out_features, rank, 2.0),
        )
        .write_native(&path);

    let base = f32_tensor(
        fill(out_features, in_features, 5.0),
        (out_features, in_features),
    );
    let targets = small_targets(TARGET, out_features, in_features);

    // ‖δ(s)‖: the Frobenius norm of (adapted − base) at strength `s`.
    let delta_norm = |scale: f32| -> f64 {
        let plan = plan_for(&[spec(&path, scale)], &targets, &Device::Cpu).expect("plan");
        let adapted = apply_adapter_ops(&base, plan.ops_for(TARGET).unwrap()).expect("fold");
        rows(&adapted)
            .iter()
            .flatten()
            .zip(rows(&base).iter().flatten())
            .map(|(got, want)| {
                let difference = (*got - *want) as f64;
                difference * difference
            })
            .sum::<f64>()
            .sqrt()
    };

    let unit = delta_norm(1.0);
    assert!(
        unit > 1e-3,
        "the fixture's delta is degenerate ({unit}); the ratios below would be vacuous"
    );
    for scale in [0.25f32, 0.5, 1.0, 2.0] {
        let got = delta_norm(scale);
        let want = scale as f64 * unit;
        assert!(
            (got - want).abs() <= 1e-5 * want.max(1.0),
            "‖δ({scale})‖ = {got}, expected {want} = {scale}·‖δ(1)‖; the strength is not linear"
        );
    }
    // The sweep is discriminating only because the values genuinely differ from one another.
    assert!(
        delta_norm(2.0) - delta_norm(0.25) > 1e-3,
        "the strength sweep produced a constant delta norm"
    );
}

/// `dora-rows` leaves **every row** with norm exactly `|magnitude_r[i]|`.
///
/// This is the strongest available single assertion for the row family: it fails if the
/// normalization is skipped, if it runs on the wrong axis, if the magnitude is not applied, or if
/// the magnitude is applied before the normalization instead of after. And it is *not* satisfied by
/// the base weight, which is asserted separately so the case cannot pass on a no-op.
#[test]
fn dora_rows_normalizes_rows_and_rescales_by_magnitude_r() {
    let dir = scratch("dora-rows");
    let (out_features, in_features, rank) = (4usize, 5usize, 2usize);
    let magnitudes: Vec<f32> = vec![0.5, 1.25, 2.0, 3.5];
    let path = dir.join("dora-rows.safetensors");
    AdapterRecipe::new("dora-rows", rank, 1.0)
        .factor(
            TARGET,
            "lora_A",
            vec![rank, in_features],
            fill(rank, in_features, 1.0),
        )
        .factor(
            TARGET,
            "lora_B",
            vec![out_features, rank],
            fill(out_features, rank, 2.0),
        )
        .factor(
            TARGET,
            "magnitude_r",
            vec![out_features],
            magnitudes.clone(),
        )
        .write_native(&path);

    let base = f32_tensor(
        fill(out_features, in_features, 5.0),
        (out_features, in_features),
    );
    let targets = small_targets(TARGET, out_features, in_features);
    let plan = plan_for(&[spec(&path, 1.0)], &targets, &Device::Cpu).expect("plan");
    let adapted = apply_adapter_ops(&base, plan.ops_for(TARGET).unwrap()).expect("fold");

    for (index, row) in rows(&adapted).iter().enumerate() {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!(
            (norm - magnitudes[index]).abs() < 1e-4,
            "row {index} norm {norm} != magnitude_r {}",
            magnitudes[index]
        );
    }
    // The base does not already satisfy it, so the assertion above is discriminating.
    for (index, row) in rows(&base).iter().enumerate() {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!(
            (norm - magnitudes[index]).abs() > 1e-3,
            "base row {index} already has norm {norm}; the fixture cannot discriminate"
        );
    }
}

/// The column mirror of the case above — and the reason both exist is that a single-axis test
/// cannot tell `dora-rows` from `dora-cols`.
#[test]
fn dora_cols_normalizes_columns_and_rescales_by_magnitude_c() {
    let dir = scratch("dora-cols");
    let (out_features, in_features, rank) = (4usize, 3usize, 2usize);
    let magnitudes: Vec<f32> = vec![0.75, 1.5, 2.25];
    let path = dir.join("dora-cols.safetensors");
    AdapterRecipe::new("dora-cols", rank, 1.0)
        .factor(
            TARGET,
            "lora_A",
            vec![rank, in_features],
            fill(rank, in_features, 1.0),
        )
        .factor(
            TARGET,
            "lora_B",
            vec![out_features, rank],
            fill(out_features, rank, 2.0),
        )
        .factor(TARGET, "magnitude_c", vec![in_features], magnitudes.clone())
        .write_native(&path);

    let base = f32_tensor(
        fill(out_features, in_features, 5.0),
        (out_features, in_features),
    );
    let targets = small_targets(TARGET, out_features, in_features);
    let plan = plan_for(&[spec(&path, 1.0)], &targets, &Device::Cpu).expect("plan");
    let adapted = apply_adapter_ops(&base, plan.ops_for(TARGET).unwrap()).expect("fold");

    let table = rows(&adapted);
    for column in 0..in_features {
        let norm = table
            .iter()
            .map(|row| row[column] * row[column])
            .sum::<f32>()
            .sqrt();
        assert!(
            (norm - magnitudes[column]).abs() < 1e-4,
            "column {column} norm {norm} != magnitude_c {}",
            magnitudes[column]
        );
    }
}

/// `bora` is rows **then** columns, and the order within the type is observable.
///
/// The columns-last property holds exactly; the rows-first half is proven by contrast against
/// `dora-cols` built from the same column magnitude — if `bora` ignored `magnitude_r` it would be
/// `dora-cols` exactly, and it is asserted not to be.
#[test]
fn bora_applies_rows_then_columns_and_is_not_dora_cols() {
    let dir = scratch("bora");
    let (out_features, in_features, rank) = (4usize, 3usize, 2usize);
    let row_magnitudes: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0];
    let col_magnitudes: Vec<f32> = vec![0.75, 1.5, 2.25];
    let targets = small_targets(TARGET, out_features, in_features);
    let base = f32_tensor(
        fill(out_features, in_features, 5.0),
        (out_features, in_features),
    );

    let bora = dir.join("bora.safetensors");
    AdapterRecipe::new("bora", rank, 1.0)
        .factor(
            TARGET,
            "lora_A",
            vec![rank, in_features],
            fill(rank, in_features, 1.0),
        )
        .factor(
            TARGET,
            "lora_B",
            vec![out_features, rank],
            fill(out_features, rank, 2.0),
        )
        .factor(TARGET, "magnitude_r", vec![out_features], row_magnitudes)
        .factor(
            TARGET,
            "magnitude_c",
            vec![in_features],
            col_magnitudes.clone(),
        )
        .write_native(&bora);
    let cols = dir.join("cols.safetensors");
    AdapterRecipe::new("dora-cols", rank, 1.0)
        .factor(
            TARGET,
            "lora_A",
            vec![rank, in_features],
            fill(rank, in_features, 1.0),
        )
        .factor(
            TARGET,
            "lora_B",
            vec![out_features, rank],
            fill(out_features, rank, 2.0),
        )
        .factor(
            TARGET,
            "magnitude_c",
            vec![in_features],
            col_magnitudes.clone(),
        )
        .write_native(&cols);

    let fold = |path: &Path| {
        let plan = plan_for(&[spec(path, 1.0)], &targets, &Device::Cpu).expect("plan");
        rows(&apply_adapter_ops(&base, plan.ops_for(TARGET).unwrap()).expect("fold"))
    };
    let bora_rows = fold(&bora);
    let cols_rows = fold(&cols);

    // Columns last ⇒ column norms land on `magnitude_c` exactly.
    for column in 0..in_features {
        let norm = bora_rows
            .iter()
            .map(|row| row[column] * row[column])
            .sum::<f32>()
            .sqrt();
        assert!(
            (norm - col_magnitudes[column]).abs() < 1e-4,
            "bora column {column} norm {norm} != {}",
            col_magnitudes[column]
        );
    }
    // …but the row magnitude still moved the result, so `bora` is not `dora-cols` in disguise.
    assert_ne!(
        bora_rows, cols_rows,
        "bora ignored its row magnitude and collapsed to dora-cols"
    );
}

/// The `-xs` half, gated on a base weight whose SVD is known in closed form.
///
/// `base = diag(5, 3, 2)` has `U = V = I` and `S = (5, 3, 2)`, already descending and already
/// sign-canonical (every column's largest entry is `+1`). So `U @ M @ Vᵀ` **is** `M`, padded into
/// the leading `rank × rank` block, and the expected adapted weight is exact arithmetic rather than
/// a bound. Ordering drift, truncation drift and sign drift each break it visibly.
#[test]
fn lora_xs_reconstructs_through_the_base_weights_own_singular_bases() {
    let dir = scratch("lora-xs");
    let rank = 2usize;
    let path = dir.join("xs.safetensors");
    AdapterRecipe::new("lora-xs", rank, 2.0)
        .factor(TARGET, "lora_M", vec![rank, rank], vec![1.0, 2.0, 3.0, 4.0])
        .write_native(&path);

    let base = f32_tensor(vec![5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0], (3, 3));
    let targets = small_targets(TARGET, 3, 3);
    // alpha/rank = 1.0, scale = 1.0 ⇒ effective 1.0.
    let plan = plan_for(&[spec(&path, 1.0)], &targets, &Device::Cpu).expect("plan");
    let adapted = rows(&apply_adapter_ops(&base, plan.ops_for(TARGET).unwrap()).expect("fold"));

    let expected = [
        [5.0 + 1.0, 2.0, 0.0],
        [3.0, 3.0 + 4.0, 0.0],
        [0.0, 0.0, 2.0],
    ];
    for (i, row) in expected.iter().enumerate() {
        for (j, want) in row.iter().enumerate() {
            assert!(
                (adapted[i][j] - want).abs() < 1e-4,
                "[{i}][{j}]: {} vs {want}",
                adapted[i][j]
            );
        }
    }
}

/// The `-xs` sign rule, gated where it is actually observable.
///
/// `diag(5, -3, 2)`'s second left singular vector is `-e₂` before canonicalization; the rule flips
/// it to `+e₂` and **must flip `v₂` with it**. If the flip propagates, the rank-2 block of the
/// delta picks up a `-1` on both its row and its column and the reconstruction is exact. If `v`
/// were canonicalized independently — the drift this rule exists to prevent — the off-diagonal
/// entries come out with the wrong sign, which this asserts against directly.
#[test]
fn xs_sign_canonicalization_propagates_from_u_to_v() {
    let dir = scratch("xs-sign");
    let rank = 2usize;
    let path = dir.join("xs-sign.safetensors");
    AdapterRecipe::new("lora-xs", rank, 2.0)
        .factor(TARGET, "lora_M", vec![rank, rank], vec![1.0, 2.0, 3.0, 4.0])
        .write_native(&path);

    let base = f32_tensor(vec![5.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 2.0], (3, 3));
    let targets = small_targets(TARGET, 3, 3);
    let plan = plan_for(&[spec(&path, 1.0)], &targets, &Device::Cpu).expect("plan");
    let adapted = rows(&apply_adapter_ops(&base, plan.ops_for(TARGET).unwrap()).expect("fold"));

    // u₁ = e₀ (+), u₂ = +e₁ after the flip, v₁ = e₀, v₂ = -e₁.
    // delta = u₁M₀₀v₁ᵀ + u₁M₀₁v₂ᵀ + u₂M₁₀v₁ᵀ + u₂M₁₁v₂ᵀ
    //       = [[1, -2, 0], [3, -4, 0], [0, 0, 0]]
    let expected = [
        [5.0 + 1.0, -2.0, 0.0],
        [3.0, -3.0 - 4.0, 0.0],
        [0.0, 0.0, 2.0],
    ];
    for (i, row) in expected.iter().enumerate() {
        for (j, want) in row.iter().enumerate() {
            assert!(
                (adapted[i][j] - want).abs() < 1e-4,
                "[{i}][{j}]: {} vs {want} — the u/v sign flip did not travel together",
                adapted[i][j]
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 3. Order semantics — the thing the image lane does not have to care about
// ---------------------------------------------------------------------------------------------

/// Two DoRA adapters do **not** commute; two classic-LoRA adapters do.
///
/// Both halves matter. The inequality alone could be satisfied by any bug that made the fold
/// order-sensitive for the wrong reason; the equality proves the harness is measuring the
/// parametrization's own algebra rather than incidental noise, and it pins the documented contrast
/// with the image lane.
#[test]
fn dora_stacks_do_not_commute_and_classic_lora_stacks_do() {
    let dir = scratch("order");
    let device = Device::Cpu;
    let (out_features, in_features, rank) = (4usize, 4usize, 2usize);
    let base = f32_tensor(
        fill(out_features, in_features, 7.0),
        (out_features, in_features),
    );
    let targets = small_targets(TARGET, out_features, in_features);

    // Each adapter carries its OWN strength, so reversing the pair reverses only the *order* —
    // if the scales travelled with the slot instead, even classic LoRA would look order-dependent
    // and the contrast this case exists to draw would be meaningless.
    let fold_pair = |first: (&Path, f32), second: (&Path, f32)| {
        let plan = plan_for(
            &[spec(first.0, first.1), spec(second.0, second.1)],
            &targets,
            &device,
        )
        .expect("plan the pair");
        assert_eq!(plan.op_count(), 2, "both adapters must reach the target");
        rows(&apply_adapter_ops(&base, plan.ops_for(TARGET).unwrap()).expect("fold"))
    };

    for (kind, expect_commutes) in [
        (AdapterType::Lora, true),
        (AdapterType::DoraRows, false),
        (AdapterType::DoraCols, false),
        (AdapterType::Bora, false),
    ] {
        let a = dir.join(format!("{}-a.safetensors", kind.as_str()));
        let b = dir.join(format!("{}-b.safetensors", kind.as_str()));
        recipe_for(kind, TARGET, out_features, in_features, rank, 2.0, 1.0).write_native(&a);
        recipe_for(kind, TARGET, out_features, in_features, rank, 3.0, 9.0).write_native(&b);

        let forward = fold_pair((&a, 0.8), (&b, 1.3));
        let reverse = fold_pair((&b, 1.3), (&a, 0.8));
        // The separation is the point, and it is four orders of magnitude wide. Classic LoRA's
        // two orderings differ only because F32 addition is not associative — a last-ulp
        // disagreement — so the bound below is a rounding bound, not a tolerance chosen to make
        // the case pass. DoRA/BoRA disagree macroscopically because the strength sits inside the
        // normalization. Asserting BOTH directions is what proves the harness measures order
        // rather than noise: a fold that ignored order entirely would fail the second half, and
        // one that was order-sensitive for an unrelated reason would fail the first.
        let spread = |left: &Vec<Vec<f32>>, right: &Vec<Vec<f32>>| {
            left.iter()
                .flatten()
                .zip(right.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max)
        };
        let observed = spread(&forward, &reverse);
        if expect_commutes {
            assert!(
                observed < 1e-5,
                "{} is linear in its delta, so the two orderings must agree to F32 rounding; \
                 observed {observed}",
                kind.as_str()
            );
        } else {
            assert!(
                observed > 1e-3,
                "{} folded order-independently ({observed}); the strength is not inside the \
                 normalization",
                kind.as_str()
            );
        }
        // Either way, the stack is not just one of its members.
        let single = {
            let plan = plan_for(&[spec(&a, 0.8)], &targets, &device).expect("plan single");
            rows(&apply_adapter_ops(&base, plan.ops_for(TARGET).unwrap()).expect("fold"))
        };
        assert_ne!(
            forward,
            single,
            "{}: the second adapter of the stack was dropped",
            kind.as_str()
        );
    }
}

/// The stack's fold is the **absolute** sequence `A` then `B`, not merely "some order".
///
/// ⚠ This case exists because the one above was **vacuous under the mutation it was credited with
/// catching**, and running that mutation is what found it. Reversing `apply_adapter_ops`'s fold
/// loop passed all 28 cases: `dora_stacks_do_not_commute_and_classic_lora_stacks_do` compares
/// `[A, B]` against `[B, A]`, so a *uniform* reversal simply swaps the two sides and the inequality
/// survives untouched. That is sc-14548's "a test comparing two of its own renders is blind to any
/// mutation applied uniformly to both", in its purest form — and
/// `the_plan_preserves_request_order_at_both_ends` did not close it either, because it inspects the
/// **plan's** recorded indices, which a reversal in the *fold* never touches.
///
/// The fix is an absolute reference: fold `A` alone onto the base, fold `B` alone onto **that**
/// result, and require the two-deep stack to equal it exactly. The two sides are not the same
/// binding — one walks a two-element slice, the other two one-element slices, and reversing a
/// one-element slice is a no-op — so the comparison cannot agree with itself.
#[test]
fn a_two_deep_stack_folds_the_first_adapter_first() {
    let dir = scratch("absolute-order");
    let device = Device::Cpu;
    let (out_features, in_features, rank) = (4usize, 4usize, 2usize);
    let base = f32_tensor(
        fill(out_features, in_features, 7.0),
        (out_features, in_features),
    );
    let targets = small_targets(TARGET, out_features, in_features);

    let first = dir.join("first.safetensors");
    let second = dir.join("second.safetensors");
    recipe_for(
        AdapterType::DoraRows,
        TARGET,
        out_features,
        in_features,
        rank,
        2.0,
        1.0,
    )
    .write_native(&first);
    recipe_for(
        AdapterType::DoraRows,
        TARGET,
        out_features,
        in_features,
        rank,
        3.0,
        9.0,
    )
    .write_native(&second);

    let single = |path: &Path, scale: f32| {
        plan_for(&[spec(path, scale)], &targets, &device).expect("single plan")
    };
    let after_first =
        apply_adapter_ops(&base, single(&first, 0.8).ops_for(TARGET).unwrap()).expect("fold");
    let composed = apply_adapter_ops(&after_first, single(&second, 1.3).ops_for(TARGET).unwrap())
        .expect("fold");

    let stacked =
        plan_for(&[spec(&first, 0.8), spec(&second, 1.3)], &targets, &device).expect("stack plan");
    let stacked = apply_adapter_ops(&base, stacked.ops_for(TARGET).unwrap()).expect("fold");

    assert_eq!(
        rows(&stacked),
        rows(&composed),
        "the two-deep stack is not `first` then `second`"
    );
    // Discriminating control: the OTHER composition order really is different, so the assertion
    // above is pinned to one specific sequence rather than to any sequence.
    let after_second =
        apply_adapter_ops(&base, single(&second, 1.3).ops_for(TARGET).unwrap()).expect("fold");
    let reversed = apply_adapter_ops(&after_second, single(&first, 0.8).ops_for(TARGET).unwrap())
        .expect("fold");
    assert_ne!(
        rows(&stacked),
        rows(&reversed),
        "the fixture cannot tell the two orders apart"
    );
}

/// A stack is folded in **request order**, and the plan records that order.
///
/// Probing both ends, per sc-14549's finding that a gate catching a loop which stops early is blind
/// to one that starts late: this asserts the **first** op's index is 0 and the **last** op's index
/// is `n - 1`, for a three-deep stack.
#[test]
fn the_plan_preserves_request_order_at_both_ends() {
    let dir = scratch("order-index");
    let (out_features, in_features, rank) = (4usize, 4usize, 2usize);
    let targets = small_targets(TARGET, out_features, in_features);
    let mut specs = Vec::new();
    for index in 0..3 {
        let path = dir.join(format!("stack-{index}.safetensors"));
        recipe_for(
            AdapterType::DoraRows,
            TARGET,
            out_features,
            in_features,
            rank,
            1.0 + index as f32,
            index as f32,
        )
        .write_native(&path);
        specs.push(spec(&path, 1.0 + index as f32 * 0.1));
    }
    let plan = plan_for(&specs, &targets, &Device::Cpu).expect("plan");
    let ops = plan.ops_for(TARGET).expect("target");
    assert_eq!(ops.len(), 3);
    assert_eq!(
        ops.first().unwrap().adapter_index,
        0,
        "the stack started late"
    );
    assert_eq!(
        ops.last().unwrap().adapter_index,
        2,
        "the stack stopped early"
    );
    assert_eq!(ops[1].adapter_index, 1);
}

/// `adapter_index` is the position in the caller's **original** request, across the zero-scale
/// filter.
///
/// `plan_for` validates the whole stack, then rebuilds the plan from the zero-scale-**filtered**
/// slice. Numbering the ops positionally while rebuilding would renumber the survivors: for
/// `[zero, live]` the live op would report index 0 and an error message would blame the wrong
/// adapter — the inert one the caller explicitly turned off. Every other case in this file stacks
/// only live adapters, so the filter is invisible to them and the bug is unobservable there.
#[test]
fn adapter_index_survives_the_zero_scale_filter() {
    let dir = scratch("index-filter");
    let (out_features, in_features, rank) = (4usize, 4usize, 2usize);
    let targets = small_targets(TARGET, out_features, in_features);

    let inert = dir.join("inert.safetensors");
    let live = dir.join("live.safetensors");
    for (path, seed) in [(&inert, 1.0f32), (&live, 2.0)] {
        recipe_for(
            AdapterType::Lora,
            TARGET,
            out_features,
            in_features,
            rank,
            2.0,
            seed,
        )
        .write_native(path);
    }

    // `[zero, live]`: the surviving op is request entry 1, not entry 0.
    let plan = plan_for(
        &[spec(&inert, 0.0), spec(&live, 1.0)],
        &targets,
        &Device::Cpu,
    )
    .expect("plan");
    let ops = plan.ops_for(TARGET).expect("target");
    assert_eq!(ops.len(), 1, "the zero-scale member contributed an op");
    assert_eq!(
        ops[0].adapter_index, 1,
        "the surviving op reports its position in the filtered slice, not in the caller's request"
    );

    // The mirror image, so the assertion cannot be satisfied by a constant: `[live, zero]` keeps 0.
    let plan = plan_for(
        &[spec(&live, 1.0), spec(&inert, 0.0)],
        &targets,
        &Device::Cpu,
    )
    .expect("plan");
    let ops = plan.ops_for(TARGET).expect("target");
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].adapter_index, 0);
}

// ---------------------------------------------------------------------------------------------
// 4. `scale == 0.0`
// ---------------------------------------------------------------------------------------------

/// A zero-scale stack is **fully validated** and then produces an **empty plan**, so the weights
/// are the un-adapted bytes rather than a fold that happens to add zero.
///
/// The `0 * NaN` half is the discriminating one and it is asserted by *construction*: the same
/// adapter's factors are large enough that `B @ A` overflows F32 to `inf`, so a multiply-by-zero
/// implementation would yield `NaN` in every entry. The case proves both that the shipped path does
/// not (empty plan, identical bytes) and that the hazard is real (the same op folded by hand *does*
/// produce non-finite values), so the fast path is shown to be load-bearing rather than cosmetic.
#[test]
fn a_zero_scale_stack_validates_fully_and_then_mutates_nothing() {
    let dir = scratch("zero");
    let device = Device::Cpu;
    let (out_features, in_features, rank) = (4usize, 4usize, 2usize);
    let base = f32_tensor(
        fill(out_features, in_features, 2.0),
        (out_features, in_features),
    );
    let targets = small_targets(TARGET, out_features, in_features);

    let overflowing = dir.join("overflowing.safetensors");
    AdapterRecipe::new("lora", rank, 1.0)
        .factor(
            TARGET,
            "lora_A",
            vec![rank, in_features],
            vec![1e30; rank * in_features],
        )
        .factor(
            TARGET,
            "lora_B",
            vec![out_features, rank],
            vec![1e30; out_features * rank],
        )
        .write_native(&overflowing);

    let plan = plan_for(&[spec(&overflowing, 0.0)], &targets, &device).expect("zero-scale plan");
    assert!(
        plan.is_empty(),
        "a zero-scale stack must resolve to an empty plan, not to a zero-valued fold"
    );
    assert_eq!(plan.op_count(), 0);

    // Bit identity, asserted on raw bytes rather than on a norm.
    let untouched = apply_adapter_ops(&base, plan.ops_for(TARGET).unwrap_or(&[])).expect("fold");
    assert_eq!(
        base.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        untouched.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        "the un-adapted weight was not returned bit-identically"
    );

    // The hazard is real: at scale 1.0 the very same adapter overflows, so `0 * delta` would be NaN.
    let hot = plan_for(&[spec(&overflowing, 1.0)], &targets, &device).expect("plan");
    let folded = apply_adapter_ops(&base, hot.ops_for(TARGET).unwrap()).expect("fold");
    assert!(
        folded
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .any(|value| !value.is_finite()),
        "the fixture no longer overflows, so it cannot demonstrate the 0*inf hazard"
    );

    // A zero-scale adapter is still *validated*: a mistyped key in one is refused, not skipped.
    let broken = dir.join("broken.safetensors");
    AdapterRecipe::new("lora", rank, 1.0)
        .factor(
            "model.model.nope.weight",
            "lora_A",
            vec![rank, in_features],
            fill(rank, in_features, 1.0),
        )
        .factor(
            "model.model.nope.weight",
            "lora_B",
            vec![out_features, rank],
            fill(out_features, rank, 1.0),
        )
        .write_native(&broken);
    let error = plan_for(&[spec(&broken, 0.0)], &targets, &device)
        .expect_err("a zero-scale adapter with a bad key must still be refused");
    assert!(
        error.to_string().contains("no_target_matched"),
        "unexpected message: {error}"
    );
}

/// A stack that mixes a live adapter with a zero-scale one folds **only** the live one, and the
/// result is byte-identical to requesting the live one alone.
#[test]
fn a_zero_scale_member_of_a_mixed_stack_contributes_nothing() {
    let dir = scratch("zero-mixed");
    let device = Device::Cpu;
    let (out_features, in_features, rank) = (4usize, 4usize, 2usize);
    let base = f32_tensor(
        fill(out_features, in_features, 2.0),
        (out_features, in_features),
    );
    let targets = small_targets(TARGET, out_features, in_features);

    let live = dir.join("live.safetensors");
    let inert = dir.join("inert.safetensors");
    recipe_for(
        AdapterType::Bora,
        TARGET,
        out_features,
        in_features,
        rank,
        2.0,
        1.0,
    )
    .write_native(&live);
    recipe_for(
        AdapterType::DoraRows,
        TARGET,
        out_features,
        in_features,
        rank,
        5.0,
        8.0,
    )
    .write_native(&inert);

    let mixed = plan_for(&[spec(&live, 1.0), spec(&inert, 0.0)], &targets, &device).expect("plan");
    let alone = plan_for(&[spec(&live, 1.0)], &targets, &device).expect("plan");
    assert_eq!(mixed.op_count(), 1);
    assert_eq!(
        rows(&apply_adapter_ops(&base, mixed.ops_for(TARGET).unwrap()).unwrap()),
        rows(&apply_adapter_ops(&base, alone.ops_for(TARGET).unwrap()).unwrap()),
    );
}

// ---------------------------------------------------------------------------------------------
// 5. Targets: DiT + conditioner, never SAME, never T5Gemma
// ---------------------------------------------------------------------------------------------

/// The adaptable-target rule, applied to the **real** key inventory.
#[test]
fn only_dit_and_conditioner_linear_and_conv1d_weights_are_targets() {
    let shapes = real_key_shapes();
    let targets = adaptable_targets(&shapes);

    for key in [
        "model.model.to_cond_embed.0.weight",
        "model.model.transformer.layers.0.cross_attn.to_q.weight",
        "model.model.transformer.layers.0.cross_attn.to_kv.weight",
        "model.model.preprocess_conv.weight",
        "model.model.postprocess_conv.weight",
        "conditioner.conditioners.seconds_total.embedder.embedding.1.weight",
    ] {
        assert!(targets.contains_key(key), "{key} should be adaptable");
    }
    for key in [
        // SAME lives in the same file and must never be reachable.
        "pretransform.model.encoder.layers.0.mapping.weight_v",
        "pretransform.model.bottleneck.bias",
        // 1-D gains, biases and embeddings: none of the eight types is a bias-diff type.
        "model.model.transformer.layers.0.ff_norm.gamma",
        "conditioner.conditioners.seconds_total.embedder.embedding.1.bias",
        "conditioner.conditioners.prompt.padding_embedding",
        // 2-D, under an allowed prefix, and excluded **only** by the `.weight` suffix rule: a
        // learned parameter table, not a projection. This is the one key in the fixture that makes
        // that rule discriminating, so `targets.len()` below is a real assertion about it.
        "model.model.transformer.memory_tokens",
    ] {
        assert!(!targets.contains_key(key), "{key} must not be adaptable");
    }
    assert_eq!(targets.len(), 8, "the adaptable set drifted: {targets:?}");

    // The prefix list is the whole allowlist, so anything outside it is out by construction.
    assert_eq!(ADAPTABLE_PREFIXES, &["model.", "conditioner."]);
    for key in targets.keys() {
        assert!(
            ADAPTABLE_PREFIXES
                .iter()
                .any(|prefix| key.starts_with(prefix)),
            "{key} escaped the prefix allowlist"
        );
    }
    // Direct, shape-aware: a 4-D tensor under an allowed prefix is still not a target.
    assert!(!is_adaptable_target(
        "model.model.something.weight",
        &[2, 2, 2, 2]
    ));
    assert!(is_adaptable_target("model.model.x.weight", &[4, 4]));
    assert!(is_adaptable_target("model.model.x.weight", &[4, 4, 3]));
    // The suffix rule alone, isolated from prefix and rank: same prefix, same rank, no `.weight`.
    assert!(
        !is_adaptable_target("model.model.transformer.memory_tokens", &[64, 1024]),
        "a 2-D non-`.weight` tensor under an allowed prefix became an adapter target"
    );
}

/// An adapter naming a SAME tensor, or a T5Gemma one, is refused loudly.
///
/// T5Gemma's own key spelling begins `model.encoder.…`, which *does* pass the prefix rule — so the
/// thing that actually keeps it out is that no such key exists in the root checkpoint's target set,
/// because T5Gemma is a **different file with a different backend that has no plan to consult**.
/// This case pins the refusal; the structural half is pinned by
/// `the_adapter_backend_serves_unplanned_keys_untouched` below.
#[test]
fn same_and_t5gemma_keys_are_refused_as_no_target_matched() {
    let dir = scratch("forbidden");
    let targets = adaptable_targets(&real_key_shapes());
    for forbidden in [
        // SAME — present in the very file the wrapper serves, and excluded by prefix.
        "pretransform.model.encoder.layers.0.mapping.weight",
        // T5Gemma's own spelling. It passes the `model.` prefix rule, and is still refused,
        // because it is not a key of the root checkpoint at all.
        "model.encoder.layers.0.self_attn.q_proj.weight",
        // A DiT key that simply does not exist — the ordinary key-mapping-mismatch case.
        "model.model.transformer.layers.99.cross_attn.to_q.weight",
    ] {
        let path = dir.join(format!("{}.safetensors", forbidden.replace('.', "_")));
        AdapterRecipe::new("lora", 2, 1.0)
            .factor(forbidden, "lora_A", vec![2, 4], fill(2, 4, 1.0))
            .factor(forbidden, "lora_B", vec![4, 2], fill(4, 2, 1.0))
            .write_native(&path);
        let message = plan_for(&[spec(&path, 1.0)], &targets, &Device::Cpu)
            .err()
            .unwrap_or_else(|| panic!("{forbidden} was accepted as an adapter target"))
            .to_string();
        assert!(
            message.contains("no_target_matched"),
            "{forbidden} was refused for the wrong reason: {message}"
        );
    }
}

/// The conditioner's one adaptable module is reachable, at its real shape.
///
/// sc-14548's carry-forward applies: a gate that only ever touches the DiT would be green on an
/// implementation that silently skipped the conditioner entirely, because the DiT dominates every
/// aggregate measurement.
#[test]
fn the_seconds_total_conditioner_linear_is_an_adapter_target() {
    let dir = scratch("conditioner");
    let key = "conditioner.conditioners.seconds_total.embedder.embedding.1.weight";
    let targets = adaptable_targets(&real_key_shapes());
    let (out_features, in_features, rank) = (768usize, 256usize, 2usize);

    let path = dir.join("cond.safetensors");
    recipe_for(
        AdapterType::DoraRows,
        key,
        out_features,
        in_features,
        rank,
        1.0,
        1.0,
    )
    .write_native(&path);

    let plan = plan_for(&[spec(&path, 1.0)], &targets, &Device::Cpu).expect("plan");
    assert_eq!(plan.targets().collect::<Vec<_>>(), vec![key]);

    let base = f32_tensor(
        fill(out_features, in_features, 1.0),
        (out_features, in_features),
    );
    let adapted = apply_adapter_ops(&base, plan.ops_for(key).unwrap()).expect("fold");
    assert_eq!(adapted.dims(), &[out_features, in_features]);
    assert_ne!(rows(&adapted), rows(&base));
}

/// Conv1d `[out, in, k]` is flattened to `[out, in*k]`, adapted, and restored **exactly**.
///
/// `k > 1` is used deliberately: both of SA3's real Conv1d targets have `k == 1`, where a wrong
/// flatten is invisible on shape alone. The delta is checked entrywise against the flattened
/// arithmetic, so a transposed restore fails.
#[test]
fn conv1d_targets_flatten_apply_and_restore_exactly() {
    let dir = scratch("conv1d");
    let key = "model.model.preprocess_conv.weight";
    let (out_channels, in_channels, kernel, rank) = (3usize, 2usize, 2usize, 1usize);
    let flat_in = in_channels * kernel;

    let path = dir.join("conv.safetensors");
    // A = [[1, 2, 3, 4]], B = [[1], [0], [-2]] ⇒ B@A rows: [1,2,3,4], [0,0,0,0], [-2,-4,-6,-8]
    AdapterRecipe::new("lora", rank, 1.0)
        .factor(key, "lora_A", vec![rank, flat_in], vec![1.0, 2.0, 3.0, 4.0])
        .factor(
            key,
            "lora_B",
            vec![out_channels, rank],
            vec![1.0, 0.0, -2.0],
        )
        .write_native(&path);

    let mut targets = BTreeMap::new();
    targets.insert(key.to_string(), vec![out_channels, in_channels, kernel]);
    let plan = plan_for(&[spec(&path, 1.0)], &targets, &Device::Cpu).expect("plan");

    let base_values: Vec<f32> = (0..out_channels * flat_in)
        .map(|v| v as f32 * 10.0)
        .collect();
    let base = Tensor::from_vec(
        base_values.clone(),
        (out_channels, in_channels, kernel),
        &Device::Cpu,
    )
    .unwrap();
    let adapted = apply_adapter_ops(&base, plan.ops_for(key).unwrap()).expect("fold");
    assert_eq!(
        adapted.dims(),
        &[out_channels, in_channels, kernel],
        "the Conv1d shape was not restored"
    );

    let delta = [
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        [-2.0, -4.0, -6.0, -8.0],
    ];
    let got = adapted.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (out, row) in delta.iter().enumerate().take(out_channels) {
        for (column, entry) in row.iter().enumerate().take(flat_in) {
            let index = out * flat_in + column;
            let want = base_values[index] + entry;
            assert!(
                (got[index] - want).abs() < 1e-4,
                "conv element [{out}][{column}]: {} vs {want}",
                got[index]
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 6. Formats and the pickle boundary
// ---------------------------------------------------------------------------------------------

/// ⚠ The scratch directory is deliberately **not** named "pickle".
///
/// It was, and that made this case vacuous: emptying `PICKLE_EXTENSIONS` entirely leaves every
/// container falling through to the generic "not a recognized adapter" refusal — which is still an
/// error, and whose message interpolates `path.display()`, so the old assertion
/// `message.contains("pickle")` was satisfied by the **temp directory's own name**. That mutation
/// ran GREEN. The assertions now match the specific sentences only the pickle branch emits, and no
/// fixture path can supply them.
#[test]
fn every_pickle_container_extension_is_refused_with_a_typed_message() {
    // ⚠ The scratch tag is deliberately NOT "pickle" — see the doc comment above.
    let dir = scratch("containers");
    for extension in ["ckpt", "pt", "pth", "bin", "PT", "Ckpt"] {
        let path = dir.join(format!("adapter.{extension}"));
        std::fs::write(&path, b"\x80\x04not-really-a-pickle").unwrap();
        let error =
            AdapterSource::classify(&path).expect_err(&format!(".{extension} must be refused"));
        let message = error.to_string();
        assert!(
            message.contains("is a Python pickle container"),
            ".{extension} was not refused by the pickle branch: {message}"
        );
        assert!(
            message.contains("Unpickling executes arbitrary code"),
            ".{extension} refusal does not state the security boundary: {message}"
        );
        assert!(
            message.contains("safetensors"),
            ".{extension} refusal does not name the accepted format: {message}"
        );
    }

    // An unrecognized extension is refused too, so the denylist is not an implicit allowlist.
    let odd = dir.join("adapter.weights");
    std::fs::write(&odd, b"anything").unwrap();
    assert!(AdapterSource::classify(&odd).is_err());

    // `svd_bases.pt` — the 1.27/3.84 GB startup cache shipped beside the `-base` checkpoints — is
    // refused by the same rule, which is why nothing has to special-case it.
    let bases = dir.join("svd_bases.pt");
    std::fs::write(&bases, b"pickle").unwrap();
    let message = AdapterSource::classify(&bases).unwrap_err().to_string();
    assert!(
        message.contains("is a Python pickle container"),
        "{message}"
    );
}

#[test]
fn a_peft_directory_round_trips_the_whole_family() {
    let dir = scratch("peft");
    let device = Device::Cpu;
    let (out_features, in_features, rank) = (4usize, 4usize, 2usize);
    let base = f32_tensor(
        fill(out_features, in_features, 3.0),
        (out_features, in_features),
    );
    let targets = small_targets(TARGET, out_features, in_features);

    for kind in AdapterType::ALL {
        let recipe = recipe_for(kind, TARGET, out_features, in_features, rank, 2.0, 4.0);
        let peft_dir = dir.join(format!("peft-{}", kind.as_str()));
        recipe.write_peft(&peft_dir);
        let native = dir.join(format!("native-{}.safetensors", kind.as_str()));
        recipe.write_native(&native);

        assert_eq!(
            AdapterSource::classify(&peft_dir).unwrap(),
            AdapterSource::PeftDirectory
        );
        let loaded = load_adapter(&spec(&peft_dir, 1.0), &device)
            .unwrap_or_else(|error| panic!("{}: {error}", kind.as_str()));
        assert_eq!(loaded.kind, kind);
        assert_eq!(loaded.rank, rank);

        // Native and PEFT spellings of the same adapter must fold identically — otherwise one of
        // the two readers is silently transposing or dropping a factor.
        let from_peft = plan_for(&[spec(&peft_dir, 1.0)], &targets, &device).expect("peft plan");
        let from_native = plan_for(&[spec(&native, 1.0)], &targets, &device).expect("native plan");
        assert_eq!(
            rows(&apply_adapter_ops(&base, from_peft.ops_for(TARGET).unwrap()).unwrap()),
            rows(&apply_adapter_ops(&base, from_native.ops_for(TARGET).unwrap()).unwrap()),
            "{} folds differently from a PEFT directory than from a native file",
            kind.as_str()
        );
    }

    // A directory missing either sidecar is not a PEFT adapter.
    let partial = dir.join("partial");
    std::fs::create_dir_all(&partial).unwrap();
    std::fs::write(partial.join("adapter_config.json"), "{}").unwrap();
    assert!(AdapterSource::classify(&partial).is_err());
}

/// Every declaration a PEFT `adapter_config.json` is trusted for is **required**, not defaulted.
///
/// `r` and `lora_alpha` set `effective_scale = (alpha / rank) * scale`. Silently defaulting either
/// one — `r` to some constant, `lora_alpha` to `1.0` — produces an adapter that loads, folds, and
/// applies the **wrong strength**: a plausible-looking result that is quietly not what was trained,
/// which is strictly worse than a refusal because nothing downstream can detect it. Both defaults
/// are green without this case.
#[test]
fn a_malformed_peft_config_is_refused_rather_than_defaulted() {
    let dir = scratch("peft-config");
    let (out_features, in_features, rank) = (4usize, 4usize, 2usize);

    // One valid PEFT directory, re-used by rewriting only `adapter_config.json` each time, so each
    // case differs from a known-good load in exactly the config field under test.
    let peft_dir = dir.join("peft");
    recipe_for(
        AdapterType::Lora,
        TARGET,
        out_features,
        in_features,
        rank,
        4.0,
        1.0,
    )
    .write_peft(&peft_dir);
    let config_path = peft_dir.join("adapter_config.json");
    let good = std::fs::read_to_string(&config_path).expect("baseline config");

    // The baseline itself must load, or every refusal below is unattributable.
    let loaded = load_adapter(&spec(&peft_dir, 1.0), &Device::Cpu).expect("baseline peft");
    assert_eq!(loaded.rank, rank);
    assert_eq!(loaded.alpha, 4.0);

    for (case, config) in [
        // `r` absent — must not fall back to a constant rank.
        ("missing r", serde_json::json!({"lora_alpha": 4.0})),
        // `lora_alpha` absent — must not fall back to 1.0.
        ("missing lora_alpha", serde_json::json!({"r": 2})),
        (
            "non-finite lora_alpha",
            serde_json::json!({"r": 2, "lora_alpha": "NaN"}),
        ),
        (
            "null lora_alpha",
            serde_json::json!({"r": 2, "lora_alpha": null}),
        ),
        // Parses as a JSON number but overflows f64/f32 to infinity — the one spelling that reaches
        // the explicit `is_finite` guard, since JSON has no NaN/Infinity literal.
        (
            "overflowing lora_alpha",
            serde_json::from_str::<serde_json::Value>(r#"{"r": 2, "lora_alpha": 1e400}"#)
                .unwrap_or_else(|_| serde_json::json!({"r": 2, "lora_alpha": 1e39})),
        ),
        // A zero rank would make `alpha / rank` a division by zero.
        ("r: 0", serde_json::json!({"r": 0, "lora_alpha": 4.0})),
        (
            "negative r",
            serde_json::json!({"r": -2, "lora_alpha": 4.0}),
        ),
        (
            "non-string target_modules entry",
            serde_json::json!({"r": 2, "lora_alpha": 4.0, "target_modules": ["to_q", 7]}),
        ),
        (
            "non-array target_modules",
            serde_json::json!({"r": 2, "lora_alpha": 4.0, "target_modules": 7}),
        ),
        // `exclude_modules` goes through the same `json_filter`, and its call site was ungated
        // while `target_modules`' was RED — one arm gated, the sibling dark, the same class of hole
        // this whole case exists to close. Dropping this filter is worse than dropping `include`:
        // an adapter whose `exclude_modules` is silently defaulted to empty folds into precisely
        // the modules its author named as must-not-touch, and every surface reports success.
        (
            "non-array exclude_modules",
            serde_json::json!({"r": 2, "lora_alpha": 4.0, "exclude_modules": 7}),
        ),
    ] {
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap())
            .expect("write config");
        assert!(
            load_adapter(&spec(&peft_dir, 1.0), &Device::Cpu).is_err(),
            "a PEFT config with `{case}` was accepted; the declaration was defaulted instead of \
             refused, so the adapter folds at a strength it was not trained at"
        );
    }

    // The config is read before it is parsed, and both steps are refusals rather than fallbacks.
    // These two cases are written as raw bytes because neither is expressible as a
    // `serde_json::Value`: the first is not UTF-8 and the second is not JSON.
    for (case, bytes) in [
        // Not UTF-8 — `read_to_string` itself fails. Reachable without touching permissions, so
        // this runs the same way on every platform and as any user.
        ("invalid UTF-8", vec![0xffu8, 0xfe, 0x00, 0x7b]),
        // Valid UTF-8, not JSON. A silently defaulted config here is the same wrong-strength fold
        // as a missing `r`, with the added twist that the file looks present and populated.
        ("not JSON at all", b"r = 2, lora_alpha = 4.0".to_vec()),
        // JSON, but not an object — `get("r")` returns `None` on every non-object value.
        ("a JSON array", b"[2, 4.0]".to_vec()),
    ] {
        std::fs::write(&config_path, &bytes).expect("write raw config");
        assert!(
            load_adapter(&spec(&peft_dir, 1.0), &Device::Cpu).is_err(),
            "a PEFT config that is `{case}` was accepted; the declarations were defaulted instead \
             of refused, so the adapter folds at a strength it was not trained at"
        );
    }

    // Restoring the baseline loads again, proving the refusals came from the edited field and not
    // from a directory this test left broken.
    std::fs::write(&config_path, &good).expect("restore config");
    load_adapter(&spec(&peft_dir, 1.0), &Device::Cpu).expect("restored baseline");
}

/// A native adapter's `{target}.{index}.{factor}` index segment must be an integer.
///
/// The index is what separates the target stem from the factor name, so a non-integer there means
/// the stem was split at the wrong dot and the target key is being guessed. Accepting `…x.lora_A`
/// would silently retarget the module. The check is green to delete without this case.
#[test]
fn a_native_adapter_index_segment_must_be_an_integer() {
    let dir = scratch("native-index");
    let path = dir.join("bad-index.safetensors");

    // Written by hand rather than through `AdapterRecipe`, which always emits the well-formed `.0.`
    // index — the malformed spelling is the entire point of the fixture.
    let values: Vec<f32> = fill(2, 4, 1.0);
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let stem = TARGET.strip_suffix(".weight").unwrap();
    let name = format!("{stem}.x.lora_A");
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 4], &bytes)
        .expect("view");
    let mut metadata = HashMap::new();
    metadata.insert("adapter_type".to_string(), "lora".to_string());
    metadata.insert("rank".to_string(), "2".to_string());
    metadata.insert("alpha".to_string(), "1".to_string());
    safetensors::serialize_to_file(vec![(name, view)], &Some(metadata), &path).expect("write");

    let error = load_adapter(&spec(&path, 1.0), &Device::Cpu)
        .expect_err("a non-integer adapter index must be refused");
    let text = error.to_string();
    assert!(
        text.contains("is not an integer"),
        "the refusal does not name the index segment: {text}"
    );
}

// ---------------------------------------------------------------------------------------------
// 6b. Tensor-level rules that BOTH readers enforce, driven over both readers
// ---------------------------------------------------------------------------------------------

/// Write a safetensors file from explicitly spelled tensor names.
///
/// [`AdapterRecipe`] always emits well-formed names; these fixtures exist to emit malformed ones,
/// so the name is an input rather than something derived from a target key.
fn write_raw_safetensors(
    path: &Path,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
    metadata: Option<HashMap<String, String>>,
) {
    let owned: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, shape, values)| {
            (
                name.clone(),
                shape.clone(),
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            )
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = owned
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .expect("view"),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, &metadata, path).expect("write safetensors");
}

/// One tensor-level refusal rule, spelled once per reader.
///
/// `load_native` and `load_peft` are two independent parsers over two different on-disk spellings,
/// and each duplicates the same tensor-level rules. A case written against one reader says nothing
/// about the other's copy, so every rule here carries both spellings and is asserted twice.
struct TwoReaderRefusal {
    /// What the rule is, for the failure message.
    rule: &'static str,
    /// The declared adapter type, written to native metadata and to the PEFT config alike. `"lora"`
    /// for every row whose subject is a tensor rather than the declaration itself.
    declared: &'static str,
    /// Tensors for the native single-file layout, `"{stem}.{index}.{factor}"`.
    native: Vec<(String, Vec<usize>, Vec<f32>)>,
    /// Tensors for the PEFT directory layout, `"base_model.model.{stem}.{factor}.weight"`.
    peft: Vec<(String, Vec<usize>, Vec<f32>)>,
    /// A substring both readers' refusals must carry.
    expect: &'static str,
}

impl TwoReaderRefusal {
    /// Assert the rule is enforced by `load_native` **and** by `load_peft`.
    fn assert_refused_by_both_readers(&self, dir: &Path) {
        let device = Device::Cpu;
        let slug: String = self
            .rule
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("adapter_type".to_string(), self.declared.to_string());
        metadata.insert("rank".to_string(), "2".to_string());
        metadata.insert("alpha".to_string(), "1".to_string());
        let native = dir.join(format!("native-{slug}.safetensors"));
        write_raw_safetensors(&native, &self.native, Some(metadata));
        self.assert_one(&native, "native", &device);

        let peft = dir.join(format!("peft-{slug}"));
        std::fs::create_dir_all(&peft).expect("peft dir");
        write_raw_safetensors(&peft.join("adapter_model.safetensors"), &self.peft, None);
        std::fs::write(
            peft.join("adapter_config.json"),
            serde_json::json!({"r": 2, "lora_alpha": 1.0, "sa3_adapter_type": self.declared})
                .to_string(),
        )
        .expect("write peft config");
        self.assert_one(&peft, "PEFT", &device);
    }

    fn assert_one(&self, path: &Path, reader: &str, device: &Device) {
        match load_adapter(&spec(path, 1.0), device) {
            Ok(_) => panic!(
                "the {reader} reader ACCEPTED an adapter violating `{}`; that reader's copy of the \
                 rule is not enforced",
                self.rule
            ),
            Err(error) => {
                let text = error.to_string();
                assert!(
                    text.contains(self.expect),
                    "the {reader} reader refused `{}` but not for that reason (expected {:?}): \
                     {text}",
                    self.rule,
                    self.expect
                );
            }
        }
    }
}

/// The refusals `load_native` and `load_peft` both carry, asserted on both.
///
/// Round-2 review found three of these — non-finite factors, unknown factor segments, duplicate
/// factors — gated on `load_native` only, because every fixture that feeds tensors went through
/// `write_native` and the single PEFT tensor case is a happy-path round trip. Deleting each of
/// `load_peft`'s three copies left the suite green, which meant a PEFT adapter carrying NaN
/// factors would load, plan and fold into the checkpoint.
///
/// The table shape is the actual fix. Three one-off PEFT cases would have gated those three and
/// left the next shared rule in the same state — and there were three more: a name with no
/// separator, a name missing its middle segment, and an unrecognized declared adapter type, that
/// last one dark on **both** readers. A shared rule is now one row with two spellings, and adding
/// a row asserts it on both parsers by construction.
#[test]
fn both_readers_enforce_the_same_tensor_level_rules() {
    let dir = scratch("two-reader");
    let stem = TARGET.strip_suffix(".weight").expect("target stem");

    let a = (vec![2usize, 4usize], fill(2, 4, 1.0));
    let b = (vec![4usize, 2usize], fill(4, 2, 2.0));
    let mut nan = fill(2, 4, 1.0);
    nan[3] = f32::NAN;

    for case in [
        // A NaN factor. Folded into the checkpoint it poisons every later sample, and nothing
        // downstream re-checks, so the refusal has to happen at load.
        TwoReaderRefusal {
            rule: "a non-finite factor value",
            declared: "lora",
            native: vec![
                (format!("{stem}.0.lora_A"), a.0.clone(), nan.clone()),
                (format!("{stem}.0.lora_B"), b.0.clone(), b.1.clone()),
            ],
            peft: vec![
                (
                    format!("base_model.model.{stem}.lora_A.weight"),
                    a.0.clone(),
                    nan.clone(),
                ),
                (
                    format!("base_model.model.{stem}.lora_B.weight"),
                    b.0.clone(),
                    b.1.clone(),
                ),
            ],
            expect: "non-finite value",
        },
        // An unrecognized factor segment. `finish_modules` only removes the five known names, so
        // an accepted `lora_Q` is silently dropped in a release build and "every adapter tensor is
        // consumed exactly once" stops holding.
        TwoReaderRefusal {
            rule: "an unknown factor segment",
            declared: "lora",
            native: vec![
                (format!("{stem}.0.lora_A"), a.0.clone(), a.1.clone()),
                (format!("{stem}.0.lora_B"), b.0.clone(), b.1.clone()),
                (format!("{stem}.0.lora_Q"), vec![2, 2], fill(2, 2, 3.0)),
            ],
            peft: vec![
                (
                    format!("base_model.model.{stem}.lora_A.weight"),
                    a.0.clone(),
                    a.1.clone(),
                ),
                (
                    format!("base_model.model.{stem}.lora_B.weight"),
                    b.0.clone(),
                    b.1.clone(),
                ),
                (
                    format!("base_model.model.{stem}.lora_Q.weight"),
                    vec![2, 2],
                    fill(2, 2, 3.0),
                ),
            ],
            expect: "lora_Q",
        },
        // Two tensors collapsing to the same `(target, factor)`. Native reaches it through two
        // adapter indices; PEFT reaches it because the `base_model.model.` prefix is optional, so
        // the prefixed and bare spellings of one module strip to the same stem. Whichever tensor
        // lost the race would be silently discarded.
        TwoReaderRefusal {
            rule: "a duplicated factor for one target",
            declared: "lora",
            native: vec![
                (format!("{stem}.0.lora_A"), a.0.clone(), a.1.clone()),
                (format!("{stem}.1.lora_A"), a.0.clone(), fill(2, 4, 5.0)),
                (format!("{stem}.0.lora_B"), b.0.clone(), b.1.clone()),
            ],
            peft: vec![
                (
                    format!("base_model.model.{stem}.lora_A.weight"),
                    a.0.clone(),
                    a.1.clone(),
                ),
                (
                    format!("{stem}.lora_A.weight"),
                    a.0.clone(),
                    fill(2, 4, 5.0),
                ),
                (
                    format!("base_model.model.{stem}.lora_B.weight"),
                    b.0.clone(),
                    b.1.clone(),
                ),
            ],
            expect: "more than once",
        },
        // A name with no dot at all — nothing can be split out of it, so the target key would be
        // whatever the parser guessed.
        TwoReaderRefusal {
            rule: "a tensor name with no separator",
            declared: "lora",
            native: vec![("loraA".to_string(), a.0.clone(), a.1.clone())],
            peft: vec![("loraA".to_string(), a.0.clone(), a.1.clone())],
            expect: "has unparseable tensor",
        },
        // A name that splits once and then runs out: native loses its index segment, PEFT loses its
        // factor segment. Either way the remaining text is not a target stem.
        //
        // The PEFT spelling is `lora_A.weight` rather than the more obvious `to_q.weight` on
        // purpose. Without the refusal, `to_q.weight` still fails — the salvaged `to_q` is not a
        // known factor — so it does not discriminate. `lora_A.weight` salvages a name that *is* a
        // known factor, so dropping the refusal makes the file load, under the nonsense target key
        // `lora_A.weight`.
        //
        // `expect` is the bare word "segment" because the two readers word this refusal
        // differently — native says "without an adapter index segment", PEFT "with no factor
        // segment" — and "segment" is their longest shared substring. It is looser than the other
        // rows: the adjacent `NATIVE_FACTORS` refusal also contains "segment", so neutering the
        // rule under test leaves this row failing on the substring rather than on acceptance. That
        // is over-strict, not under-strict — the row is still RED — but tightening it needs a
        // per-reader expected substring rather than one shared field, which is a change to the
        // table's shape and is deliberately not made here.
        TwoReaderRefusal {
            rule: "a tensor name missing its middle segment",
            declared: "lora",
            native: vec![("to_q.lora_A".to_string(), a.0.clone(), a.1.clone())],
            peft: vec![
                ("lora_A.weight".to_string(), a.0.clone(), a.1.clone()),
                ("lora_B.weight".to_string(), b.0.clone(), b.1.clone()),
            ],
            expect: "segment",
        },
        // No adapter modules at all. An empty adapter that loads is an adapter that folds nothing
        // while every surface reports it as applied.
        TwoReaderRefusal {
            rule: "an adapter with no modules",
            declared: "lora",
            native: Vec::new(),
            peft: Vec::new(),
            expect: "no adapter modules",
        },
        // An adapter type neither reader implements. The declaration is spelled differently on each
        // side — native metadata `adapter_type`, PEFT config `sa3_adapter_type` — but both hand it
        // to the same `resolve_adapter_type`, and that refusal was ungated on *both* readers. A
        // silent fallback to classic LoRA folds a file the crate does not implement and reports
        // success.
        TwoReaderRefusal {
            rule: "an unrecognized declared adapter type",
            declared: "lokr",
            native: vec![
                (format!("{stem}.0.lora_A"), a.0.clone(), a.1.clone()),
                (format!("{stem}.0.lora_B"), b.0.clone(), b.1.clone()),
            ],
            peft: vec![
                (
                    format!("base_model.model.{stem}.lora_A.weight"),
                    a.0.clone(),
                    a.1.clone(),
                ),
                (
                    format!("base_model.model.{stem}.lora_B.weight"),
                    b.0.clone(),
                    b.1.clone(),
                ),
            ],
            expect: "unknown Stable Audio 3 adapter type",
        },
    ] {
        case.assert_refused_by_both_readers(&dir);
    }

    // The control: the same two spellings, well formed, load from both readers — so every refusal
    // above is attributable to the rule under test and not to a fixture this case cannot write.
    let good = AdapterRecipe::new("lora", 2, 1.0)
        .factor(TARGET, "lora_A", a.0.clone(), a.1.clone())
        .factor(TARGET, "lora_B", b.0.clone(), b.1.clone());
    let good_native = dir.join("control.safetensors");
    good.write_native(&good_native);
    load_adapter(&spec(&good_native, 1.0), &Device::Cpu).expect("native control");
    let good_peft = dir.join("control-peft");
    good.write_peft(&good_peft);
    load_adapter(&spec(&good_peft, 1.0), &Device::Cpu).expect("peft control");
}

/// The native reader's safetensors metadata declarations are required, not defaulted.
///
/// The PEFT sibling of this case is `a_malformed_peft_config_is_refused_rather_than_defaulted`.
/// The two readers do **not** agree here and are not supposed to: `sa3_adapter_type` is optional in
/// a PEFT config and defaults to `"lora"`, because that is what upstream PEFT writes, while a
/// native file is written by this crate and declares its type. That asymmetry is exactly why the
/// native side needs its own case — the shared table above cannot express a rule only one reader
/// has, and dropping `adapter_type` from a native file was green without this.
#[test]
fn native_metadata_declarations_are_required_not_defaulted() {
    let dir = scratch("native-metadata");
    let stem = TARGET.strip_suffix(".weight").expect("target stem");
    let path = dir.join("metadata.safetensors");
    let tensors = vec![
        (format!("{stem}.0.lora_A"), vec![2, 4], fill(2, 4, 1.0)),
        (format!("{stem}.0.lora_B"), vec![4, 2], fill(4, 2, 2.0)),
    ];
    let good: Vec<(&str, &str)> = vec![
        ("adapter_type", "lora"),
        ("rank", "2"),
        ("alpha", "1"),
        ("include", ""),
        ("exclude", ""),
    ];
    let write = |entries: &[(&str, &str)]| {
        let metadata: HashMap<String, String> = entries
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        write_raw_safetensors(&path, &tensors, Some(metadata));
    };

    // The baseline loads, so every refusal below is attributable to the edited key.
    write(&good);
    load_adapter(&spec(&path, 1.0), &Device::Cpu).expect("baseline native metadata");

    // Each of the three required keys, dropped one at a time. A default for any of them produces an
    // adapter that loads and folds at a strength or in a form it was not trained at.
    for dropped in ["adapter_type", "rank", "alpha"] {
        let entries: Vec<(&str, &str)> = good
            .iter()
            .copied()
            .filter(|(key, _)| *key != dropped)
            .collect();
        write(&entries);
        assert!(
            load_adapter(&spec(&path, 1.0), &Device::Cpu).is_err(),
            "a native adapter with no `{dropped}` metadata was accepted; the declaration was \
             defaulted instead of refused"
        );
    }

    // A present-but-unparseable `rank` or `alpha` is refused too, not silently coerced.
    //
    // `alpha` carries two rows because it has two guards and only one of them is reachable by a
    // non-numeric spelling. `"loud"` dies at `parse`, so it leaves the `is_finite` guard behind it
    // ungated — deleting that guard was green. `"inf"` parses *successfully* as an `f32`, which is
    // the only way to reach it: without the guard a native adapter declaring `alpha = inf` loads
    // and folds `inf` into every targeted weight. The PEFT sibling of this row is the `1e400` case
    // in `a_malformed_peft_config_is_refused_rather_than_defaulted`, which had to be spelled as an
    // overflowing literal because JSON has no infinity token; native metadata is plain strings, so
    // here the word itself is enough.
    //
    // The `include`/`exclude` rows gate the two `parse_filter` call sites. Both were ungated:
    // `.unwrap_or_default()` on either one was green, and a dropped filter is not inert — an
    // adapter with an unreadable `exclude` folds into exactly the modules its author excluded, and
    // one with an unreadable `include` folds into every target instead of the chosen few. An
    // inverted bracket range is malformed for both keys, so one spelling covers both call sites.
    for (key, value) in [
        ("rank", "two"),
        ("alpha", "loud"),
        ("alpha", "inf"),
        ("rank", "0"),
        ("include", "layers.[3-0]"),
        ("exclude", "layers.[3-0]"),
    ] {
        let entries: Vec<(&str, &str)> = good
            .iter()
            .copied()
            .map(|(k, v)| if k == key { (k, value) } else { (k, v) })
            .collect();
        write(&entries);
        assert!(
            load_adapter(&spec(&path, 1.0), &Device::Cpu).is_err(),
            "a native adapter declaring `{key}` as {value:?} was accepted"
        );
    }

    // `__metadata__` present but not an object at all. `write_raw_safetensors` cannot spell this —
    // the safetensors writer always emits an object — so the file is assembled by hand: an 8-byte
    // little-endian header length, the header JSON, then the tensor data.
    //
    // This row asserts the refusal *message*, not merely that a refusal happened, because the two
    // are not the same thing here. Reading the non-object as empty metadata does not make the file
    // load: `read_safetensors_metadata` runs *before* `MmapedSafetensors::new` (`adapters.rs:947`
    // then `:950`), and safetensors deserializes `__metadata__` as a map, so the neutered form dies
    // at the header parse — `invalid JSON in header: invalid type: integer 7, expected a map` — and
    // never reaches the `adapter_type` check at all. Either way an `is_err()` row would hold and
    // this site would stay dark. The message is what separates the rule under test from the one
    // standing behind it.
    //
    // An earlier revision of this comment, of `b3976a26`'s commit message and of the PR body all
    // named that downstream refusal as the missing `adapter_type`. It is not — the mmap open sits
    // between. Same defect class as the completeness claim this branch already corrected: a
    // measured-sounding step that does not reproduce. The conclusion is unchanged.
    {
        let odd = dir.join("non-object-metadata.safetensors");
        let values = fill(2, 4, 1.0);
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), serde_json::json!(7));
        header.insert(
            format!("{stem}.0.lora_A"),
            serde_json::json!({
                "dtype": "F32",
                "shape": [2, 4],
                "data_offsets": [0, data.len()],
            }),
        );
        let header = serde_json::to_vec(&serde_json::Value::Object(header)).expect("header bytes");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&data);
        std::fs::write(&odd, &bytes).expect("write non-object metadata");
        let error = load_adapter(&spec(&odd, 1.0), &Device::Cpu)
            .expect_err("a native adapter whose `__metadata__` is not an object was accepted")
            .to_string();
        assert!(
            error.contains("non-object safetensors __metadata__"),
            "the file was refused, but one step later than the rule under test: a non-object \
             `__metadata__` was read as empty metadata instead of refused. {error}"
        );
    }

    // Restoring the baseline loads again, proving the refusals came from the edited key.
    write(&good);
    load_adapter(&spec(&path, 1.0), &Device::Cpu).expect("restored native metadata");
}

// ---------------------------------------------------------------------------------------------
// 7. Filters, matching, and "every tensor consumed exactly once"
// ---------------------------------------------------------------------------------------------

#[test]
fn bracket_ranges_expand_and_malformed_ones_are_refused() {
    assert_eq!(
        expand_bracket_ranges("layers.[0-3].cross_attn").unwrap(),
        vec![
            "layers.0.cross_attn",
            "layers.1.cross_attn",
            "layers.2.cross_attn",
            "layers.3.cross_attn"
        ]
    );
    assert_eq!(
        expand_bracket_ranges("to_q").unwrap(),
        vec!["to_q".to_string()]
    );
    assert_eq!(
        expand_bracket_ranges("a.[0-1].b.[2-3]").unwrap(),
        vec!["a.0.b.2", "a.0.b.3", "a.1.b.2", "a.1.b.3"]
    );
    assert_eq!(expand_bracket_ranges("layers.[5-5].x").unwrap().len(), 1);
    for malformed in [
        "layers.[3-0]",
        "layers.[0-",
        "layers.0]",
        "layers.[a-b]",
        "layers.[03]",
    ] {
        assert!(
            expand_bracket_ranges(malformed).is_err(),
            "{malformed} should be refused rather than treated as a literal substring"
        );
    }
}

/// `include` narrows the candidate set, and a module outside the adapter's **own** filters is a
/// contradiction in the file rather than a caller's subsetting choice — so it fails loudly.
#[test]
fn include_and_exclude_filters_are_honored_and_orphaned_modules_fail_loudly() {
    let dir = scratch("filters");
    let device = Device::Cpu;
    let mut targets = BTreeMap::new();
    for layer in 0..4 {
        targets.insert(
            format!("model.model.transformer.layers.{layer}.cross_attn.to_q.weight"),
            vec![4, 4],
        );
        targets.insert(
            format!("model.model.transformer.layers.{layer}.ff.ff.2.weight"),
            vec![4, 4],
        );
    }

    // An adapter covering layers 0-1's `to_q`, declaring exactly that with a bracket range.
    let good = dir.join("scoped.safetensors");
    let mut recipe = AdapterRecipe::new("lora", 2, 1.0).include("layers.[0-1].cross_attn");
    for layer in 0..2 {
        let key = format!("model.model.transformer.layers.{layer}.cross_attn.to_q.weight");
        recipe = recipe
            .factor(&key, "lora_A", vec![2, 4], fill(2, 4, layer as f32))
            .factor(&key, "lora_B", vec![4, 2], fill(4, 2, layer as f32 + 1.0));
    }
    recipe.write_native(&good);
    let plan = plan_for(&[spec(&good, 1.0)], &targets, &device).expect("scoped plan");
    assert_eq!(plan.op_count(), 2);
    assert_eq!(
        plan.targets().collect::<Vec<_>>(),
        vec![
            "model.model.transformer.layers.0.cross_attn.to_q.weight",
            "model.model.transformer.layers.1.cross_attn.to_q.weight",
        ]
    );

    // ⚠ `include` needs its OWN orphan case. Neutering the include branch of `passes_filters`
    // — `if false { return false; }` — passed every other case in this file, because an adapter
    // whose modules all sit inside its declared `include` behaves identically whether the filter
    // narrows the candidate set or not: the plan is driven by the file's modules, so `include` is
    // observable only when a module falls OUTSIDE it. That is this case.
    let outside_include = dir.join("outside-include.safetensors");
    let mut recipe = AdapterRecipe::new("lora", 2, 1.0).include("layers.0.cross_attn");
    for layer in 0..2 {
        let key = format!("model.model.transformer.layers.{layer}.cross_attn.to_q.weight");
        recipe = recipe
            .factor(&key, "lora_A", vec![2, 4], fill(2, 4, layer as f32))
            .factor(&key, "lora_B", vec![4, 2], fill(4, 2, layer as f32 + 1.0));
    }
    recipe.write_native(&outside_include);
    let message = plan_for(&[spec(&outside_include, 1.0)], &targets, &device)
        .expect_err("a module outside the adapter's own `include` must be refused")
        .to_string();
    assert!(message.contains("no_target_matched"), "{message}");
    assert!(
        message.contains("layers.1.cross_attn.to_q.weight"),
        "the refusal does not name the orphaned module: {message}"
    );

    // The same adapter, but its own `exclude` orphans layer 1's tensors. They would never be
    // consumed, so the load fails rather than half-applying.
    let orphaned = dir.join("orphaned.safetensors");
    let mut recipe = AdapterRecipe::new("lora", 2, 1.0).exclude("layers.1.");
    for layer in 0..2 {
        let key = format!("model.model.transformer.layers.{layer}.cross_attn.to_q.weight");
        recipe = recipe
            .factor(&key, "lora_A", vec![2, 4], fill(2, 4, layer as f32))
            .factor(&key, "lora_B", vec![4, 2], fill(4, 2, layer as f32 + 1.0));
    }
    recipe.write_native(&orphaned);
    let error = plan_for(&[spec(&orphaned, 1.0)], &targets, &device)
        .expect_err("an orphaned module must be refused");
    assert!(
        error.to_string().contains("no_target_matched"),
        "unexpected message: {error}"
    );
}

/// Every tensor in an adapter must be consumed exactly once. Four ways that can break, four
/// refusals.
#[test]
fn every_adapter_tensor_must_be_consumed_exactly_once() {
    let dir = scratch("consumed");
    let device = Device::Cpu;
    let targets = small_targets(TARGET, 4, 4);

    // (a) A tensor whose trailing segment is not a known factor.
    let stray = dir.join("stray.safetensors");
    AdapterRecipe::new("lora", 2, 1.0)
        .factor(TARGET, "lora_A", vec![2, 4], fill(2, 4, 1.0))
        .factor(TARGET, "lora_B", vec![4, 2], fill(4, 2, 2.0))
        .factor(TARGET, "lora_Q", vec![2, 2], fill(2, 2, 3.0))
        .write_native(&stray);
    let message = load_adapter(&spec(&stray, 1.0), &device)
        .unwrap_err()
        .to_string();
    assert!(message.contains("lora_Q"), "{message}");

    // (b) A factor the declared type cannot use.
    let unusable = dir.join("unusable.safetensors");
    AdapterRecipe::new("lora", 2, 1.0)
        .factor(TARGET, "lora_A", vec![2, 4], fill(2, 4, 1.0))
        .factor(TARGET, "lora_B", vec![4, 2], fill(4, 2, 2.0))
        .factor(TARGET, "magnitude_r", vec![4], vec![1.0; 4])
        .write_native(&unusable);
    let message = plan_for(&[spec(&unusable, 1.0)], &targets, &device)
        .unwrap_err()
        .to_string();
    assert!(message.contains("magnitude_r"), "{message}");

    // (c) A required factor missing.
    let incomplete = dir.join("incomplete.safetensors");
    AdapterRecipe::new("dora-rows", 2, 1.0)
        .factor(TARGET, "lora_A", vec![2, 4], fill(2, 4, 1.0))
        .factor(TARGET, "lora_B", vec![4, 2], fill(4, 2, 2.0))
        .write_native(&incomplete);
    let message = plan_for(&[spec(&incomplete, 1.0)], &targets, &device)
        .unwrap_err()
        .to_string();
    assert!(message.contains("magnitude_r"), "{message}");

    // (d) An `-xs` file that also ships lora_A/lora_B — its bases come from the weight, so those
    //     tensors have no consumer.
    let overspecified = dir.join("overspecified.safetensors");
    AdapterRecipe::new("lora-xs", 2, 1.0)
        .factor(TARGET, "lora_M", vec![2, 2], fill(2, 2, 1.0))
        .factor(TARGET, "lora_A", vec![2, 4], fill(2, 4, 1.0))
        .factor(TARGET, "lora_B", vec![4, 2], fill(4, 2, 2.0))
        .write_native(&overspecified);
    let message = plan_for(&[spec(&overspecified, 1.0)], &targets, &device)
        .unwrap_err()
        .to_string();
    assert!(message.contains("lora_A"), "{message}");
}

/// Shape and rank disagreements are refused before any weight is touched.
#[test]
fn shape_and_rank_mismatches_are_refused_at_plan_time() {
    let dir = scratch("shapes");
    let device = Device::Cpu;
    let targets = small_targets(TARGET, 4, 6);

    // `lora_A` at the wrong in-features — the classic "adapter trained against the other
    // checkpoint" case.
    let wrong = dir.join("wrong-in.safetensors");
    AdapterRecipe::new("lora", 2, 1.0)
        .factor(TARGET, "lora_A", vec![2, 5], fill(2, 5, 1.0))
        .factor(TARGET, "lora_B", vec![4, 2], fill(4, 2, 2.0))
        .write_native(&wrong);
    let message = plan_for(&[spec(&wrong, 1.0)], &targets, &device)
        .unwrap_err()
        .to_string();
    assert!(message.contains("lora_A"), "{message}");

    // A/B swapped: `lora_A` shaped like `lora_B` and vice versa.
    let swapped = dir.join("swapped.safetensors");
    AdapterRecipe::new("lora", 2, 1.0)
        .factor(TARGET, "lora_A", vec![4, 2], fill(4, 2, 1.0))
        .factor(TARGET, "lora_B", vec![2, 6], fill(2, 6, 2.0))
        .write_native(&swapped);
    assert!(plan_for(&[spec(&swapped, 1.0)], &targets, &device).is_err());

    // A rank larger than the target's smaller dimension.
    let overrank = dir.join("overrank.safetensors");
    AdapterRecipe::new("lora", 9, 1.0)
        .factor(TARGET, "lora_A", vec![9, 6], fill(9, 6, 1.0))
        .factor(TARGET, "lora_B", vec![4, 9], fill(4, 9, 2.0))
        .write_native(&overrank);
    let message = plan_for(&[spec(&overrank, 1.0)], &targets, &device)
        .unwrap_err()
        .to_string();
    assert!(message.contains("rank"), "{message}");
}

/// Non-finite factors and non-finite metadata are refused rather than folded into the checkpoint.
#[test]
fn non_finite_adapter_values_are_refused() {
    let dir = scratch("finite");
    let device = Device::Cpu;

    let nan = dir.join("nan.safetensors");
    let mut values = fill(2, 4, 1.0);
    values[3] = f32::NAN;
    AdapterRecipe::new("lora", 2, 1.0)
        .factor(TARGET, "lora_A", vec![2, 4], values)
        .factor(TARGET, "lora_B", vec![4, 2], fill(4, 2, 2.0))
        .write_native(&nan);
    assert!(load_adapter(&spec(&nan, 1.0), &device).is_err());

    let good = dir.join("good.safetensors");
    AdapterRecipe::new("lora", 2, 1.0)
        .factor(TARGET, "lora_A", vec![2, 4], fill(2, 4, 1.0))
        .factor(TARGET, "lora_B", vec![4, 2], fill(4, 2, 2.0))
        .write_native(&good);
    // The control: the same fixture without the NaN loads, so the case above is discriminating.
    assert!(load_adapter(&spec(&good, 1.0), &device).is_ok());
    // A non-finite *request* scale is refused too.
    assert!(load_adapter(&spec(&good, f32::NAN), &device).is_err());
    assert!(load_adapter(&spec(&good, f32::INFINITY), &device).is_err());
}

// ---------------------------------------------------------------------------------------------
// 8. Contract surface: what the provider advertises and what it refuses
// ---------------------------------------------------------------------------------------------

/// All six registered ids advertise `supports_lora`, and none advertises `supports_lokr`.
#[test]
fn every_registered_variant_advertises_lora_and_refuses_lokr() {
    let variants = [
        model::Variant::SmallMusic,
        model::Variant::SmallSfx,
        model::Variant::Medium,
        model::Variant::SmallMusicBase,
        model::Variant::SmallSfxBase,
        model::Variant::MediumBase,
    ];
    for variant in variants {
        let descriptor = model::descriptor_for(variant);
        assert!(
            descriptor.capabilities.supports_lora,
            "{} does not advertise LoRA",
            variant.model_id()
        );
        assert!(
            !descriptor.capabilities.supports_lokr,
            "{} advertises LoKr, which this family has no decomposition for",
            variant.model_id()
        );
    }

    let lokr = AdapterSpec::new(
        PathBuf::from("/nonexistent.safetensors"),
        1.0,
        AdapterKind::Lokr,
    );
    let message = validate_spec_shape(&lokr).unwrap_err().to_string();
    assert!(message.contains("Lokr"), "{message}");
}

/// Neither of the two model-specific knobs on the shared `AdapterSpec` is silently honored.
///
/// `pass_scales` is LTX-2.3's and `moe_expert` is Wan2.2's. Repurposing either would be a knob that
/// appears to work; ignoring either would be a knob that appears to work and does nothing. Both are
/// refused by name, and a spec carrying neither is accepted, so the case discriminates.
#[test]
fn ltx_and_wan_specific_adapter_knobs_are_refused_by_name() {
    let path = PathBuf::from("/nonexistent.safetensors");
    let plain = AdapterSpec::new(path.clone(), 1.0, AdapterKind::Lora);
    assert!(validate_spec_shape(&plain).is_ok());

    let ltx = plain.clone().with_pass_scales(vec![1.0, 0.5]);
    let message = validate_spec_shape(&ltx).unwrap_err().to_string();
    assert!(message.contains("pass_scales"), "{message}");

    let wan = plain.with_moe_expert(candle_audio::gen_core::MoeExpert::High);
    let message = validate_spec_shape(&wan).unwrap_err().to_string();
    assert!(message.contains("moe_expert"), "{message}");
}

// ---------------------------------------------------------------------------------------------
// 9. The backend seam
// ---------------------------------------------------------------------------------------------

/// The wrapper adapts exactly the keys the plan names and serves every other key **untouched**.
///
/// This is the structural half of "T5Gemma is never adapted": the backend consults a plan, and the
/// text encoder is built on a *different* backend with no plan at all. Here the same property is
/// gated directly — an unplanned key comes back bit-identical, and a planned one does not.
#[test]
fn the_adapter_backend_serves_unplanned_keys_untouched() {
    use candle_nn::var_builder::SimpleBackend;

    let dir = scratch("backend");
    let device = Device::Cpu;
    let (out_features, in_features, rank) = (4usize, 4usize, 2usize);
    let targets = small_targets(TARGET, out_features, in_features);
    let path = dir.join("backend.safetensors");
    recipe_for(
        AdapterType::Bora,
        TARGET,
        out_features,
        in_features,
        rank,
        2.0,
        1.0,
    )
    .write_native(&path);
    let plan = plan_for(&[spec(&path, 1.0)], &targets, &device).expect("plan");

    let adapted_values = fill(out_features, in_features, 1.0);
    let bystander_values = fill(out_features, in_features, 9.0);
    let mut inner: HashMap<String, Tensor> = HashMap::new();
    inner.insert(
        TARGET.to_string(),
        f32_tensor(adapted_values.clone(), (out_features, in_features)),
    );
    inner.insert(
        "model.model.transformer.layers.9.ff.ff.2.weight".to_string(),
        f32_tensor(bystander_values.clone(), (out_features, in_features)),
    );
    inner.insert(
        "pretransform.model.encoder.layers.0.mapping.weight".to_string(),
        f32_tensor(bystander_values.clone(), (out_features, in_features)),
    );

    let backend = AdapterBackend::new(Box::new(inner), plan);
    let shape = candle_audio::candle_core::Shape::from((out_features, in_features));
    let fetch = |name: &str| {
        SimpleBackend::get(
            &backend,
            shape.clone(),
            name,
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .expect(name)
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
    };

    assert_ne!(
        fetch(TARGET),
        adapted_values,
        "the planned key was served unadapted"
    );
    assert_eq!(
        fetch("model.model.transformer.layers.9.ff.ff.2.weight"),
        bystander_values,
        "an unplanned DiT key was mutated"
    );
    assert_eq!(
        fetch("pretransform.model.encoder.layers.0.mapping.weight"),
        bystander_values,
        "a SAME key was mutated"
    );

    // The same contract through `get_unchecked`, the shape-free entry point.
    //
    // `AdapterBackend` overrides *both* `SimpleBackend` methods, and no case in this file reached
    // the second one: replacing its whole body with `match None::<&[AdapterOp]>` — i.e. serving
    // every key unadapted — was green. The override is kept rather than dropped because which of
    // the two `VarBuilder` routes through is candle's decision, not this crate's, and a backend
    // whose two entry points disagree would adapt or not adapt depending on how a tensor happened
    // to be requested. Gating it is what makes keeping it meaningful.
    let fetch_unchecked = |name: &str| {
        SimpleBackend::get_unchecked(&backend, name, DType::F32, &device)
            .expect(name)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    assert_eq!(
        fetch_unchecked(TARGET),
        fetch(TARGET),
        "get_unchecked and get disagree on a planned key"
    );
    assert_ne!(
        fetch_unchecked(TARGET),
        adapted_values,
        "get_unchecked served the planned key unadapted"
    );
    assert_eq!(
        fetch_unchecked("model.model.transformer.layers.9.ff.ff.2.weight"),
        bystander_values,
        "get_unchecked mutated an unplanned DiT key"
    );
    assert_eq!(
        fetch_unchecked("pretransform.model.encoder.layers.0.mapping.weight"),
        bystander_values,
        "get_unchecked mutated a SAME key"
    );
}

/// The load-time target set is derived from a real safetensors header, not from an in-test map.
///
/// Closes the gap between `adaptable_targets` (unit-tested above on a hand-built map) and the
/// shipped `safetensors_shapes` reader that feeds it in production: a header parser that lost the
/// shapes would make every adapter fail — or, worse, make a Conv1d look 2-D.
#[test]
fn the_target_set_is_read_from_a_real_safetensors_header() {
    let dir = scratch("header");
    let path = dir.join("model.safetensors");

    let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (key, shape) in real_key_shapes() {
        let count: usize = shape.iter().product();
        let bytes = (0..count)
            .flat_map(|index| (index as f32).to_le_bytes())
            .collect::<Vec<u8>>();
        owned.push((key, shape, bytes));
    }
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = owned
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .expect("view"),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, &None, &path).expect("write checkpoint");

    let shapes = safetensors_shapes(&path).expect("read header");
    assert_eq!(
        shapes,
        real_key_shapes(),
        "the header reader lost information"
    );
    let targets = adaptable_targets(&shapes);
    assert_eq!(targets.len(), 8);
    assert_eq!(
        targets.get("model.model.preprocess_conv.weight"),
        Some(&vec![256usize, 256, 1]),
        "the Conv1d target lost its kernel dimension"
    );
}

/// An empty plan is the identity, and nothing installs it.
#[test]
fn an_empty_plan_is_empty_and_folds_nothing() {
    let plan = AdapterPlan::default();
    assert!(plan.is_empty());
    assert_eq!(plan.op_count(), 0);
    assert_eq!(plan.targets().count(), 0);
    assert!(plan.ops_for(TARGET).is_none());

    let base = f32_tensor(fill(3, 3, 1.0), (3, 3));
    let same = apply_adapter_ops(&base, &[]).unwrap();
    assert_eq!(
        base.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        same.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    );

    // A `LoadSpec` with no adapters resolves to the empty plan without reading anything.
    let targets = small_targets(TARGET, 4, 4);
    let resolved = plan_for(&[], &targets, &Device::Cpu).expect("no adapters");
    assert!(resolved.is_empty());
}

// ---------------------------------------------------------------------------------------------
// 10. Real weights
//
// ⚠ These do **not** validate a real adapter artifact. None exists — see the module header and
// sc-15347. What they validate is the *path*: that a plan resolved from a real checkpoint's header
// reaches the real backend (including Metal's packed-buffer variant, which is a different
// `SimpleBackend` from CPU/CUDA's mmap), that the fold changes the rendered audio, and — the
// correctly-signed half — that `scale == 0.0` renders **byte-identical** audio to a request with no
// adapters at all.
//
// The signs are chosen deliberately. "Adapted audio differs from the base" is satisfied by a
// *misapplied* adapter too, usually by more, so it is used only as a liveness check and never as
// the gate. The gates are the two exact ones: zero-scale ⇒ equality, adapter A vs adapter B ⇒
// inequality. Neither can be passed by tuning a threshold and neither improves when the feature
// breaks.
// ---------------------------------------------------------------------------------------------

struct RealCase {
    variant: model::Variant,
    env: &'static str,
    prompt: &'static str,
}

const REAL_CASES: &[RealCase] = &[
    RealCase {
        variant: model::Variant::SmallMusic,
        env: "SA3_SMALL_MUSIC_SNAPSHOT",
        prompt: "warm cinematic post-rock with bowed strings and restrained drums",
    },
    RealCase {
        variant: model::Variant::SmallSfx,
        env: "SA3_SMALL_SFX_SNAPSHOT",
        prompt: "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style",
    },
    RealCase {
        variant: model::Variant::Medium,
        env: "SA3_MEDIUM_SNAPSHOT",
        prompt: "Meditative lo-fi ambient piano jazz, soft acoustic drum kit",
    },
    RealCase {
        variant: model::Variant::SmallMusicBase,
        env: "SA3_SMALL_MUSIC_BASE_SNAPSHOT",
        prompt: "A beautiful piano arpeggio grows into a grand cinematic climax",
    },
    RealCase {
        variant: model::Variant::SmallSfxBase,
        env: "SA3_SMALL_SFX_BASE_SNAPSHOT",
        prompt: "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style",
    },
    RealCase {
        variant: model::Variant::MediumBase,
        env: "SA3_MEDIUM_BASE_SNAPSHOT",
        prompt: "Meditative lo-fi ambient piano jazz, soft acoustic drum kit",
    },
];

fn snapshot_root(env: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(env).unwrap_or_else(|_| panic!("set {env} to the pinned immutable snapshot")),
    )
}

/// Pick real adapter targets out of a real checkpoint header: the conditioner Linear plus the two
/// smallest DiT Linears, so the synthetic adapter is genuinely spread across **both** target
/// families rather than exercising the DiT alone.
fn real_targets(root: &Path, count: usize) -> Vec<(String, usize, usize)> {
    let shapes = safetensors_shapes(&root.join("model.safetensors")).expect("checkpoint header");
    let targets = adaptable_targets(&shapes);
    let conditioner: Vec<(String, usize, usize)> = targets
        .iter()
        .filter(|(key, _)| key.starts_with("conditioner."))
        .map(|(key, shape)| (key.clone(), shape[0], shape[1..].iter().product()))
        .collect();
    assert!(
        !conditioner.is_empty(),
        "the checkpoint has no adaptable conditioner module; the conditioner half is untested"
    );
    let mut dit: Vec<(String, usize, usize)> = targets
        .iter()
        .filter(|(key, _)| key.starts_with("model."))
        .map(|(key, shape)| (key.clone(), shape[0], shape[1..].iter().product()))
        .collect();
    dit.sort_by_key(|(key, out, input)| (out * input, key.clone()));
    let mut picked = conditioner;
    picked.extend(dit.into_iter().take(count));
    picked
}

fn write_real_adapter(
    path: &Path,
    kind: AdapterType,
    targets: &[(String, usize, usize)],
    seed: f32,
) {
    let rank = 2usize;
    let mut recipe = AdapterRecipe::new(kind.as_str(), rank, 4.0);
    for (key, out_features, in_features) in targets {
        let single = recipe_for(kind, key, *out_features, *in_features, rank, 4.0, seed);
        for (target, factors) in single.modules {
            for (factor, (shape, values)) in factors {
                recipe = recipe.factor(&target, &factor, shape, values);
            }
        }
    }
    recipe.write_native(path);
}

fn render(generator: &dyn candle_audio::gen_core::Generator, prompt: &str, seed: u64) -> Vec<f32> {
    let request = candle_audio::gen_core::GenerationRequest {
        prompt: prompt.into(),
        seed: Some(seed),
        steps: Some(4),
        audio: Some(candle_audio::gen_core::AudioParams {
            target_duration: Some(4.0),
            sample_rate: Some(44_100),
            ..Default::default()
        }),
        ..Default::default()
    };
    match generator.generate(&request, &mut |_| {}).expect("generate") {
        candle_audio::gen_core::GenerationOutput::Audio(track) => track.samples,
        other => panic!("expected audio, got {other:?}"),
    }
}

/// The two exactly-signed real-weight gates, on all six pinned checkpoints.
#[test]
#[ignore = "requires all six pinned immutable snapshots; set SA3_*_SNAPSHOT"]
fn real_adapters_change_the_render_and_a_zero_scale_one_changes_nothing() {
    let dir = scratch("real");
    for case in REAL_CASES {
        let root = snapshot_root(case.env);
        let targets = real_targets(&root, 3);
        let id = case.variant.model_id();

        let first = dir.join(format!("{id}-a.safetensors"));
        let second = dir.join(format!("{id}-b.safetensors"));
        write_real_adapter(&first, AdapterType::Lora, &targets, 1.0);
        write_real_adapter(&second, AdapterType::Lora, &targets, 11.0);

        let load = |adapters: Vec<AdapterSpec>| {
            let mut load_spec = candle_audio::gen_core::LoadSpec::new(
                candle_audio::gen_core::WeightsSource::Dir(root.clone()),
            );
            load_spec.adapters = adapters;
            model::load_variant(case.variant, &load_spec)
                .unwrap_or_else(|error| panic!("{id}: {error}"))
        };

        let plain = render(&load(Vec::new()), case.prompt, 7);
        let inert = render(&load(vec![spec(&first, 0.0)]), case.prompt, 7);
        let adapted = render(&load(vec![spec(&first, 1.0)]), case.prompt, 7);
        let other = render(&load(vec![spec(&second, 1.0)]), case.prompt, 7);

        assert!(
            plain.iter().all(|value| value.is_finite()) && plain.iter().any(|value| *value != 0.0),
            "{id}: the un-adapted render is not usable audio"
        );
        // GATE 1 — exactly zero difference. A fold that ran at all breaks this.
        assert_eq!(
            plain, inert,
            "{id}: a scale-0.0 adapter changed the render; the no-mutation fast path did not take"
        );
        // GATE 2 — two different adapters cannot produce the same audio.
        assert_ne!(
            adapted, other,
            "{id}: two adapters with different weights rendered identically; the fold is not \
             reading the adapter"
        );
        // Liveness only, deliberately NOT the gate: a misapplied adapter also differs from the base.
        assert_ne!(adapted, plain, "{id}: the adapter had no effect at all");
        assert!(
            adapted.iter().all(|value| value.is_finite()),
            "{id}: the adapted render emitted non-finite PCM"
        );
        assert_eq!(adapted.len(), plain.len());
    }
}

/// Stacking order is observable on real weights, for a type that does not commute.
#[test]
#[ignore = "requires all six pinned immutable snapshots; set SA3_*_SNAPSHOT"]
fn real_stacked_dora_adapters_are_order_dependent() {
    let dir = scratch("real-order");
    for case in REAL_CASES {
        let root = snapshot_root(case.env);
        let targets = real_targets(&root, 2);
        let id = case.variant.model_id();
        let first = dir.join(format!("{id}-x.safetensors"));
        let second = dir.join(format!("{id}-y.safetensors"));
        write_real_adapter(&first, AdapterType::DoraRows, &targets, 2.0);
        write_real_adapter(&second, AdapterType::DoraRows, &targets, 21.0);

        let load = |adapters: Vec<AdapterSpec>| {
            let mut load_spec = candle_audio::gen_core::LoadSpec::new(
                candle_audio::gen_core::WeightsSource::Dir(root.clone()),
            );
            load_spec.adapters = adapters;
            model::load_variant(case.variant, &load_spec).expect("load")
        };
        // Each adapter keeps its own strength across both orderings, so the only thing that moves
        // is the order.
        let forward = render(
            &load(vec![spec(&first, 0.9), spec(&second, 1.4)]),
            case.prompt,
            7,
        );
        let reverse = render(
            &load(vec![spec(&second, 1.4), spec(&first, 0.9)]),
            case.prompt,
            7,
        );
        assert_ne!(
            forward, reverse,
            "{id}: reversing a two-deep DoRA stack produced identical audio; the fold is not \
             sequential"
        );
    }
}

/// The `-xs` half, on real weights, deliberately scoped to the conditioner's single `[768, 256]`
/// Linear.
///
/// That scope is a **cost** decision, stated rather than hidden: the deterministic host SVD in
/// [`candle_audio_stable_audio_3::svd`] takes ~1.9 s for a `768x256` target and ~113 s for a
/// `1024x1024` one on this machine, so an `-xs` adapter covering the DiT's attention stack is a
/// multi-hour cold start. The math is identical at either scale; only the wall clock is not. See
/// the follow-up filed on sc-14550.
#[test]
#[ignore = "requires all six pinned immutable snapshots; set SA3_*_SNAPSHOT"]
fn real_lora_xs_folds_through_the_host_svd() {
    let dir = scratch("real-xs");
    for case in REAL_CASES {
        let root = snapshot_root(case.env);
        let id = case.variant.model_id();
        let targets: Vec<(String, usize, usize)> = real_targets(&root, 0);
        assert_eq!(
            targets.len(),
            1,
            "{id}: expected exactly the conditioner target"
        );

        let path = dir.join(format!("{id}-xs.safetensors"));
        write_real_adapter(&path, AdapterType::LoraXs, &targets, 3.0);

        let load = |adapters: Vec<AdapterSpec>| {
            let mut load_spec = candle_audio::gen_core::LoadSpec::new(
                candle_audio::gen_core::WeightsSource::Dir(root.clone()),
            );
            load_spec.adapters = adapters;
            model::load_variant(case.variant, &load_spec).expect("load")
        };
        let plain = render(&load(Vec::new()), case.prompt, 7);
        let inert = render(&load(vec![spec(&path, 0.0)]), case.prompt, 7);
        let adapted = render(&load(vec![spec(&path, 1.0)]), case.prompt, 7);
        assert_eq!(
            plain, inert,
            "{id}: a scale-0.0 -xs adapter changed the render"
        );
        assert_ne!(
            adapted, plain,
            "{id}: the -xs fold on the conditioner Linear had no effect on the render"
        );
        assert!(adapted.iter().all(|value| value.is_finite()));
    }
}

/// A key-mapping mismatch is refused by **`load_variant` itself**, not by the first `generate`.
///
/// This is the story's acceptance bullet — "adapter key-mapping mismatches fail loudly at load
/// time, never silently no-op" — and it is the one gate for the `resolve_adapter_plan` call in
/// `load_variant`. Deleting that call is green in the weight-free lane and green in every other
/// real-weight case here, because none of them ever loads an adapter that does not resolve: this
/// case is the only one that constructs the mismatch.
///
/// **What is observable, stated precisely.** `StableAudio3Generator` builds its pipeline lazily and
/// exposes no residency accessor, so "no pipeline was built" cannot be read off the object. It does
/// not need to be: the assertion is that `load_variant` returns `Err`, so **no generator exists at
/// all**, and a pipeline that is only ever constructed from a generator therefore cannot have been
/// built. Under the mutation the call is deleted, `load_variant` returns `Ok`, and the mismatch
/// surfaces from `generate` instead — which is exactly the failure the bullet exists to prevent,
/// and which flips this assertion.
///
/// The valid-adapter control on the same snapshot is what makes the refusal attributable to the
/// adapter rather than to the checkpoint, the layout or the identity check.
#[test]
#[ignore = "requires all six pinned immutable snapshots; set SA3_*_SNAPSHOT"]
fn a_key_mismatched_adapter_is_refused_at_load_variant_not_at_first_generate() {
    let dir = scratch("real-mismatch");
    // A well-formed adapter naming a layer index the DiT does not have. Everything about the file
    // is valid — type, rank, alpha, finite factors, consistent shapes — so the *only* thing that can
    // reject it is matching its key against the checkpoint's adaptable target set.
    const ABSENT: &str = "model.model.transformer.layers.99.cross_attn.to_q.weight";

    for case in REAL_CASES {
        let root = snapshot_root(case.env);
        let id = case.variant.model_id();

        let bogus = dir.join(format!("{id}-absent.safetensors"));
        write_real_adapter(
            &bogus,
            AdapterType::Lora,
            &[(ABSENT.to_string(), 1024, 1024)],
            5.0,
        );

        let load = |adapters: Vec<AdapterSpec>| {
            let mut load_spec = candle_audio::gen_core::LoadSpec::new(
                candle_audio::gen_core::WeightsSource::Dir(root.clone()),
            );
            load_spec.adapters = adapters;
            model::load_variant(case.variant, &load_spec)
        };

        // THE GATE — the refusal happens here, at load, with no generator produced.
        let error = match load(vec![spec(&bogus, 1.0)]) {
            Err(error) => error,
            Ok(_) => panic!(
                "{id}: load_variant accepted an adapter naming {ABSENT:?}, a key the checkpoint \
                 does not have. The mismatch is now deferred to the first generate."
            ),
        };
        assert!(
            matches!(error, candle_audio::gen_core::Error::Unsupported(_)),
            "{id}: expected a typed Unsupported refusal, got {error:?}"
        );
        let text = error.to_string();
        assert!(
            text.contains("no_target_matched"),
            "{id}: the refusal does not name no_target_matched: {text}"
        );
        assert!(
            text.contains(ABSENT),
            "{id}: the refusal does not name the offending key: {text}"
        );

        // A zero-scale mismatch is refused identically: validation is not what `scale == 0.0` skips.
        assert!(
            load(vec![spec(&bogus, 0.0)]).is_err(),
            "{id}: a scale-0.0 adapter skipped key validation"
        );

        // CONTROL — a valid adapter on the same snapshot loads, so the refusal above is a statement
        // about the adapter's keys and not about the checkpoint.
        let valid = dir.join(format!("{id}-valid.safetensors"));
        write_real_adapter(&valid, AdapterType::Lora, &real_targets(&root, 1), 5.0);
        load(vec![spec(&valid, 1.0)])
            .unwrap_or_else(|error| panic!("{id}: the valid control adapter was refused: {error}"));
    }
}
