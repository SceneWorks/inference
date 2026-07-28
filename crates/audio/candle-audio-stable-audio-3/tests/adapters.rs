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
    AdapterSource, AdapterType, ADAPTABLE_PREFIXES,
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
