//! The ComfyUI-namespace conversion: every alpha spelling against its diffusers twin, the alpha
//! division across the qkv unfuse, and the weights-free conversion gates.

use super::*;

// ─── sc-19443: the ComfyUI key space ───────────────────────────────────────────────────────────

/// **Where a ComfyUI export stamps the alpha of its fused `attn.qkv_proj`.**
///
/// The fixture generator used to emit the in-band `.alpha` tensor and nothing else, which made the
/// twin-equivalence gate — a well-built gate — span exactly **one of four** spellings. The other
/// three routed around the conversion's block-diagonal `÷3` and folded attention 3× too strong with
/// no error, and the gate could not see any of them. The generator is parameterized over this so
/// every arm below runs four times.
///
/// The `__metadata__` spelling is not a hypothetical: it is what **every published lightx2v file**
/// uses, and none of them carries an in-band `.alpha` at all. On this lane the PEFT blob is a second
/// bypass, because `apply_one_lora` honors it too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AlphaSpelling {
    /// A per-target `.alpha` tensor beside the factors — the kohya / ComfyUI in-band convention,
    /// and the only spelling the original fixture generator could write.
    InBand,
    /// A PEFT `lora_adapter_metadata` JSON blob in the top-level `__metadata__` (sc-5513).
    PeftBlob,
    /// A bare top-level `__metadata__["alpha"]` string — the lightx2v spelling.
    TopLevelMetadata,
    /// No alpha anywhere; resolution must land on `DEFAULT_LORA_ALPHA` **and still divide it**.
    Absent,
}

/// Every spelling, so an arm cannot quietly cover three of four.
pub(super) const ALPHA_SPELLINGS: [AlphaSpelling; 4] = [
    AlphaSpelling::InBand,
    AlphaSpelling::PeftBlob,
    AlphaSpelling::TopLevelMetadata,
    AlphaSpelling::Absent,
];

/// The alpha the fused `qkv_proj` declares, in whichever spelling is under test.
///
/// **48, not the published 24.** `24/3 = 8` is `DEFAULT_LORA_ALPHA`, so the published pairing cannot
/// distinguish "divided the resolved alpha" from "dropped the alpha and defaulted". `48/3 = 16`
/// distinguishes them.
pub(super) const FUSED_ALPHA: f32 = 48.0;

/// The in-band alpha the twin's `mlp.fc1` carries, in **every** arm.
///
/// `fc1` is not fused, so its alpha is never divided and its spelling is not what is under test. It
/// is pinned in-band — where it outranks every file-level source — so that varying the *qkv*
/// spelling changes exactly one thing. Non-default, and different from every qkv value, so a
/// conversion that crossed the two alphas over is visible rather than symmetric.
pub(super) const FC1_ALPHA: f32 = 32.0;

/// What the converted per-target `.alpha` **must** be: the resolved fused alpha, divided by three
/// exactly when the block-diagonal un-fuse divided the rank by three.
///
/// Written out here rather than read back from the converter, so this is an independent statement of
/// the rule and not a restatement of the implementation.
pub(super) fn expected_target_alpha(spelling: AlphaSpelling, block_diagonal: bool) -> f32 {
    let resolved = match spelling {
        AlphaSpelling::Absent => DEFAULT_LORA_ALPHA,
        _ => FUSED_ALPHA,
    };
    if block_diagonal {
        resolved / 3.0
    } else {
        resolved
    }
}

/// How a ComfyUI twin declares its fused `attn.qkv_proj`: the alpha, **where** that alpha lives, and
/// which of the two legitimate fused shapes to write.
#[derive(Clone, Copy, Debug)]
pub(super) struct FusedQkvSpec {
    pub(super) alpha: f32,
    pub(super) spelling: AlphaSpelling,
    pub(super) block_diagonal: bool,
}

impl FusedQkvSpec {
    /// The standard fixture: [`FUSED_ALPHA`] in `spelling`, at the chosen fused shape.
    pub(super) fn new(spelling: AlphaSpelling, block_diagonal: bool) -> Self {
        Self {
            alpha: FUSED_ALPHA,
            spelling,
            block_diagonal,
        }
    }

    /// The same, at an explicit alpha — for the arms that pin the published `24 → 8` pairing.
    pub(super) fn at(alpha: f32, block_diagonal: bool) -> Self {
        Self {
            alpha,
            spelling: AlphaSpelling::InBand,
            block_diagonal,
        }
    }
}

/// Write a **ComfyUI** twin of one block's modules, built from the same factors a diffusers twin
/// would carry, so any difference in the folded result is the conversion's fault.
///
/// `fused.block_diagonal` selects which of the two legitimate fused forms to write;
/// `fused.spelling` selects **where the fused alpha lives**, which is the axis the equivalence gate
/// was blind along.
pub(super) fn write_comfyui_twin(
    dir: &Path,
    name: &str,
    cfg: &MiniMaxH3DitConfig,
    fused: FusedQkvSpec,
    qkv: &[(Array, Array); 3],
    fc1: (&Array, &Array),
) -> PathBuf {
    let FusedQkvSpec {
        alpha,
        spelling,
        block_diagonal,
    } = fused;
    let path = dir.join(name);
    let r = PUBLISHED_RANK;
    let out = cfg.inner_dim();
    let mut arrays: Vec<(String, Array)> = Vec::new();

    if block_diagonal {
        let a = concatenate_axis(&[qkv[0].0.clone(), qkv[1].0.clone(), qkv[2].0.clone()], 0)
            .expect("concat A");
        let rows: Vec<Array> = (0..3usize)
            .map(|i| {
                let parts: Vec<Array> = (0..3usize)
                    .map(|j| {
                        if i == j {
                            qkv[i].1.clone()
                        } else {
                            bf16(&Array::zeros::<f32>(&[out, r]).unwrap())
                        }
                    })
                    .collect();
                concatenate_axis(&parts, 1).expect("concat row")
            })
            .collect();
        let b = concatenate_axis(&rows, 0).expect("concat B");
        arrays.push(("blocks.0.attn.qkv_proj.lora_A.weight".into(), a));
        arrays.push(("blocks.0.attn.qkv_proj.lora_B.weight".into(), b));
    } else {
        arrays.push((
            "blocks.0.attn.qkv_proj.lora_A.weight".into(),
            qkv[0].0.clone(),
        ));
        let b = concatenate_axis(&[qkv[0].1.clone(), qkv[1].1.clone(), qkv[2].1.clone()], 0)
            .expect("concat B");
        arrays.push(("blocks.0.attn.qkv_proj.lora_B.weight".into(), b));
    }
    // The fused alpha, written into exactly ONE of the four places it can live.
    if spelling == AlphaSpelling::InBand {
        arrays.push((
            "blocks.0.attn.qkv_proj.alpha".into(),
            Array::from_slice(&[alpha], &[1]),
        ));
    }

    // `mlp.fc1` carries the SwiGLU halves the OTHER way round — `[gate | value]` where the DiT is
    // `[value | gate]` — so the twin's B is the diffusers B with its row halves swapped.
    let (fc1_a, fc1_b) = fc1;
    let rows = fc1_b.shape()[0];
    let half = rows / 2;
    let top = mlx_gen_minimax_h3::tensor::slice_axis(fc1_b, 0, 0, half).expect("top half");
    let bottom = mlx_gen_minimax_h3::tensor::slice_axis(fc1_b, 0, half, rows).expect("bottom half");
    let swapped = concatenate_axis(&[bottom, top], 0).expect("swap halves");
    arrays.push(("blocks.0.mlp.fc1.lora_A.weight".into(), fc1_a.clone()));
    arrays.push(("blocks.0.mlp.fc1.lora_B.weight".into(), swapped));
    // `fc1` is unfused, so its alpha is never divided — pinned in-band in every arm so the only
    // thing `spelling` varies is the fused module's.
    arrays.push((
        "blocks.0.mlp.fc1.alpha".into(),
        Array::from_slice(&[FC1_ALPHA], &[1]),
    ));

    let mut meta: HashMap<String, String> = [(
        "target_format".to_string(),
        "ComfyUI generic LoRA".to_string(),
    )]
    .into_iter()
    .collect();
    match spelling {
        // A contradictory file-level value makes the precedence claim non-vacuous: the per-target
        // tensor must win independently for qkv and fc1.
        AlphaSpelling::InBand => {
            meta.insert("alpha".to_string(), "3".to_string());
        }
        AlphaSpelling::Absent => {}
        // `r` equals the per-target factor rank on BOTH fused forms (a block-diagonal `[3r, in]` A
        // splits into three rank-`r` ones), so this blob is self-consistent and is not rejected by
        // the rank cross-check — it is the alpha, and only the alpha, that is under test.
        AlphaSpelling::PeftBlob => {
            meta.insert(
                "lora_adapter_metadata".to_string(),
                format!(r#"{{"lora_alpha": {alpha}, "r": {PUBLISHED_RANK}}}"#),
            );
        }
        AlphaSpelling::TopLevelMetadata => {
            meta.insert("alpha".to_string(), alpha.to_string());
        }
    }
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &path).expect("write the comfyui twin");
    path
}

/// Re-key the small raw-module fixture into the exact trainer spelling. Production validates the
/// full 50×4 census before this rewrite; this weights-light fixture isolates the numerical seam.
pub(super) fn trainer_keys_from_comfyui(w: &Weights) -> Weights {
    let mut trainer = Weights::empty();
    for source in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let target = source
            .replace("blocks.0.attn.qkv_proj", "lora_unet_blocks_0_attn_qkv_proj")
            .replace("blocks.0.mlp.fc1", "lora_unet_blocks_0_mlp_fc1")
            .replace(".lora_A.weight", ".lora_down.weight")
            .replace(".lora_B.weight", ".lora_up.weight");
        trainer.insert(target, w.require(&source).unwrap().clone());
    }
    trainer
}

/// Write the **diffusers** counterpart of [`write_comfyui_twin`]'s modules, from the same factors.
///
/// Every target carries an explicit in-band `.alpha`: `qkv_alpha` on q/k/v and [`FC1_ALPHA`] on the
/// feed-forward input. The header's `alpha` is deliberately a value **no target should ever reach**
/// — if the control itself started falling through to a file-level alpha, its residual would move
/// and the comparison would stop meaning what it claims.
pub(super) fn write_diffusers_twin(
    dir: &Path,
    name: &str,
    qkv_alpha: f32,
    qkv: &[(Array, Array); 3],
    fc1: (&Array, &Array),
) -> PathBuf {
    let path = dir.join(name);
    let mut arrays: Vec<(String, Array)> = Vec::new();
    for (i, n) in ["to_q", "to_k", "to_v"].iter().enumerate() {
        arrays.push((
            format!("transformer_blocks.0.attn.{n}.lora_A.default.weight"),
            qkv[i].0.clone(),
        ));
        arrays.push((
            format!("transformer_blocks.0.attn.{n}.lora_B.default.weight"),
            qkv[i].1.clone(),
        ));
        arrays.push((
            format!("transformer_blocks.0.attn.{n}.alpha"),
            Array::from_slice(&[qkv_alpha], &[1]),
        ));
    }
    arrays.push((
        "transformer_blocks.0.ff.net.0.proj.lora_A.default.weight".into(),
        fc1.0.clone(),
    ));
    arrays.push((
        "transformer_blocks.0.ff.net.0.proj.lora_B.default.weight".into(),
        fc1.1.clone(),
    ));
    arrays.push((
        "transformer_blocks.0.ff.net.0.proj.alpha".into(),
        Array::from_slice(&[FC1_ALPHA], &[1]),
    ));
    let meta: HashMap<String, String> = [("alpha".to_string(), "1".to_string())]
        .into_iter()
        .collect();
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &path).expect("write the diffusers twin");
    path
}

/// The factors both twins are built from — distinct per projection, so a conversion that mixes up
/// q/k/v is observable rather than symmetric.
pub(super) fn twin_factors(cfg: &MiniMaxH3DitConfig) -> ([(Array, Array); 3], (Array, Array)) {
    let r = PUBLISHED_RANK;
    let qkv = [
        (
            bf16(&tensor(&[r, cfg.hidden_size], 1.0)),
            bf16(&tensor(&[cfg.inner_dim(), r], 2.0)),
        ),
        (
            bf16(&tensor(&[r, cfg.hidden_size], 3.0)),
            bf16(&tensor(&[cfg.inner_dim(), r], 4.0)),
        ),
        (
            bf16(&tensor(&[r, cfg.hidden_size], 5.0)),
            bf16(&tensor(&[cfg.inner_dim(), r], 6.0)),
        ),
    ];
    let fc1 = (
        bf16(&tensor(&[r, cfg.hidden_size], 7.0)),
        bf16(&tensor(&[2 * cfg.ffn_dim, r], 8.0)),
    );
    (qkv, fc1)
}

/// The exact trainer spelling reaches independently constructed diffusers-shaped factors.
/// Q/K/V factors are position-distinct, FC1 halves are distinct, and qkv/fc1 alphas disagree, so a
/// wrong slice, either half-order swap, or crossed/default alpha independently makes this red.
#[test]
fn trainer_namespace_converts_to_independent_expected_factors_at_relative_max_abs() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let (qkv, fc1) = twin_factors(&cfg);
    let comfy_path = write_comfyui_twin(
        dir.path(),
        "trainer_source.safetensors",
        &cfg,
        FusedQkvSpec::new(AlphaSpelling::InBand, false),
        &qkv,
        (&fc1.0, &fc1.1),
    );
    let comfy = Weights::from_file(&comfy_path).expect("raw fixture");
    let trainer = trainer_keys_from_comfyui(&comfy);
    let (raw, alphas) =
        unflatten_minimax_h3_trainer_tensors(&trainer).expect("exact trainer unflatten");
    assert_eq!(alphas, vec![FC1_ALPHA, FUSED_ALPHA]);

    let got = convert_comfyui_key_space(&raw).expect("trainer conversion");
    assert_eq!(got.keys().count(), 12, "q/k/v and fc1 each carry A/B/alpha");
    for (index, name) in ["to_q", "to_k", "to_v"].iter().enumerate() {
        for (suffix, expected) in [
            ("lora_A.weight", &qkv[0].0),
            ("lora_B.weight", &qkv[index].1),
        ] {
            let key = format!("transformer_blocks.0.attn.{name}.{suffix}");
            let actual = got.require(&key).unwrap();
            let drift =
                max_abs(&subtract(actual, expected).unwrap()) / max_abs(expected).max(1e-12);
            assert!(
                drift <= 1e-6,
                "{key}: trainer conversion drifted at relative max-abs {drift:.3e}"
            );
        }
        let key = format!("transformer_blocks.0.attn.{name}.alpha");
        let expected = Array::from_slice(&[FUSED_ALPHA], &[1]);
        let actual = got.require(&key).unwrap();
        let drift = max_abs(&subtract(actual, &expected).unwrap()) / max_abs(&expected);
        assert!(
            drift <= 1e-6,
            "{key}: trainer/raw conversion drifted at relative max-abs {drift:.3e}"
        );
    }
    for (suffix, expected) in [("lora_A.weight", &fc1.0), ("lora_B.weight", &fc1.1)] {
        let key = format!("transformer_blocks.0.ff.net.0.proj.{suffix}");
        let actual = got.require(&key).unwrap();
        let drift = max_abs(&subtract(actual, expected).unwrap()) / max_abs(expected).max(1e-12);
        assert!(drift <= 1e-6, "{key}: relative max-abs {drift:.3e}");
    }
    let expected = Array::from_slice(&[FC1_ALPHA], &[1]);
    let drift = max_abs(
        &subtract(
            got.require("transformer_blocks.0.ff.net.0.proj.alpha")
                .unwrap(),
            &expected,
        )
        .unwrap(),
    ) / max_abs(&expected);
    assert!(
        drift <= 1e-6,
        "FC1 alpha drifted at relative max-abs {drift:.3e}"
    );
}

/// **A converted ComfyUI file folds to the SAME residual as its diffusers twin**, gated on relative
/// max-abs-diff.
///
/// That equivalence is the only honest gate for this conversion, and it is what makes each of the
/// three transforms individually load-bearing: un-fusing `qkv_proj` wrong mixes q/k/v, dropping the
/// alpha division folds 3× strong, and skipping the SwiGLU swap computes `w2(silu(value)·gate)` —
/// the sc-18740 defect, which shipped green at cosine 0.73–0.78. Cosine cannot see any of them.
///
/// **Both fused forms are covered.** Block-diagonal is the lightx2v twins' shape; shared-`A` is what
/// a LoRA trained natively on the fused module looks like, and it carries a DIFFERENT adapter (one
/// down factor shared by all three projections), so it gets its own twin.
///
/// **And all four alpha spellings are covered**, which is the sc-19443 review's finding. The gate
/// itself was sound; its fixture generator only ever wrote the in-band `.alpha` tensor, so the
/// `__metadata__`, PEFT-blob and no-alpha routes — the first of which is the ONLY spelling any
/// published file for this family uses — never reached the block-diagonal `÷3` and folded attention
/// 3× too strong at rel-max-abs `2.007e0`, installing `Ok`.
///
/// Each spelling is **its own `#[test]`** (see the generated arms below) rather than an inner loop,
/// so a regression names the route it broke instead of only the gate, and a mutation can be shown to
/// red one route while leaving the others green.
pub(super) fn assert_twin_equivalence(spelling: AlphaSpelling) {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let (qkv, fc1) = twin_factors(&cfg);
    let shared_qkv = [
        (qkv[0].0.clone(), qkv[0].1.clone()),
        (qkv[0].0.clone(), qkv[1].1.clone()),
        (qkv[0].0.clone(), qkv[2].1.clone()),
    ];

    for (block_diagonal, twin_qkv, form) in [
        (true, &qkv, "block-diagonal (the lightx2v twin shape)"),
        (false, &shared_qkv, "shared-A (natively fused training)"),
    ] {
        // The diffusers control declares, per target, exactly what the conversion must arrive at:
        // `FUSED_ALPHA/3` (or `DEFAULT_LORA_ALPHA/3`) on a block-diagonal un-fuse, and the undivided
        // value when the rank was not split.
        let want_alpha = expected_target_alpha(spelling, block_diagonal);
        let label = format!("{form} / {spelling:?} (per-target alpha {want_alpha})");
        let diffusers = write_diffusers_twin(
            dir.path(),
            &format!("twin_{block_diagonal}_{spelling:?}.safetensors"),
            want_alpha,
            twin_qkv,
            (&fc1.0, &fc1.1),
        );
        let comfy = write_comfyui_twin(
            dir.path(),
            &format!("comfy_{block_diagonal}_{spelling:?}.safetensors"),
            &cfg,
            FusedQkvSpec::new(spelling, block_diagonal),
            twin_qkv,
            (&fc1.0, &fc1.1),
        );

        for probe in [
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.to_k",
            "transformer_blocks.0.attn.to_v",
            "transformer_blocks.0.ff.net.0.proj",
        ] {
            let segs: Vec<&str> = probe.split('.').collect();
            let x = tensor(&[1, 4, cfg.hidden_size], 0.31);

            // The bare base is the same for both files, so it is computed once per probe.
            let mut base = tiny_dit(&cfg);
            let y0 = base
                .adaptable_mut(&segs)
                .expect("probe")
                .forward(&x)
                .expect("base forward");
            let residual = |file: &Path| -> Array {
                let mut adapted = tiny_dit(&cfg);
                apply_minimax_h3_adapters(&mut adapted, &[spec(file.to_path_buf(), 1.0)])
                    .expect("install");
                let y1 = adapted
                    .adaptable_mut(&segs)
                    .expect("probe")
                    .forward(&x)
                    .expect("adapted forward");
                subtract(&y1, &y0).expect("residual")
            };

            let want = residual(&diffusers);
            let got = residual(&comfy);
            assert!(
                max_abs(&want) > 1e-4,
                "{probe}: the twin's own residual must be non-trivial, else this is vacuous"
            );
            let drift = max_abs(&subtract(&got, &want).unwrap()) / max_abs(&want);
            // The **fold ratio** is the quantity the defect is stated in — an undivided alpha on a
            // block-diagonal split reads 3.007×, and the fix reads 1.000×. Printed alongside the
            // gated relative max-abs-diff (never cosine: a pure fold-ratio error is exactly a scale
            // error, and cosine is scale-invariant).
            let ratio = max_abs(&got) / max_abs(&want);
            println!(
                "[{label}] {probe}: fold ratio {ratio:.4}x, converted-vs-twin rel-max-abs = \
                 {drift:.3e}"
            );
            assert!(
                drift < 1e-2,
                "{label} / {probe}: a converted ComfyUI file must fold like its diffusers twin \
                 whatever spelling carried its alpha; got fold ratio {ratio:.4}x, rel-max-abs \
                 {drift:.3e}"
            );
        }
    }
    crate::model_inference_device::record_model_inference_device();
}

/// One `#[test]` per alpha spelling, for both the residual gate and the emitted-alpha gate.
///
/// The point of the split is diagnostic resolution: the sc-19443 blocker made exactly three of the
/// four routes wrong, and a single test covering all four can only say "the gate failed".
macro_rules! per_alpha_spelling {
    ($($name:ident, $chain:ident => $spelling:expr;)+) => {
        /// Exactly the spellings the arms below were generated for — cross-checked against
        /// [`ALPHA_SPELLINGS`] by `every_alpha_spelling_has_generated_arms`.
        const GENERATED_SPELLINGS: &[AlphaSpelling] = &[$($spelling),+];
        $(
            #[test]
            fn $name() {
                assert_twin_equivalence($spelling);
            }

            #[test]
            fn $chain() {
                assert_alpha_resolves_before_the_split($spelling);
            }
        )+
    };
}

per_alpha_spelling! {
    a_converted_comfyui_file_folds_like_its_diffusers_twin_in_band,
        the_in_band_alpha_resolves_before_the_qkv_split => AlphaSpelling::InBand;
    a_converted_comfyui_file_folds_like_its_diffusers_twin_peft_blob,
        the_peft_blob_alpha_resolves_before_the_qkv_split => AlphaSpelling::PeftBlob;
    a_converted_comfyui_file_folds_like_its_diffusers_twin_metadata,
        the_metadata_alpha_resolves_before_the_qkv_split => AlphaSpelling::TopLevelMetadata;
    a_converted_comfyui_file_folds_like_its_diffusers_twin_absent,
        an_absent_alpha_resolves_before_the_qkv_split => AlphaSpelling::Absent;
}

/// **Every `AlphaSpelling` gets generated arms.** A fifth spelling added to the enum breaks the
/// exhaustive `match` below until it is listed in [`ALPHA_SPELLINGS`], and the membership check then
/// forces it into the `per_alpha_spelling!` list too — so the gate cannot silently go back to
/// covering a subset, which is precisely how it missed the `__metadata__` route.
#[test]
fn every_alpha_spelling_has_generated_arms() {
    for s in ALPHA_SPELLINGS {
        match s {
            AlphaSpelling::InBand
            | AlphaSpelling::PeftBlob
            | AlphaSpelling::TopLevelMetadata
            | AlphaSpelling::Absent => {}
        }
        assert!(
            GENERATED_SPELLINGS.contains(&s),
            "{s:?} has no generated twin-equivalence arm"
        );
    }
    assert_eq!(GENERATED_SPELLINGS.len(), ALPHA_SPELLINGS.len());
}

/// **The alpha is resolved through the WHOLE chain before the qkv split, in every spelling.**
///
/// The residual gate above proves the end-to-end fold; this one names the number, so a failure says
/// *which* link broke rather than only that the folds disagree. It reads the `.alpha` the converter
/// emitted, for all four spellings × both fused forms, and it is the guard that would have caught
/// the sc-19443 blocker on its own.
///
/// Three claims per arm:
///
/// * a per-target `.alpha` is emitted **at all** — the old converter emitted none unless the file
///   carried an in-band tensor, and "no alpha" is exactly how the undivided file-level value leaked
///   back in downstream;
/// * its value is the resolved alpha divided by three **iff** the rank was;
/// * `fc1`'s alpha is untouched by the qkv spelling, so the division is not a blanket file-level
///   rescale that happens to land right on q/k/v.
pub(super) fn assert_alpha_resolves_before_the_split(spelling: AlphaSpelling) {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let (qkv, fc1) = twin_factors(&cfg);

    let read = |file: &PathBuf, key: &str| -> f32 {
        let w = Weights::from_file(file).unwrap();
        let converted = convert_comfyui_key_space(&w).unwrap();
        converted
            .require(key)
            .unwrap_or_else(|_| panic!("no converted alpha at {key}"))
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()[0]
    };

    for block_diagonal in [true, false] {
        let f = write_comfyui_twin(
            dir.path(),
            &format!("chain_{block_diagonal}_{spelling:?}.safetensors"),
            &cfg,
            FusedQkvSpec::new(spelling, block_diagonal),
            &qkv,
            (&fc1.0, &fc1.1),
        );
        let want = expected_target_alpha(spelling, block_diagonal);
        for n in ["to_q", "to_k", "to_v"] {
            let got = read(&f, &format!("transformer_blocks.0.attn.{n}.alpha"));
            assert_eq!(
                got, want,
                "{spelling:?} / block_diagonal={block_diagonal} / {n}: the fused alpha must be \
                 resolved through in-band → PEFT blob → __metadata__ → DEFAULT_LORA_ALPHA and THEN \
                 divided by three iff the rank was; got {got}, want {want}"
            );
        }
        // The unfused feed-forward alpha rides along untouched, in every arm.
        assert_eq!(
            read(&f, "transformer_blocks.0.ff.net.0.proj.alpha"),
            FC1_ALPHA,
            "{spelling:?}: an unfused module's alpha must not be divided"
        );
        // Non-vacuity: on the block-diagonal form no arm may land on the default alpha, or a
        // converter that dropped the alpha entirely would pass. (`FUSED_ALPHA/3 = 16` and
        // `DEFAULT_LORA_ALPHA/3 = 8/3`; neither is `DEFAULT_LORA_ALPHA`.)
        if block_diagonal {
            assert_ne!(
                want, DEFAULT_LORA_ALPHA,
                "{spelling:?}: this arm must not assert the default alpha"
            );
        }
    }
}

/// The published pairing, stated as the invariant the block-diagonal division exists to hold.
#[test]
fn the_alpha_division_holds_alpha_over_rank_fixed_across_the_unfuse() {
    assert_eq!(
        alpha_rank_fold(24.0, 3.0 * PUBLISHED_RANK as f32),
        alpha_rank_fold(8.0, PUBLISHED_RANK as f32),
        "the division exists to hold alpha/rank fixed across the un-fuse"
    );
    assert_eq!(
        alpha_rank_fold(FUSED_ALPHA / 3.0, PUBLISHED_RANK as f32),
        0.125
    );
    assert_ne!(
        alpha_rank_fold(FUSED_ALPHA / 3.0, PUBLISHED_RANK as f32),
        alpha_rank_fold(DEFAULT_LORA_ALPHA, PUBLISHED_RANK as f32),
        "FUSED_ALPHA is chosen so no arm asserts the default fold"
    );
}

/// **The alpha division tracks the rank split, and lands on a NON-default number.**
///
/// The published `24 → 8` is a trap for a test: `8` is also `DEFAULT_LORA_ALPHA`, so a converter
/// that dropped the alpha entirely would land on the same fold and pass. The first arm therefore
/// uses `48 → 16`, which is neither the default nor the fused value.
///
/// The shared-`A` arm is the other half of the claim: its rank is NOT divided, so dividing its alpha
/// would fold it three times too weak.
#[test]
fn the_alpha_division_tracks_the_rank_split_and_is_not_the_default() {
    let cfg = dit_fixture_config();
    let dir = tempfile::tempdir().expect("fixture dir");
    let (qkv, fc1) = twin_factors(&cfg);

    let read_alpha = |file: &PathBuf| -> f32 {
        let w = Weights::from_file(file).unwrap();
        let converted = convert_comfyui_key_space(&w).unwrap();
        converted
            .require("transformer_blocks.0.attn.to_q.alpha")
            .expect("the converted per-target alpha")
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()[0]
    };

    // Block-diagonal, a NON-default alpha: 48 / 3 = 16, and 16/128 = 0.125 ≠ the default's 0.0625.
    let f48 = write_comfyui_twin(
        dir.path(),
        "a48.safetensors",
        &cfg,
        FusedQkvSpec::at(48.0, true),
        &qkv,
        (&fc1.0, &fc1.1),
    );
    assert_eq!(read_alpha(&f48), 16.0);
    assert_eq!(alpha_rank_fold(16.0, PUBLISHED_RANK as f32), 0.125);
    assert_ne!(
        alpha_rank_fold(16.0, PUBLISHED_RANK as f32),
        alpha_rank_fold(DEFAULT_LORA_ALPHA, PUBLISHED_RANK as f32),
        "this arm must not land on the default fold, or a converter that dropped the alpha passes"
    );

    // The published pairing: 24 / 3 = 8, holding `alpha/rank` at 24/384 == 8/128 == 0.0625.
    let f24 = write_comfyui_twin(
        dir.path(),
        "a24.safetensors",
        &cfg,
        FusedQkvSpec::at(24.0, true),
        &qkv,
        (&fc1.0, &fc1.1),
    );
    assert_eq!(read_alpha(&f24), 8.0);
    assert_eq!(
        alpha_rank_fold(24.0, 3.0 * PUBLISHED_RANK as f32),
        alpha_rank_fold(8.0, PUBLISHED_RANK as f32)
    );

    // Shared-A keeps rank `r`, so its alpha is UNCHANGED.
    let shared = write_comfyui_twin(
        dir.path(),
        "shared.safetensors",
        &cfg,
        FusedQkvSpec::at(48.0, false),
        &qkv,
        (&fc1.0, &fc1.1),
    );
    assert_eq!(
        read_alpha(&shared),
        48.0,
        "a shared-A fused LoRA is already at per-projection rank; dividing its alpha folds it 3x \
         too weak"
    );
}

/// **Block-diagonality is measured on the BYTES, not inferred from the shape.**
///
/// A shared-`A` fused LoRA whose rank happens to divide by three has exactly the shape a
/// block-diagonal one would, so a converter that took `r % 3 == 0` as proof would split it into
/// three rank-`r/3` LoRAs and divide its alpha — folding it three times too weak, on factors it was
/// never trained with. The rank here is 96 precisely so the shape alone cannot decide.
#[test]
fn block_diagonality_is_measured_on_the_bytes_not_inferred_from_the_shape() {
    let cfg = dit_fixture_config();
    let r: i32 = 96;
    assert_eq!(r % 3, 0);
    let out = cfg.inner_dim();

    let mut w = Weights::empty();
    w.insert(
        "blocks.0.attn.qkv_proj.lora_A.weight",
        bf16(&tensor(&[r, cfg.hidden_size], 1.0)),
    );
    // A DENSE `[3·out, r]` B — emphatically not block-diagonal.
    w.insert(
        "blocks.0.attn.qkv_proj.lora_B.weight",
        bf16(&tensor(&[3 * out, r], 2.0)),
    );
    w.insert(
        "blocks.0.attn.qkv_proj.alpha",
        Array::from_slice(&[48.0f32], &[1]),
    );

    let converted = convert_comfyui_key_space(&w).unwrap();
    let a = converted
        .require("transformer_blocks.0.attn.to_q.lora_A.weight")
        .unwrap();
    assert_eq!(
        a.shape()[0],
        r,
        "a shared-A fused LoRA keeps its full rank; splitting it uses factors it never had"
    );
    let alpha = converted
        .require("transformer_blocks.0.attn.to_q.alpha")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()[0];
    assert_eq!(
        alpha, 48.0,
        "the alpha must NOT be divided when the rank was not split"
    );
}

/// `Weights` is not `Debug`, so `unwrap_err` is unavailable — match the error out by hand.
pub(super) fn expect_conversion_error(w: &Weights) -> String {
    match convert_comfyui_key_space(w) {
        Ok(_) => panic!("a malformed fused qkv must be refused, not silently split"),
        Err(e) => e.to_string(),
    }
}

/// A fused `qkv_proj` whose factors do not compose, or whose `B` is not three equal projections, is
/// an error naming the module — never a guessed split.
#[test]
fn a_malformed_fused_qkv_is_refused_by_name() {
    let cfg = dit_fixture_config();
    let r = PUBLISHED_RANK;
    let out = cfg.inner_dim();

    let mut w = Weights::empty();
    w.insert(
        "blocks.0.attn.qkv_proj.lora_A.weight",
        bf16(&tensor(&[3 * r, cfg.hidden_size], 1.0)),
    );
    w.insert(
        "blocks.0.attn.qkv_proj.lora_B.weight",
        bf16(&tensor(&[3 * out, r], 2.0)),
    );
    let e = expect_conversion_error(&w);
    assert!(e.contains("do not compose"), "{e}");
    assert!(e.contains("qkv_proj"), "{e}");

    let mut w = Weights::empty();
    w.insert(
        "blocks.0.attn.qkv_proj.lora_A.weight",
        bf16(&tensor(&[r, cfg.hidden_size], 1.0)),
    );
    w.insert(
        "blocks.0.attn.qkv_proj.lora_B.weight",
        bf16(&tensor(&[3 * out + 1, r], 2.0)),
    );
    let e = expect_conversion_error(&w);
    assert!(e.contains("three equal q/k/v projections"), "{e}");

    let mut w = Weights::empty();
    w.insert(
        "blocks.0.attn.qkv_proj.lora_A.weight",
        bf16(&tensor(&[r, cfg.hidden_size], 1.0)),
    );
    let e = expect_conversion_error(&w);
    assert!(e.contains("needs both"), "{e}");
}
