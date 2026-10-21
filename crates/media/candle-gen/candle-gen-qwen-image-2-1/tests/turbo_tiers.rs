//! Installable Q8/Q4 tiers of the distilled **Qwen-Image 2.1 Turbo** checkpoint on the Candle
//! backend (sc-25906).
//!
//! A Turbo tier here is **the artefact the MLX converter writes**, composed the way `tiers.rs`
//! composes a base tier: the committed packed `transformer/` + `text_encoder/` of
//! `mlx-gen-qwen-image-2-1/tests/fixtures/tiers/{q8,q4}/` with their marker configs, plus the
//! remainder of a Turbo-shaped snapshot ([`turbo_snapshot_with`]: `model_index.json` saving
//! `sample_sigmas`, the static scheduler config, the bf16 VAE, the tokenizer). That composition is
//! what the converter produces and not an approximation of it: the MLX twin's
//! `turbo_tiers::a_turbo_tier_differs_from_the_base_tier_only_where_the_snapshots_do` holds that a
//! converted Turbo tier and a converted base tier differ in exactly `model_index.json`,
//! `scheduler/scheduler_config.json`, the VAE file, `CHANGES.md` and the `NOTICE` attribution file
//! only a Turbo tier carries (composed here too) — the packed components are the same bytes.
//!
//! Every assertion is made on the generator loaded from that directory through the provider
//! catalog, on its render, or on the contract priced from it.

use std::path::{Path, PathBuf};

use candle_gen::gen_core::{
    CancelFlag, Error as CoreError, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    OffloadPolicy, Progress, Quant, WeightsSource,
};
use candle_gen_qwen_image_2_1::memory_strategy::{
    compute_width, MEMORY_CALIBRATION_FINGERPRINT, TURBO_MEMORY_CALIBRATION_FINGERPRINT,
};
use candle_gen_qwen_image_2_1::quant::{installed_tier, Tier, GROUP_SIZE};
use candle_gen_qwen_image_2_1::{
    create_noise, decode_rgb, denoise_reported, load_text_encoder, load_tokenizer,
    load_transformer, load_vae, loader, system_prompt_drop_count, DenoiseInputs, DenoiseReport,
    KvCacheMode,
};

use crate::common::{
    fixtures, read_json, stored_dtypes, tiny_snapshot, turbo_snapshot, turbo_snapshot_with,
    TURBO_SAMPLE_SIGMAS,
};

const BASE: &str = "qwen_image_2_1";
const TURBO: &str = "qwen_image_2_1_turbo";
const PROMPT: &str = "a red fox in the forest";
const SEED: u64 = 42;
const SIZE: u32 = 32;

// ── composing a tier ─────────────────────────────────────────────────────────────────────────────

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let (path, target) = (entry.path(), dst.join(entry.file_name()));
        if path.is_dir() {
            copy_dir(&path, &target);
        } else {
            std::fs::copy(&path, &target).unwrap();
        }
    }
}

/// The `tier` snapshot the converter writes for the dense snapshot at `source`: the committed
/// packed components with `source`'s component config plus the quantization marker, and
/// everything the converter copies through verbatim taken from `source` itself.
pub(crate) fn compose_tier(source: &Path, dir: &Path, tier: Tier) -> PathBuf {
    let out = dir.join(tier.dir_name());
    let packed = fixtures().join("tiers").join(tier.dir_name());
    for (component, bits) in [
        ("transformer", tier.transformer_bits().unwrap()),
        ("text_encoder", tier.text_encoder_bits().unwrap()),
    ] {
        let dst = out.join(component);
        std::fs::create_dir_all(&dst).unwrap();
        std::fs::copy(
            packed.join(component).join("model.safetensors"),
            dst.join("model.safetensors"),
        )
        .unwrap();
        let mut config = read_json(&source.join(component).join("config.json"));
        config["quantization"] = serde_json::json!({ "bits": bits, "group_size": GROUP_SIZE });
        std::fs::write(
            dst.join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();
    }
    for component in ["vae", "processor", "scheduler"] {
        copy_dir(&source.join(component), &out.join(component));
    }
    for file in ["model_index.json", "LICENSE"] {
        std::fs::copy(source.join(file), out.join(file)).unwrap();
    }
    // A Turbo tier (and only a Turbo tier) carries the licence's §3(c) attribution as `NOTICE`.
    if !read_json(&source.join("model_index.json"))["sample_sigmas"].is_null() {
        std::fs::write(
            out.join("NOTICE"),
            format!("{}\n", candle_gen_qwen_image_2_1::UPSTREAM_LICENSE_NOTICE),
        )
        .unwrap();
    }
    out
}

fn spec(root: &Path, quant: Option<Quant>, policy: OffloadPolicy) -> LoadSpec {
    let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_offload_policy(policy);
    match quant {
        Some(quant) => spec.with_quant(quant),
        None => spec,
    }
}

fn try_load(
    id: &str,
    root: &Path,
    quant: Option<Quant>,
    policy: OffloadPolicy,
) -> candle_gen::gen_core::Result<Box<dyn Generator>> {
    candle_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(id, &spec(root, quant, policy))
}

fn load(id: &str, root: &Path, quant: Option<Quant>, policy: OffloadPolicy) -> Box<dyn Generator> {
    try_load(id, root, quant, policy).unwrap_or_else(|e| {
        panic!(
            "{id} loads {} through the catalog path: {e}",
            root.display()
        )
    })
}

fn request(steps: Option<u32>) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.to_owned(),
        width: SIZE,
        height: SIZE,
        steps,
        seed: Some(SEED),
        ..Default::default()
    }
}

/// The generator's RGB8 pixels and the `Step` events it reported.
fn render(generator: &dyn Generator, req: &GenerationRequest) -> (Vec<u8>, Vec<(u32, u32)>) {
    let mut steps = Vec::new();
    let out = generator
        .generate(req, &mut |p| {
            if let Progress::Step { current, total } = p {
                steps.push((current, total));
            }
        })
        .expect("render");
    let GenerationOutput::Images(mut images) = out else {
        panic!("images expected");
    };
    assert_eq!(images.len(), 1);
    (images.remove(0).pixels, steps)
}

/// One render of [`PROMPT`] at [`SEED`] assembled from the crate's public pipeline pieces over the
/// components of `root` and an **explicit** sigma list, with the prefix KV cache off — the path
/// the generator runs for a request that selects nothing (sc-25905: the cache is a request
/// selection and unset is the prefill path). The independent side of the schedule assertions:
/// the generator chooses its schedule, this does not.
fn render_over(root: &Path, sigmas: &[f32]) -> Vec<u8> {
    let device = candle_gen::default_device().unwrap();
    let tokenizer = load_tokenizer(root).unwrap();
    let encoder = load_text_encoder(root, &device).unwrap();
    let transformer = load_transformer(root, &device).unwrap();
    let vae = load_vae(root, &device).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let prompt = encoder.encode_prompt(&tokenizer, PROMPT, drop).unwrap();
    let seed = candle_gen::image_seed(SEED, 0);
    let noise = create_noise(seed, SIZE, SIZE, transformer.config().in_channels, &device).unwrap();
    let cancel = CancelFlag::new();
    let mut report = DenoiseReport::default();
    let latents = denoise_reported(
        DenoiseInputs {
            transformer: &transformer,
            sigmas,
            latents: noise,
            prompt_embeds: &prompt,
            negative_embeds: None,
            true_cfg_scale: 1.0,
            width: SIZE,
            height: SIZE,
            sampler: None,
            seed,
            cancel: &cancel,
            references: None,
            kv_cache: KvCacheMode::Off,
        },
        &mut report,
        &mut |_| {},
    )
    .unwrap();
    decode_rgb(&vae, &latents, SIZE, SIZE, None, Some(&cancel))
        .unwrap()
        .pixels
}

fn plus_zero(sigmas: &[f64]) -> Vec<f32> {
    sigmas.iter().map(|s| *s as f32).chain([0.0]).collect()
}

fn every_step(total: u32) -> Vec<(u32, u32)> {
    (1..=total).map(|current| (current, total)).collect()
}

/// `(max |Δ|, mean |Δ|)` over two RGB8 buffers, in 0..=255 units.
fn pixel_errors(got: &[u8], want: &[u8]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "buffer length");
    let mut max = 0f32;
    let mut sum = 0f32;
    for (a, b) in got.iter().zip(want) {
        let d = (*a as f32 - *b as f32).abs();
        max = max.max(d);
        sum += d;
    }
    (max, sum / got.len().max(1) as f32)
}

fn profile() -> gen_core_testkit::Profile {
    gen_core_testkit::Profile {
        prompt: PROMPT.to_owned(),
        width: SIZE,
        height: SIZE,
        steps: 3,
        seed: SEED,
        cancel_steps: 6,
    }
}

// ── load + render ────────────────────────────────────────────────────────────────────────────────

/// A Turbo q8 and q4 tier loads under `qwen_image_2_1_turbo` with its matching `quantize`
/// selector, holds the bf16 VAE at the lane's compute dtype, renders over the eight sigmas its
/// own `model_index.json` saves, and passes the generator contract under both residency policies.
#[test]
fn turbo_tiers_load_and_render_under_the_turbo_id() {
    let source = turbo_snapshot();
    let tmp = tempfile::tempdir().unwrap();
    let device = candle_gen::default_device().unwrap();
    let (dense, _) = render(
        load(TURBO, source.path(), None, OffloadPolicy::Resident).as_ref(),
        &request(None),
    );

    let mut mean_by_tier = Vec::new();
    for tier in [Tier::Q8, Tier::Q4] {
        let root = compose_tier(source.path(), tmp.path(), tier);
        let quant = tier.selected_quant();
        let name = tier.dir_name();
        assert_eq!(installed_tier(&root).unwrap(), tier);
        assert_eq!(
            read_json(&root.join("model_index.json"))["sample_sigmas"],
            serde_json::json!(TURBO_SAMPLE_SIGMAS)
        );
        // The tier's VAE file is the Turbo snapshot's bf16 one, held at the lane's compute dtype.
        assert_eq!(
            stored_dtypes(&root.join("vae/diffusion_pytorch_model.safetensors")),
            ["BF16"]
        );
        assert_eq!(
            load_vae(&root, &device).unwrap().weight_dtype(),
            loader::compute_dtype_on(&device)
        );

        let resident = load(TURBO, &root, quant, OffloadPolicy::Resident);
        assert_eq!(resident.descriptor().id, TURBO);
        let (pixels, steps) = render(resident.as_ref(), &request(None));
        assert_eq!(steps, every_step(8), "{name}: one step per saved sigma");
        // The schedule the sampler received, on the render: the packed components of this very
        // tier over the literal saved list.
        assert_eq!(
            pixels,
            render_over(&root, &plus_zero(&TURBO_SAMPLE_SIGMAS)),
            "{name}: the tier renders over its saved sigmas plus 0"
        );

        let (max, mean) = pixel_errors(&pixels, &dense);
        eprintln!("candle turbo {name} vs dense turbo: max|Δ|={max:.2}/255 mean|Δ|={mean:.4}/255");
        // Bars for a PARTIALLY packed miniature tier, as in `tiers.rs`.
        let (max_bound, mean_bound) = match tier {
            Tier::Q8 => (16.0, 1.0),
            _ => (64.0, 8.0),
        };
        assert!(max <= max_bound, "{name}: max|Δ|={max} exceeds {max_bound}");
        assert!(
            mean <= mean_bound,
            "{name}: mean|Δ|={mean} exceeds {mean_bound}"
        );
        mean_by_tier.push(mean);

        // Staging is an execution decision: the same pixels.
        let sequential = load(TURBO, &root, quant, OffloadPolicy::Sequential);
        assert_eq!(render(sequential.as_ref(), &request(None)).0, pixels);

        for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
            gen_core_testkit::conformance(|| load(TURBO, &root, quant, policy), &profile());
        }
    }
    assert!(
        mean_by_tier[1] > mean_by_tier[0],
        "Q4 ({:.4}) must be strictly further from the dense render than Q8 ({:.4})",
        mean_by_tier[1],
        mean_by_tier[0]
    );
}

/// **The schedule is the tier's own.** A tier whose `model_index.json` saves a different list
/// renders over *that* list — its length is the default step count and its values are what the
/// sampler receives — and not over the released eight.
///
/// *Mutation that reds this:* the Turbo load reading a constant sigma list instead of the
/// snapshot's.
#[test]
fn a_turbo_tier_renders_over_the_sigmas_its_own_model_index_saves() {
    let other = [1.0, 0.9, 0.7, 0.4, 0.1];
    let source = turbo_snapshot_with(&other);
    let tmp = tempfile::tempdir().unwrap();
    let root = compose_tier(source.path(), tmp.path(), Tier::Q8);

    let generator = load(TURBO, &root, Some(Quant::Q8), OffloadPolicy::Resident);
    let (pixels, steps) = render(generator.as_ref(), &request(None));
    assert_eq!(steps, every_step(5));
    assert_eq!(pixels, render_over(&root, &plus_zero(&other)));
    // The control: the same tier's components over the released eight is a different picture.
    assert_ne!(pixels, render_over(&root, &plus_zero(&TURBO_SAMPLE_SIGMAS)));
}

// ── refusals ─────────────────────────────────────────────────────────────────────────────────────

/// A base-id refusal as the Turbo id reports it (E3, sc-25909): the base label replaced by the
/// Turbo one, nothing else.
fn turbo_labelled(base: &str) -> String {
    base.replacen("qwen_image_2_1: ", "qwen_image_2_1_turbo: ", 1)
}

/// The typed `Unsupported` refusal of `id` for the snapshot at `root`, through `load` and through
/// the memory registration, with the snapshot path masked so a tier's refusal can be compared with
/// its dense source's.
fn refusals(id: &str, root: &Path, quant: Option<Quant>) -> Vec<String> {
    let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
    let spec = spec(root, quant, OffloadPolicy::Resident);
    let by_load = match registry.load(id, &spec) {
        Ok(_) => panic!("{id} must refuse {}", root.display()),
        Err(err) => err,
    };
    let by_contract = registry
        .memory_strategy_contract(id, &spec)
        .expect_err("no contract is priced for a refused load");
    [by_load, by_contract]
        .into_iter()
        .map(|err| match err {
            CoreError::Unsupported(message) => {
                message.replace(&root.display().to_string(), "<snapshot>")
            }
            other => panic!("{id}: expected a typed Unsupported refusal, got {other:?}"),
        })
        .collect()
}

/// A converted tier carries its source's `model_index.json` and scheduler config, so each engine
/// id refuses the other checkpoint's **tier** with exactly the typed refusal its dense snapshot
/// gets (`loader::check_snapshot_variant`), through `load` and through the memory registration —
/// and before the tier/request rule is consulted.
///
/// *Mutation that reds this:* the base arm of `check_snapshot_variant` accepting a snapshot that
/// saves `sample_sigmas` (a Turbo tier would then load under `qwen_image_2_1`).
#[test]
fn each_id_refuses_the_other_checkpoints_tier_like_its_dense_snapshot() {
    let turbo_source = turbo_snapshot();
    let turbo_tiers = tempfile::tempdir().unwrap();
    let base_tiers = tempfile::tempdir().unwrap();

    let dense_turbo_under_base = refusals(BASE, turbo_source.path(), None);
    let dense_base_under_turbo = refusals(TURBO, &tiny_snapshot(), None);
    assert!(dense_turbo_under_base[0].contains("`qwen_image_2_1_turbo`"));
    assert!(dense_base_under_turbo[0].contains("no `sample_sigmas`"));

    for tier in [Tier::Q8, Tier::Q4] {
        let quant = tier.selected_quant();
        let turbo_tier = compose_tier(turbo_source.path(), turbo_tiers.path(), tier);
        let base_tier = compose_tier(&tiny_snapshot(), base_tiers.path(), tier);
        assert_eq!(
            refusals(BASE, &turbo_tier, quant),
            dense_turbo_under_base,
            "{}: a Turbo tier under the base id",
            tier.dir_name()
        );
        assert_eq!(
            refusals(TURBO, &base_tier, quant),
            dense_base_under_turbo,
            "{}: a base tier under the Turbo id",
            tier.dir_name()
        );
        // Each tier still loads under its own id.
        assert!(try_load(TURBO, &turbo_tier, quant, OffloadPolicy::Resident).is_ok());
        assert!(try_load(BASE, &base_tier, quant, OffloadPolicy::Resident).is_ok());
    }
}

/// The tier/request rules are the base checkpoint's, unchanged, under the Turbo id. On this
/// backend a **dense** snapshot with a Q4/Q8 request is the typed "no on-the-fly quantization"
/// refusal — the same one, word for word, on both ids except the id it names (E3, sc-25909:
/// `qwen_image_2_1_turbo:` under the Turbo id) — and a request that disagrees with a packed tier,
/// or none against one, is refused as it is for a base tier, under the same rule.
///
/// *Mutation that reds this:* dropping `labelled` from `load_variant` (the Turbo refusal keeps the
/// base label).
#[test]
fn turbo_keeps_the_base_tier_refusals() {
    let source = turbo_snapshot();
    let turbo_tiers = tempfile::tempdir().unwrap();
    let base_tiers = tempfile::tempdir().unwrap();
    let resident = OffloadPolicy::Resident;
    let error =
        |id: &str, root: &Path, quant: Option<Quant>| match try_load(id, root, quant, resident) {
            Ok(_) => panic!("{id}: the request must be refused"),
            Err(err) => (
                std::mem::discriminant(&err),
                err.to_string()
                    .replace(&root.display().to_string(), "<snapshot>"),
            ),
        };

    let dense_refusal =
        |id: &str, root: &Path, quant: Quant| match try_load(id, root, Some(quant), resident) {
            Err(CoreError::Unsupported(message)) => message,
            Err(other) => {
                panic!("{id} {quant:?}: expected a typed Unsupported refusal, got {other:?}")
            }
            Ok(_) => panic!("{id} {quant:?}: a dense snapshot must not be quantized at load"),
        };
    for quant in [Quant::Q8, Quant::Q4] {
        let turbo = dense_refusal(TURBO, source.path(), quant);
        assert!(
            turbo.contains("no on-the-fly Q4/Q8 quantization"),
            "{turbo}"
        );
        assert_eq!(
            turbo,
            turbo_labelled(&dense_refusal(BASE, &tiny_snapshot(), quant)),
            "{quant:?}: the base id's refusal, word for word, under the Turbo id"
        );
    }

    let turbo_q8 = compose_tier(source.path(), turbo_tiers.path(), Tier::Q8);
    let base_q8 = compose_tier(&tiny_snapshot(), base_tiers.path(), Tier::Q8);
    for quant in [Some(Quant::Q4), None] {
        let (kind, message) = error(BASE, &base_q8, quant);
        assert!(message.contains("qwen_image_2_1: "), "{message}");
        assert_eq!(
            error(TURBO, &turbo_q8, quant),
            (kind, turbo_labelled(&message)),
            "{quant:?} against a q8 tier"
        );
    }
    let (_, crossed) = error(TURBO, &turbo_q8, Some(Quant::Q4));
    assert!(crossed.contains("pre-quantized q8 tier"), "{crossed}");
}

// ── memory ───────────────────────────────────────────────────────────────────────────────────────

/// Element count of every tensor in one `.safetensors` file, from its header.
fn stored_elements(path: &Path) -> u64 {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let parsed = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("{} parses: {e}", path.display()));
    parsed
        .tensors()
        .iter()
        .map(|(_, view)| view.shape().iter().map(|d| *d as u64).product::<u64>())
        .sum()
}

/// **The Turbo contract prices the bf16 VAE at the lane's compute width, under the Turbo id's own
/// calibration identity.** The file stores bf16; candle materializes it at
/// `loader::compute_dtype()` (f32 on the CPU lane, bf16 on a GPU lane), so the priced bytes are
/// the file's element count times [`compute_width`] — the same figure the base checkpoint's f32
/// file prices at on this lane, and on the CPU lane twice what the file stores. Held on the dense
/// Turbo snapshot and on both tiers.
///
/// *Mutations that red this:* pricing the VAE at its stored width; the Turbo contract carrying
/// the base fingerprint.
#[test]
fn the_turbo_contract_prices_the_vae_at_the_compute_width_under_its_own_identity() {
    use candle_gen_qwen_image_2_1::memory_strategy::memory_strategy_contract;

    let source = turbo_snapshot();
    let tmp = tempfile::tempdir().unwrap();
    let base_tiers = tempfile::tempdir().unwrap();
    let vae = "vae/diffusion_pytorch_model.safetensors";
    let elements = stored_elements(&source.path().join(vae));
    assert_eq!(elements, stored_elements(&tiny_snapshot().join(vae)));
    assert_ne!(
        TURBO_MEMORY_CALIBRATION_FINGERPRINT,
        MEMORY_CALIBRATION_FINGERPRINT
    );

    let resident = OffloadPolicy::Resident;
    let mut cases = vec![(source.path().to_path_buf(), tiny_snapshot(), None)];
    for tier in [Tier::Q8, Tier::Q4] {
        cases.push((
            compose_tier(source.path(), tmp.path(), tier),
            compose_tier(&tiny_snapshot(), base_tiers.path(), tier),
            tier.selected_quant(),
        ));
    }
    for (turbo_root, base_root, quant) in cases {
        assert_eq!(stored_dtypes(&turbo_root.join(vae)), ["BF16"]);
        let turbo = memory_strategy_contract(TURBO, &spec(&turbo_root, quant, resident)).unwrap();
        let base = memory_strategy_contract(BASE, &spec(&base_root, quant, resident)).unwrap();

        assert_eq!(turbo.provider_id, TURBO);
        assert_eq!(
            turbo.asset_facts.decoder_bytes,
            elements * compute_width(),
            "{quant:?}: the VAE is priced at the width candle holds it at"
        );
        // The same network at the same compute width: the two checkpoints price alike here.
        assert_eq!(turbo.asset_facts, base.asset_facts, "{quant:?}");

        assert_eq!(
            turbo.calibration.as_ref().unwrap().fingerprint,
            TURBO_MEMORY_CALIBRATION_FINGERPRINT,
            "{quant:?}"
        );
        assert_eq!(
            base.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT,
            "{quant:?}"
        );
    }
}
