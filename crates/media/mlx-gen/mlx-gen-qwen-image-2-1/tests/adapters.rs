//! LoRA / LoKr on the Qwen-Image 2.1 DiT (sc-24156), on the committed miniature snapshot.
//!
//! Three layers:
//!
//! * **The host** (`AdaptableHost for QwenImage21Transformer`) — the key→module surface, the kohya
//!   flattened table, and the residual itself: an installed adapter changes the DiT's velocity on
//!   the dense base and on a load-time **Q4- or Q8-packed** base (where it must stay a residual —
//!   the base remains packed), LoRA and LoKr alike, stacked across files with their own strengths;
//!   a strength of zero is byte-identical to no adapter. Unmatched keys are a strict refusal that
//!   names them.
//! * **The load path** (`provider_registry().load(..)` with `LoadSpec::adapters`) — the refusal is
//!   gone, the descriptor advertises LoRA + LoKr, and the adapters reach the text-to-image route
//!   (Resident and Sequential), the reference/edit route, and pre-quantized Q4 and Q8 tiers whose
//!   DiT is asserted packed. An unmatched key fails the load under both residency policies.
//! * **The memory contract** — an adapter load prices its resident overlay (factors plus every
//!   materialized LoKr delta, per the tier's real packing) on the typed component axis, and an
//!   unsizable source is refused rather than priced at zero.
//!
//! The adapter files are synthesized at test time against the host's own base shapes (read through
//! the probe half, `adaptable_facts`), with deterministic bounded values — no RNG, no real weights.

use std::path::{Path, PathBuf};

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core::{Conditioning, MemoryComponentKind, MemoryFormulaVariable};
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, WeightsSource};
use mlx_gen_qwen_image_2_1::convert::prequantize_turnkey;
use mlx_gen_qwen_image_2_1::memory_strategy::memory_strategy_contract;
use mlx_gen_qwen_image_2_1::quant::Tier;
use mlx_gen_qwen_image_2_1::{
    apply_qwen_image_2_1_adapters, load_transformer, QwenImage21Transformer, BLOCK_ADAPTER_TARGETS,
    GLOBAL_ADAPTER_TARGETS,
};
use mlx_rs::Array;

use crate::common::{errors, fixture, meta_f32, meta_usize, tiny_snapshot};

const ID: &str = "qwen_image_2_1";

// ── adapter-file synthesis ───────────────────────────────────────────────────────────────────────

/// Write a raw F32 `.safetensors` with `__metadata__` `meta`. Values are deterministic and bounded
/// in `[-amp/2, amp/2]`, varied per tensor and never all-zero.
fn write_safetensors(
    path: &Path,
    entries: &[(String, Vec<usize>)],
    meta: &[(&str, &str)],
    amp: f32,
) {
    let mut data: Vec<u8> = Vec::new();
    let mut header = String::from("{\"__metadata__\":{\"format\":\"pt\"");
    for (k, v) in meta {
        header.push_str(&format!(",\"{k}\":\"{v}\""));
    }
    header.push('}');
    for (i, (name, shape)) in entries.iter().enumerate() {
        let n: usize = shape.iter().product();
        let start = data.len();
        for j in 0..n {
            let v = (((i * 131 + j * 17 + 7) % 101) as f32 / 101.0 - 0.5) * amp;
            data.extend_from_slice(&v.to_le_bytes());
        }
        let dims = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            ",\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{dims}],\"data_offsets\":[{start},{}]}}",
            data.len()
        ));
    }
    header.push('}');
    let header = header.into_bytes();
    let mut buf = (header.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(&header);
    buf.extend_from_slice(&data);
    std::fs::write(path, buf).unwrap();
}

/// `[out, in]` of the Linear at dotted `path`, through the probe half of the host.
fn base_shape(host: &mut QwenImage21Transformer, path: &str) -> (usize, usize) {
    let segs: Vec<&str> = path.split('.').collect();
    let facts = host
        .adaptable_facts(&segs)
        .unwrap_or_else(|| panic!("no adaptable Linear at {path}"));
    (facts.base_shape[0] as usize, facts.base_shape[1] as usize)
}

const RANK: usize = 4;

/// A diffusers/peft LoRA (`transformer.` namespace — the SceneWorks trainer's export) over
/// `targets`, shaped from the host. A target the host does not have (a deliberately unmatched key)
/// is written `[32, 32]` — its shape is never read, the install refuses it by name first.
fn peft_lora(
    dir: &Path,
    name: &str,
    host: &mut QwenImage21Transformer,
    targets: &[&str],
    amp: f32,
) -> PathBuf {
    let mut entries = Vec::new();
    for target in targets {
        let segs: Vec<&str> = target.split('.').collect();
        let (out, inp) = match host.adaptable_facts(&segs) {
            Some(f) => (f.base_shape[0] as usize, f.base_shape[1] as usize),
            None => (32, 32),
        };
        entries.push((
            format!("transformer.{target}.lora_A.weight"),
            vec![RANK, inp],
        ));
        entries.push((
            format!("transformer.{target}.lora_B.weight"),
            vec![out, RANK],
        ));
    }
    let path = dir.join(name);
    write_safetensors(&path, &entries, &[], amp);
    path
}

/// A kohya `lora_unet_` LoRA over `targets` (dotted paths, flattened here).
fn kohya_lora(
    dir: &Path,
    host: &mut QwenImage21Transformer,
    targets: &[&str],
    amp: f32,
) -> PathBuf {
    let mut entries = Vec::new();
    for target in targets {
        let (out, inp) = base_shape(host, target);
        let flat = target.replace('.', "_");
        entries.push((
            format!("lora_unet_{flat}.lora_down.weight"),
            vec![RANK, inp],
        ));
        entries.push((format!("lora_unet_{flat}.lora_up.weight"), vec![out, RANK]));
    }
    let path = dir.join("kohya.safetensors");
    write_safetensors(&path, &entries, &[], amp);
    path
}

/// A SceneWorks/peft LoKr (`networkType=lokr`, full Kronecker factors, `alpha == rank` ⇒ scale 1)
/// over `targets`: `w1 [2, 2] ⊗ w2 [out/2, in/2]` reconstructs the `[out, in]` delta.
fn peft_lokr(dir: &Path, host: &mut QwenImage21Transformer, targets: &[&str], amp: f32) -> PathBuf {
    let mut entries = Vec::new();
    for target in targets {
        let (out, inp) = base_shape(host, target);
        assert!(
            out.is_multiple_of(2) && inp.is_multiple_of(2),
            "{target}: [{out}, {inp}] halves"
        );
        entries.push((format!("{target}.lokr_w1"), vec![2, 2]));
        entries.push((format!("{target}.lokr_w2"), vec![out / 2, inp / 2]));
    }
    let path = dir.join("lokr.safetensors");
    write_safetensors(
        &path,
        &entries,
        &[("networkType", "lokr"), ("rank", "1"), ("alpha", "1")],
        amp,
    );
    path
}

fn lora(path: &Path, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path.to_path_buf(), scale, AdapterKind::Lora)
}

fn lokr(path: &Path, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path.to_path_buf(), scale, AdapterKind::Lokr)
}

// ── host-level forward ───────────────────────────────────────────────────────────────────────────

/// The fixture's `square` case through the DiT: the target velocity.
fn velocity(model: &QwenImage21Transformer) -> Array {
    let w = fixture("qwen21_transformer.safetensors");
    let height = meta_usize(&w, "square/height");
    let width = meta_usize(&w, "square/width");
    let timestep = meta_f32(&w, "square/timestep");
    let hidden = w.require("square/hidden_states").unwrap();
    let text = w.require("square/encoder_hidden_states").unwrap();
    let out = model
        .forward(hidden, text, timestep, height, width)
        .expect("forward");
    out.eval().unwrap();
    out
}

fn dense() -> QwenImage21Transformer {
    load_transformer(&tiny_snapshot()).expect("the tiny DiT loads")
}

/// The tiny DiT quantized at load to Q4 — every Linear at least 32 wide packs (group 32 or 64).
fn packed_q4() -> QwenImage21Transformer {
    let mut model = dense();
    model.quantize(4).expect("load-time Q4");
    model
}

/// The tiny DiT quantized at load to Q8 — same packing rule as [`packed_q4`], 8-bit codes.
fn packed_q8() -> QwenImage21Transformer {
    let mut model = dense();
    model.quantize(8).expect("load-time Q8");
    model
}

/// A DiT constructor — [`dense`], [`packed_q4`] or [`packed_q8`].
type Build = fn() -> QwenImage21Transformer;

fn is_packed(model: &mut QwenImage21Transformer, path: &str) -> bool {
    let segs: Vec<&str> = path.split('.').collect();
    model.adaptable_facts(&segs).unwrap().is_quantized
}

fn adapter_count(model: &mut QwenImage21Transformer, path: &str) -> usize {
    let segs: Vec<&str> = path.split('.').collect();
    model.adaptable_facts(&segs).unwrap().adapter_count
}

/// `max |a − b| > 1e-3 · max(1, peak)` — the adapter visibly moved the velocity.
fn assert_moved(name: &str, got: &Array, base: &Array) {
    let (max_abs, peak, mean) = errors(got, base);
    eprintln!("{name}: max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e} peak={peak:.3e}");
    assert!(
        max_abs > 1e-3 * peak.max(1.0),
        "{name}: the adapter did not change the velocity (max|Δ|={max_abs:.3e}, peak {peak:.3e})"
    );
}

fn assert_identical(name: &str, got: &Array, base: &Array) {
    let (max_abs, _, _) = errors(got, base);
    assert_eq!(max_abs, 0.0, "{name}: expected byte-identical velocity");
}

/// One block-attention, one block-MLP and one global target — the residual has to survive every
/// later stage (attention, gating, the final norm) to reach the velocity.
const TARGETS: [&str; 3] = [
    "transformer_blocks.0.attn.to_v",
    "transformer_blocks.1.img_mlp.gate_layer",
    "proj_out",
];

// ── the host surface ─────────────────────────────────────────────────────────────────────────────

/// Every Linear of the DiT is addressable, by its checkpoint path, and the kohya enumeration is that
/// exact set: 7 per block + 8 globals, each resolving, collision-free once flattened.
#[test]
fn every_dit_linear_is_an_adapter_target_and_the_kohya_table_is_collision_free() {
    let mut model = dense();
    let layers = model.config().num_layers;
    let paths = model.adaptable_paths();
    assert_eq!(
        paths.len(),
        layers * BLOCK_ADAPTER_TARGETS.len() + GLOBAL_ADAPTER_TARGETS.len()
    );
    let mut expected: Vec<String> = (0..layers)
        .flat_map(|i| {
            BLOCK_ADAPTER_TARGETS
                .iter()
                .map(move |t| format!("transformer_blocks.{i}.{t}"))
        })
        .collect();
    expected.extend(GLOBAL_ADAPTER_TARGETS.iter().map(|g| g.to_string()));
    let (mut got_sorted, mut want_sorted) = (paths.clone(), expected);
    got_sorted.sort();
    want_sorted.sort();
    assert_eq!(got_sorted, want_sorted);

    let mut flattened = std::collections::BTreeSet::new();
    for path in &paths {
        let segs: Vec<&str> = path.split('.').collect();
        assert!(
            model.adaptable_facts(&segs).is_some(),
            "{path} is enumerated but does not resolve"
        );
        assert!(
            flattened.insert(path.replace('.', "_")),
            "{path} collides once kohya-flattened"
        );
    }
    // 2512's dual-stream keys do not exist on the single-stream DiT.
    for absent in [
        "transformer_blocks.0.attn.add_q_proj",
        "transformer_blocks.0.attn.to_add_out",
        "transformer_blocks.0.txt_mlp.net.0.proj",
        "transformer_blocks.0.img_mod.1",
        "transformer_blocks.99.attn.to_q",
    ] {
        let segs: Vec<&str> = absent.split('.').collect();
        assert!(model.adaptable_facts(&segs).is_none(), "{absent} resolved");
    }
}

/// A LoRA on the dense DiT moves the velocity; scale 0 is byte-identical to no adapter.
#[test]
fn a_lora_residual_changes_the_dense_velocity() {
    let tmp = tempfile::tempdir().unwrap();
    let base = velocity(&dense());

    let mut model = dense();
    let file = peft_lora(tmp.path(), "lora.safetensors", &mut model, &TARGETS, 1.0);
    let report = apply_qwen_image_2_1_adapters(&mut model, &[lora(&file, 1.0)]).unwrap();
    assert_eq!(report.applied, TARGETS.len());
    assert!(report.unmatched_paths.is_empty());
    for t in TARGETS {
        assert_eq!(adapter_count(&mut model, t), 1, "{t}");
    }
    assert_moved("dense/lora", &velocity(&model), &base);

    let mut off = dense();
    apply_qwen_image_2_1_adapters(&mut off, &[lora(&file, 0.0)]).unwrap();
    assert_identical("dense/lora@0", &velocity(&off), &base);
}

/// On a packed DiT the adapter is a residual over the packed base: the base stays packed, the
/// adapter is installed on it, and the velocity moves relative to the same packed DiT without it.
fn assert_lora_moves_a_packed_dit(name: &str, build: Build) {
    let tmp = tempfile::tempdir().unwrap();
    let base = velocity(&build());

    let mut model = build();
    for t in TARGETS {
        assert!(
            is_packed(&mut model, t),
            "{name}: {t} packs on the tiny DiT"
        );
    }
    let file = peft_lora(tmp.path(), "lora.safetensors", &mut model, &TARGETS, 1.0);
    let report = apply_qwen_image_2_1_adapters(&mut model, &[lora(&file, 1.0)]).unwrap();
    assert_eq!(report.applied, TARGETS.len(), "{name}");
    for t in TARGETS {
        assert!(
            is_packed(&mut model, t),
            "{name}: {t}: the install must not unpack the base"
        );
        assert_eq!(adapter_count(&mut model, t), 1, "{name}: {t}");
    }
    assert_moved(&format!("{name}/lora"), &velocity(&model), &base);
}

#[test]
fn a_lora_residual_changes_the_packed_q4_velocity_and_leaves_the_base_packed() {
    assert_lora_moves_a_packed_dit("q4", packed_q4);
}

/// The Q8 twin: 8-bit codes over the same packing, so an install that unpacked a Q8 base, or a
/// residual that was lost against Q8's finer codes, shows here.
#[test]
fn a_lora_residual_changes_the_packed_q8_velocity_and_leaves_the_base_packed() {
    assert_lora_moves_a_packed_dit("q8", packed_q8);
}

/// LoKr on every base: a materialized delta over the dense DiT, the structured (never-materialized)
/// Kronecker residual over the packed Q4 and Q8 ones — all move the velocity.
#[test]
fn a_lokr_residual_changes_the_velocity_on_dense_and_packed_q4_q8() {
    let tmp = tempfile::tempdir().unwrap();
    let targets = [
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.1.attn.to_out.0",
    ];
    for (name, build) in [
        ("dense", dense as Build),
        ("q4", packed_q4 as Build),
        ("q8", packed_q8 as Build),
    ] {
        let base = velocity(&build());
        let mut model = build();
        let file = peft_lokr(tmp.path(), &mut model, &targets, 1.0);
        let report = apply_qwen_image_2_1_adapters(&mut model, &[lokr(&file, 1.0)]).unwrap();
        assert_eq!(report.applied, targets.len(), "{name}");
        assert!(report.unmatched_paths.is_empty(), "{name}");
        assert_moved(&format!("{name}/lokr"), &velocity(&model), &base);
    }
}

/// Several files stack, each at its own strength: LoRA + LoKr + a kohya LoRA (which also reaches a
/// global through the flattened table) all install, and the stack differs from each file alone.
#[test]
fn stacked_mixed_adapters_install_with_per_file_strengths() {
    let tmp = tempfile::tempdir().unwrap();
    let mut probe = dense();
    let a = peft_lora(tmp.path(), "a.safetensors", &mut probe, &TARGETS, 1.0);
    let b = peft_lokr(
        tmp.path(),
        &mut probe,
        &["transformer_blocks.0.attn.to_v"],
        1.0,
    );
    let c = kohya_lora(
        tmp.path(),
        &mut probe,
        &["transformer_blocks.1.attn.to_out.0", "txt_in.in_layer"],
        1.0,
    );

    let mut stacked = dense();
    let report = apply_qwen_image_2_1_adapters(
        &mut stacked,
        &[lora(&a, 0.75), lokr(&b, 0.5), lora(&c, 1.25)],
    )
    .unwrap();
    assert_eq!(report.applied, TARGETS.len() + 1 + 2);
    assert!(report.unmatched_paths.is_empty());
    // `to_v` carries the LoRA and the LoKr.
    assert_eq!(
        adapter_count(&mut stacked, "transformer_blocks.0.attn.to_v"),
        2
    );
    assert_eq!(adapter_count(&mut stacked, "txt_in.in_layer"), 1);

    let mut lora_only = dense();
    apply_qwen_image_2_1_adapters(&mut lora_only, &[lora(&a, 0.75)]).unwrap();
    assert_moved("stack-vs-one", &velocity(&stacked), &velocity(&lora_only));

    // Strength is per file: the same stack at a different strength for one file differs too.
    let mut restrength = dense();
    apply_qwen_image_2_1_adapters(
        &mut restrength,
        &[lora(&a, 0.75), lokr(&b, 0.5), lora(&c, 0.25)],
    )
    .unwrap();
    assert_moved(
        "per-file strength",
        &velocity(&stacked),
        &velocity(&restrength),
    );
}

/// Keys that address no 2.1 Linear — here the 2512 dual-stream text-stream keys a 2512 LoRA would
/// carry — fail the install, and the error names every one of them.
#[test]
fn unmatched_keys_are_refused_by_name() {
    let tmp = tempfile::tempdir().unwrap();
    let mut model = dense();
    let file = peft_lora(
        tmp.path(),
        "mixed.safetensors",
        &mut model,
        &[
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.add_q_proj",
            "transformer_blocks.1.txt_mlp.net.0.proj",
        ],
        1.0,
    );
    let err = apply_qwen_image_2_1_adapters(&mut model, &[lora(&file, 1.0)])
        .expect_err("unmatched keys must refuse the install")
        .to_string();
    assert!(err.contains("matched no module"), "{err}");
    assert!(
        err.contains("transformer_blocks.0.attn.add_q_proj"),
        "{err}"
    );
    assert!(
        err.contains("transformer_blocks.1.txt_mlp.net.0.proj"),
        "{err}"
    );
    assert!(err.contains(ID), "the refusal names the model: {err}");
}

// ── the load path ────────────────────────────────────────────────────────────────────────────────

#[test]
fn the_descriptor_advertises_lora_and_lokr() {
    let caps = mlx_gen_qwen_image_2_1::descriptor().capabilities;
    assert!(caps.supports_lora);
    assert!(caps.supports_lokr);
}

fn render(spec: &LoadSpec, req: &GenerationRequest) -> Image {
    let generator = mlx_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(ID, spec)
        .expect("the snapshot loads through the catalog path");
    match generator.generate(req, &mut |_| {}).expect("render") {
        GenerationOutput::Images(mut images) => images.remove(0),
        other => panic!("images expected, got {other:?}"),
    }
}

fn t2i(edge: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in the forest".to_owned(),
        width: edge,
        height: edge,
        steps: Some(2),
        seed: Some(42),
        ..Default::default()
    }
}

/// A large-amplitude LoRA over a block target and the output projection, so the change survives
/// the VAE decode and u8 quantization of a 2-step miniature render. `transformer_blocks.0.img_mlp.out`
/// is 64 wide, so it is one of the Linears the converted Q4 tier actually packs.
fn render_lora(dir: &Path) -> PathBuf {
    let mut probe = dense();
    peft_lora(
        dir,
        "render.safetensors",
        &mut probe,
        &[
            "transformer_blocks.0.attn.to_v",
            "transformer_blocks.0.img_mlp.out",
            "proj_out",
        ],
        2.0,
    )
}

fn snapshot_spec(root: PathBuf) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(root))
}

/// T2I: the adapter reaches the render, under both residency policies (Sequential rebuilds the DiT
/// per request through the same `load_heavy`, so it must render the same adapted pixels), and a
/// strength of zero is pixel-identical to no adapter.
#[test]
fn adapters_reach_the_t2i_route_under_both_residencies() {
    let tmp = tempfile::tempdir().unwrap();
    let file = render_lora(tmp.path());
    let req = t2i(32);
    let plain = render(&snapshot_spec(tiny_snapshot()), &req);

    let adapted_spec = snapshot_spec(tiny_snapshot()).with_adapters(vec![lora(&file, 1.0)]);
    let adapted = render(&adapted_spec, &req);
    assert_ne!(
        adapted.pixels, plain.pixels,
        "the LoRA must change the T2I render"
    );

    let sequential = render(
        &adapted_spec
            .clone()
            .with_offload_policy(OffloadPolicy::Sequential),
        &req,
    );
    assert_eq!(
        sequential.pixels, adapted.pixels,
        "Sequential must install the same adapters as Resident"
    );

    let off = render(
        &snapshot_spec(tiny_snapshot()).with_adapters(vec![lora(&file, 0.0)]),
        &req,
    );
    assert_eq!(
        off.pixels, plain.pixels,
        "strength 0 is byte-identical to no adapter"
    );
}

/// The reference/edit route renders through the same DiT, so it carries the adapter too.
#[test]
fn adapters_reach_the_reference_edit_route() {
    let tmp = tempfile::tempdir().unwrap();
    let file = render_lora(tmp.path());
    let edge = 64u32;
    let pixels: Vec<u8> = (0..edge * edge * 3)
        .map(|i| ((i * 37 + 11) % 251) as u8)
        .collect();
    let req = GenerationRequest {
        prompt: "make the fox blue".to_owned(),
        conditioning: vec![Conditioning::Reference {
            image: Image {
                width: edge,
                height: edge,
                pixels,
            },
            strength: None,
        }],
        ..t2i(edge)
    };
    let plain = render(&snapshot_spec(tiny_snapshot()), &req);
    let adapted = render(
        &snapshot_spec(tiny_snapshot()).with_adapters(vec![lora(&file, 1.0)]),
        &req,
    );
    assert_ne!(
        adapted.pixels, plain.pixels,
        "the LoRA must change the reference render"
    );
}

/// The documented boundary — ten ordered references — carries the adapter too (sc-24163): the
/// 10-reference edit render runs, and the LoRA changes it.
///
/// *Mutation that reds this:* the reference/edit route denoising through an un-adapted DiT.
#[test]
fn adapters_reach_a_ten_reference_edit_render() {
    let tmp = tempfile::tempdir().unwrap();
    let file = render_lora(tmp.path());
    let edge = 64u32;
    let images: Vec<Image> = (0..mlx_gen_qwen_image_2_1::MAX_REFERENCE_IMAGES as u32)
        .map(|r| Image {
            width: edge,
            height: edge,
            pixels: (0..edge * edge * 3)
                .map(|i| ((i * 37 + r * 53 + 11) % 251) as u8)
                .collect(),
        })
        .collect();
    assert_eq!(images.len(), 10);
    let req = GenerationRequest {
        prompt: "combine every image".to_owned(),
        conditioning: vec![Conditioning::MultiReference { images }],
        ..t2i(edge)
    };
    let plain = render(&snapshot_spec(tiny_snapshot()), &req);
    let adapted = render(
        &snapshot_spec(tiny_snapshot()).with_adapters(vec![lora(&file, 1.0)]),
        &req,
    );
    assert_eq!(adapted.pixels.len(), (edge * edge * 3) as usize);
    assert_ne!(
        adapted.pixels, plain.pixels,
        "the LoRA must change the 10-reference render"
    );
}

/// Convert the tiny snapshot into `tier` under `dir`.
fn convert_tier(dir: &Path, tier: Tier) -> PathBuf {
    let out = dir.join(tier.dir_name());
    prequantize_turnkey(&tiny_snapshot(), &out, tier).expect("the tiny snapshot converts");
    out
}

/// The Linear [`render_lora`] relies on reaching a packed base: `img_mlp.out` reads the 64-wide
/// FFN, a multiple of the tier's group, so the converter packs it.
const PACKED_RENDER_TARGET: &str = "transformer_blocks.0.img_mlp.out";

/// A pre-quantized tier takes the same adapters as residuals over its packed DiT. The packing is
/// asserted on the tier's own DiT, not assumed: loaded from the converted snapshot, the render
/// target is packed, and installing the render LoRA on it leaves it packed with the adapter on.
fn assert_adapters_reach_a_packed_tier(tier: Tier) {
    let tmp = tempfile::tempdir().unwrap();
    let file = render_lora(tmp.path());
    let root = convert_tier(tmp.path(), tier);
    let quant = tier.selected_quant().expect("the tier selects a quant");

    let mut dit = load_transformer(&root).expect("the converted tier's DiT loads");
    assert!(
        is_packed(&mut dit, PACKED_RENDER_TARGET),
        "{tier:?}: the converted tier must pack {PACKED_RENDER_TARGET}"
    );
    apply_qwen_image_2_1_adapters(&mut dit, &[lora(&file, 1.0)]).unwrap();
    assert!(
        is_packed(&mut dit, PACKED_RENDER_TARGET),
        "{tier:?}: the install must not unpack the tier's base"
    );
    assert_eq!(adapter_count(&mut dit, PACKED_RENDER_TARGET), 1, "{tier:?}");

    let req = t2i(32);
    let plain = render(&snapshot_spec(root.clone()).with_quant(quant), &req);
    let adapted = render(
        &snapshot_spec(root)
            .with_quant(quant)
            .with_adapters(vec![lora(&file, 1.0)]),
        &req,
    );
    assert_ne!(
        adapted.pixels, plain.pixels,
        "{tier:?}: the LoRA must change the packed-tier render"
    );
}

#[test]
fn adapters_reach_a_packed_q4_tier() {
    assert_adapters_reach_a_packed_tier(Tier::Q4);
}

#[test]
fn adapters_reach_a_packed_q8_tier() {
    assert_adapters_reach_a_packed_tier(Tier::Q8);
}

/// The load path is strict under both residency policies: an unmatched key fails the load and the
/// error names it. `Resident` builds the DiT eagerly; `Sequential` defers it to every generate, so
/// `load` resolves the stack against a lazy DiT graph rather than let each request fail instead.
#[test]
fn an_unmatched_adapter_key_fails_the_load_by_name() {
    let tmp = tempfile::tempdir().unwrap();
    let mut probe = dense();
    let file = peft_lora(
        tmp.path(),
        "bad.safetensors",
        &mut probe,
        &["proj_out", "transformer_blocks.0.attn.to_add_out"],
        1.0,
    );
    for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
        let spec = snapshot_spec(tiny_snapshot())
            .with_offload_policy(policy)
            .with_adapters(vec![lora(&file, 1.0)]);
        let err = match mlx_gen_qwen_image_2_1::provider_registry()
            .unwrap()
            .load(ID, &spec)
        {
            Ok(_) => panic!("{policy:?}: an unmatched adapter key must fail the load"),
            Err(err) => err.to_string(),
        };
        assert!(
            err.contains("transformer_blocks.0.attn.to_add_out"),
            "{policy:?}: {err}"
        );
    }
}

// ── the memory contract ──────────────────────────────────────────────────────────────────────────

/// LoKr targets that split the packing rules three ways on the tiny DiT: `to_k` reads the 32-wide
/// inner dim (packed at load time, which falls back to group 32, but left dense by the group-64
/// converter), `img_mlp.out` reads the 64-wide FFN (packed everywhere), `img_in` reads 8 latent
/// channels (dense everywhere).
const PRICED_LOKR_TARGETS: [&str; 3] = [
    "transformer_blocks.0.attn.to_k",
    "transformer_blocks.1.img_mlp.out",
    "img_in",
];

fn file_bytes(path: &Path) -> u64 {
    std::fs::metadata(path).unwrap().len()
}

/// Bytes of the bf16 `[out, in]` deltas the shared install materializes for `targets` on `host`:
/// one per target whose base is dense (a LoKr over a packed base stays structured).
fn dense_target_delta_bytes(host: &mut QwenImage21Transformer, targets: &[&str]) -> u64 {
    let mut bytes = 0;
    for target in targets {
        if !is_packed(host, target) {
            let (out, inp) = base_shape(host, target);
            bytes += (out * inp * 2) as u64;
        }
    }
    bytes
}

/// The contract for `spec`, checked against the plain contract of the same snapshot and tier:
/// the overlay is exactly `expected`, declared on the typed component axis, excluded from
/// `base_bytes`, and added to the resident total — and the contract conforms.
fn assert_overlay_priced(name: &str, spec: &LoadSpec, expected: u64) {
    let mut bare = spec.clone();
    bare.adapters.clear();
    let plain = memory_strategy_contract(ID, &bare).unwrap();
    assert_eq!(plain.asset_facts.overlay_bytes, 0, "{name}: bare overlay");
    assert!(plain.resident_components().is_empty(), "{name}");

    let contract = memory_strategy_contract(ID, spec).unwrap();
    assert!(expected > 0, "{name}: a non-empty stack prices above zero");
    assert_eq!(contract.asset_facts.overlay_bytes, expected, "{name}");
    assert_eq!(contract.auxiliary_resident_bytes(), expected, "{name}");
    assert!(
        contract
            .resident_components()
            .iter()
            .all(|c| c.kind == MemoryComponentKind::AdapterStack),
        "{name}"
    );
    assert!(
        contract.formula.uses(MemoryFormulaVariable::OverlayBytes),
        "{name}: the overlay must be load-bearing in the formula"
    );
    assert_eq!(
        contract.asset_facts.base_bytes, plain.asset_facts.base_bytes,
        "{name}: the overlay never enters base_bytes"
    );
    assert_eq!(
        contract.total_resident_bytes(),
        plain.total_resident_bytes() + expected,
        "{name}: the overlay reaches the resident total"
    );
    // The core validator's overlay legs (component sum == `overlay_bytes`, `OverlayBytes` in the
    // formula, `base_bytes` excluding the overlay) add no error the bare contract does not have.
    assert_eq!(
        contract.conformance_errors(),
        plain.conformance_errors(),
        "{name}"
    );
    gen_core_testkit::assert_memory_contract_facts_conform(&contract);
}

/// An adapter load prices what `load_heavy` keeps resident: every file's factors (residuals), plus a
/// full bf16 `[out, in]` delta for each LoKr module whose target base is dense — so the same LoKr
/// file prices differently on the dense DiT, a load-time Q8 DiT and a Q8 tier, exactly as their
/// packing differs. The expected deltas are read from the real DiT at each tier, not restated.
#[test]
fn an_adapter_load_prices_its_resident_overlay() {
    let tmp = tempfile::tempdir().unwrap();
    let mut probe = dense();
    let lora_file = peft_lora(tmp.path(), "lora.safetensors", &mut probe, &TARGETS, 1.0);
    let lokr_file = peft_lokr(tmp.path(), &mut probe, &PRICED_LOKR_TARGETS, 1.0);
    let q8 = Tier::Q8.selected_quant().unwrap();

    // A LoRA is residuals only: its file.
    assert_overlay_priced(
        "dense/lora",
        &snapshot_spec(tiny_snapshot()).with_adapters(vec![lora(&lora_file, 1.0)]),
        file_bytes(&lora_file),
    );

    // A LoKr over the dense DiT materializes every target.
    let dense_deltas = dense_target_delta_bytes(&mut dense(), &PRICED_LOKR_TARGETS);
    let all_deltas: u64 = PRICED_LOKR_TARGETS
        .iter()
        .map(|t| {
            let (out, inp) = base_shape(&mut probe, t);
            (out * inp * 2) as u64
        })
        .sum();
    assert_eq!(
        dense_deltas, all_deltas,
        "nothing is packed on the dense DiT"
    );
    assert_overlay_priced(
        "dense/lokr",
        &snapshot_spec(tiny_snapshot()).with_adapters(vec![lokr(&lokr_file, 1.0)]),
        file_bytes(&lokr_file) + dense_deltas,
    );

    // A load-time Q8 packs every Linear at least 32 wide: only `img_in` materializes.
    let load_time_deltas = dense_target_delta_bytes(&mut packed_q8(), &PRICED_LOKR_TARGETS);
    assert!(load_time_deltas > 0 && load_time_deltas < dense_deltas);
    assert_overlay_priced(
        "load-time-q8/lokr",
        &snapshot_spec(tiny_snapshot())
            .with_quant(q8)
            .with_adapters(vec![lokr(&lokr_file, 1.0)]),
        file_bytes(&lokr_file) + load_time_deltas,
    );

    // A Q8 tier packs only group-64 widths: `to_k` is dense there too.
    let tier = convert_tier(tmp.path(), Tier::Q8);
    let tier_deltas =
        dense_target_delta_bytes(&mut load_transformer(&tier).unwrap(), &PRICED_LOKR_TARGETS);
    assert!(tier_deltas > load_time_deltas && tier_deltas < dense_deltas);
    assert_overlay_priced(
        "q8-tier/lokr",
        &snapshot_spec(tier)
            .with_quant(q8)
            .with_adapters(vec![lokr(&lokr_file, 1.0)]),
        file_bytes(&lokr_file) + tier_deltas,
    );

    // A stack sums.
    assert_overlay_priced(
        "dense/lora+lokr",
        &snapshot_spec(tiny_snapshot())
            .with_adapters(vec![lora(&lora_file, 0.5), lokr(&lokr_file, 1.0)]),
        file_bytes(&lora_file) + file_bytes(&lokr_file) + dense_deltas,
    );
}

/// An adapter source that cannot be sized fails closed — in the contract and therefore in `load`,
/// under both policies — rather than pricing a zero the shared validator would wave through.
#[test]
fn an_unsizable_adapter_is_refused_rather_than_priced_at_zero() {
    let tmp = tempfile::tempdir().unwrap();
    let absent = tmp.path().join("absent.safetensors");
    let spec = snapshot_spec(tiny_snapshot()).with_adapters(vec![lora(&absent, 1.0)]);
    let err = memory_strategy_contract(ID, &spec)
        .expect_err("an unsizable adapter must refuse the contract")
        .to_string();
    assert!(err.contains("could not be sized"), "{err}");
    for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
        let err = match mlx_gen_qwen_image_2_1::provider_registry()
            .unwrap()
            .load(ID, &spec.clone().with_offload_policy(policy))
        {
            Ok(_) => panic!("{policy:?}: an unsizable adapter must fail the load"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("could not be sized"), "{policy:?}: {err}");
    }
}

// ── sc-24158: every adapter format, on the dense and the packed DiT ──────────────────────────────

/// [`write_safetensors`] with optional explicit per-entry values: `Some(v)` fills the tensor with
/// `v` (a per-module `.alpha`), `None` with the deterministic ramp.
fn write_safetensors_with(
    path: &Path,
    entries: &[(String, Vec<usize>, Option<f32>)],
    meta: &[(&str, &str)],
    amp: f32,
) {
    let mut data: Vec<u8> = Vec::new();
    let mut header = String::from("{\"__metadata__\":{\"format\":\"pt\"");
    for (k, v) in meta {
        header.push_str(&format!(",\"{k}\":\"{v}\""));
    }
    header.push('}');
    for (i, (name, shape, value)) in entries.iter().enumerate() {
        let n: usize = shape.iter().product();
        let start = data.len();
        for j in 0..n {
            let ramp = (((i * 131 + j * 17 + 7) % 101) as f32 / 101.0 - 0.5) * amp;
            data.extend_from_slice(&value.unwrap_or(ramp).to_le_bytes());
        }
        let dims = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            ",\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{dims}],\"data_offsets\":[{start},{}]}}",
            data.len()
        ));
    }
    header.push('}');
    let header = header.into_bytes();
    let mut buf = (header.len() as u64).to_le_bytes().to_vec();
    buf.extend_from_slice(&header);
    buf.extend_from_slice(&data);
    std::fs::write(path, buf).unwrap();
}

fn transformer_ns(path: &str) -> String {
    format!("transformer.{path}")
}

fn peft_wrapper(path: &str) -> String {
    format!("base_model.model.{path}")
}

fn diffusion_model_ns(path: &str) -> String {
    format!("diffusion_model.{path}")
}

fn kohya(path: &str) -> String {
    format!("lora_unet_{}", path.replace('.', "_"))
}

fn lycoris(path: &str) -> String {
    format!("lycoris_{}", path.replace('.', "_"))
}

fn bare(path: &str) -> String {
    path.to_string()
}

/// One LoRA spelling: how a dotted DiT path becomes the file's module key, and the factor suffixes
/// the convention writes.
struct LoraSpelling {
    name: &'static str,
    module: fn(&str) -> String,
    down: &'static str,
    up: &'static str,
    /// Whether the convention ships a per-module `.alpha` scalar.
    alpha: bool,
}

const LORA_SPELLINGS: [LoraSpelling; 7] = [
    LoraSpelling {
        name: "diffusers transformer.",
        module: transformer_ns,
        down: "lora_A.weight",
        up: "lora_B.weight",
        alpha: false,
    },
    LoraSpelling {
        name: "raw PEFT base_model.model.",
        module: peft_wrapper,
        down: "lora_A.weight",
        up: "lora_B.weight",
        alpha: false,
    },
    LoraSpelling {
        name: "PEFT .default adapter",
        module: transformer_ns,
        down: "lora_A.default.weight",
        up: "lora_B.default.weight",
        alpha: false,
    },
    LoraSpelling {
        name: "ComfyUI diffusion_model. lora_down/up",
        module: diffusion_model_ns,
        down: "lora_down.weight",
        up: "lora_up.weight",
        alpha: true,
    },
    LoraSpelling {
        name: "ai-toolkit diffusion_model. lora_A/B",
        module: diffusion_model_ns,
        down: "lora_A.weight",
        up: "lora_B.weight",
        alpha: false,
    },
    LoraSpelling {
        name: "kohya lora_unet_",
        module: kohya,
        down: "lora_down.weight",
        up: "lora_up.weight",
        alpha: true,
    },
    LoraSpelling {
        name: "bare (SceneWorks trainer)",
        module: bare,
        down: "lora_A.weight",
        up: "lora_B.weight",
        alpha: true,
    },
];

/// The globals and the block leaves whose names already carry `_` — the kohya flattening is
/// ambiguous for them unless resolved against the host's table.
const FORMAT_TARGETS: [&str; 8] = [
    "txt_in.in_layer",
    "time_text_embed.timestep_embedder.linear_1",
    "modulation.1",
    "transformer_blocks.0.attn.to_out.0",
    "transformer_blocks.1.img_mlp.gate_layer",
    "transformer_blocks.1.img_mlp.out",
    "norm_out.linear",
    "proj_out",
];

/// Neighbours of [`FORMAT_TARGETS`] a mis-resolved key could land on instead — they must stay bare.
const FORMAT_NEIGHBOURS: [&str; 4] = [
    "txt_in.out_layer",
    "time_text_embed.timestep_embedder.linear_2",
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.1.img_mlp.proj",
];

fn spelled_lora(
    dir: &Path,
    name: &str,
    host: &mut QwenImage21Transformer,
    spelling: &LoraSpelling,
    targets: &[&str],
) -> PathBuf {
    let mut entries = Vec::new();
    for target in targets {
        let (out, inp) = base_shape(host, target);
        let module = (spelling.module)(target);
        entries.push((format!("{module}.{}", spelling.down), vec![RANK, inp], None));
        entries.push((format!("{module}.{}", spelling.up), vec![out, RANK], None));
        if spelling.alpha {
            entries.push((format!("{module}.alpha"), vec![], Some(8.0)));
        }
    }
    let path = dir.join(name);
    write_safetensors_with(&path, &entries, &[], 1.0);
    path
}

/// Every LoRA spelling the shared loader accepts — PEFT / diffusers (`transformer.`, a raw
/// `base_model.model.` wrapper, the `.default` adapter name), ComfyUI and ai-toolkit
/// (`diffusion_model.`), kohya `lora_unet_` and the trainer's bare keys — applies over the globals
/// and the underscore-ambiguous leaves, on the dense DiT and as a residual over a load-time Q4 / Q8
/// one, landing on exactly the Linears it names.
#[test]
fn every_lora_spelling_applies_to_exactly_its_targets_on_dense_and_packed() {
    let tmp = tempfile::tempdir().unwrap();
    for (tier, build) in [
        ("dense", dense as Build),
        ("q4", packed_q4 as Build),
        ("q8", packed_q8 as Build),
    ] {
        let base = velocity(&build());
        for (i, spelling) in LORA_SPELLINGS.iter().enumerate() {
            let name = format!("{tier}/{}", spelling.name);
            let mut model = build();
            let file = spelled_lora(
                tmp.path(),
                &format!("{tier}-{i}.safetensors"),
                &mut model,
                spelling,
                &FORMAT_TARGETS,
            );
            let report = apply_qwen_image_2_1_adapters(&mut model, &[lora(&file, 1.0)])
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(report.applied, FORMAT_TARGETS.len(), "{name}");
            assert!(report.unmatched_paths.is_empty(), "{name}");
            for t in FORMAT_TARGETS {
                assert_eq!(adapter_count(&mut model, t), 1, "{name}: {t}");
            }
            for t in FORMAT_NEIGHBOURS {
                assert_eq!(
                    adapter_count(&mut model, t),
                    0,
                    "{name}: {t} must stay bare"
                );
            }
            if tier != "dense" {
                for t in ["transformer_blocks.1.img_mlp.out", "proj_out"] {
                    assert!(is_packed(&mut model, t), "{name}: {t} stays packed");
                }
            }
            assert_moved(&name, &velocity(&model), &base);
        }
    }
}

/// One third-party LyCORIS LoKr layout: its name, key spelling and factor set.
type LycorisLayout = (&'static str, fn(&str) -> String, LycorisFactors);

/// The factor set of one third-party LyCORIS LoKr module.
#[derive(Clone, Copy)]
enum LycorisFactors {
    /// Full `w1 [2, 2]`, low-rank `w2 = w2_a [out/2, 2] · w2_b [2, in/2]`, per-module alpha.
    LowRank,
    /// Both Kronecker factors full (LyCORIS then forces scale 1).
    Full,
    /// A Linear tucker right factor: `t2 [2, 2, 1, 1]`, `w2_a [2, out/2]`, `w2_b [2, in/2]`.
    Tucker,
    /// A spatial (conv) tucker core `[2, 2, 3, 3]` — no Linear form.
    ConvTucker,
}

fn lycoris_lokr_file(
    dir: &Path,
    name: &str,
    host: &mut QwenImage21Transformer,
    module: fn(&str) -> String,
    factors: LycorisFactors,
    targets: &[&str],
) -> PathBuf {
    let mut entries = Vec::new();
    for target in targets {
        let (out, inp) = match host.adaptable_facts(&target.split('.').collect::<Vec<_>>()) {
            Some(f) => (f.base_shape[0] as usize, f.base_shape[1] as usize),
            None => (32, 32),
        };
        let key = module(target);
        entries.push((format!("{key}.lokr_w1"), vec![2, 2], None));
        match factors {
            LycorisFactors::LowRank => {
                entries.push((format!("{key}.lokr_w2_a"), vec![out / 2, 2], None));
                entries.push((format!("{key}.lokr_w2_b"), vec![2, inp / 2], None));
                entries.push((format!("{key}.alpha"), vec![], Some(1.0)));
            }
            LycorisFactors::Full => {
                entries.push((format!("{key}.lokr_w2"), vec![out / 2, inp / 2], None));
            }
            LycorisFactors::Tucker | LycorisFactors::ConvTucker => {
                let kernel = if matches!(factors, LycorisFactors::Tucker) {
                    1
                } else {
                    3
                };
                entries.push((format!("{key}.lokr_t2"), vec![2, 2, kernel, kernel], None));
                entries.push((format!("{key}.lokr_w2_a"), vec![2, out / 2], None));
                entries.push((format!("{key}.lokr_w2_b"), vec![2, inp / 2], None));
                entries.push((format!("{key}.alpha"), vec![], Some(1.0)));
            }
        }
    }
    let path = dir.join(name);
    // No `networkType` stamp: a third-party file.
    write_safetensors_with(&path, &entries, &[], 1.0);
    path
}

/// Every unstamped LyCORIS LoKr layout — lycoris-lib `lycoris_…` and kohya `lora_unet_…` flattened
/// keys, ai-toolkit's dotted `diffusion_model.…`, a Linear tucker `lokr_t2` — routes through
/// `apply_lokr_thirdparty` and moves the velocity on the dense DiT and on a load-time Q4 / Q8 one,
/// whichever kind the caller declared (the file carries no stamp to read it from).
#[test]
fn every_lycoris_lokr_layout_applies_on_dense_and_packed() {
    let tmp = tempfile::tempdir().unwrap();
    let targets = [
        "transformer_blocks.0.attn.to_q",
        "txt_in.in_layer",
        "transformer_blocks.1.img_mlp.gate_layer",
        "norm_out.linear",
    ];
    let layouts: [LycorisLayout; 4] = [
        ("lycoris_ low-rank", lycoris, LycorisFactors::LowRank),
        ("lora_unet_ full", kohya, LycorisFactors::Full),
        (
            "ai-toolkit diffusion_model.",
            diffusion_model_ns,
            LycorisFactors::LowRank,
        ),
        ("lycoris_ tucker", lycoris, LycorisFactors::Tucker),
    ];
    for (tier, build) in [
        ("dense", dense as Build),
        ("q4", packed_q4 as Build),
        ("q8", packed_q8 as Build),
    ] {
        let base = velocity(&build());
        for (i, (layout, module, factors)) in layouts.into_iter().enumerate() {
            for spec_of in [lokr as fn(&Path, f32) -> AdapterSpec, lora] {
                let name = format!("{tier}/{layout}");
                let mut model = build();
                let file = lycoris_lokr_file(
                    tmp.path(),
                    &format!("{tier}-lycoris-{i}.safetensors"),
                    &mut model,
                    module,
                    factors,
                    &targets,
                );
                let report = apply_qwen_image_2_1_adapters(&mut model, &[spec_of(&file, 1.0)])
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(report.applied, targets.len(), "{name}");
                assert!(report.unmatched_paths.is_empty(), "{name}");
                for t in targets {
                    assert_eq!(adapter_count(&mut model, t), 1, "{name}: {t}");
                }
                assert_moved(&name, &velocity(&model), &base);
            }
        }
    }
}

/// A LyCORIS LoHa applies under its `lycoris_…`, kohya `lora_unet_…`, namespaced
/// `diffusion_model.…` and raw PEFT `base_model.model.…` spellings — materialized over the dense
/// DiT and over a load-time Q4 one alike (MLX has no LoHa refusal; candle folds it bf16-only).
#[test]
fn loha_applies_under_every_key_spelling_on_dense_and_packed() {
    let tmp = tempfile::tempdir().unwrap();
    let targets = ["transformer_blocks.0.attn.to_q", "proj_out"];
    let spellings: [fn(&str) -> String; 4] = [lycoris, kohya, diffusion_model_ns, peft_wrapper];
    for (tier, build) in [("dense", dense as Build), ("q4", packed_q4 as Build)] {
        let base = velocity(&build());
        for (i, module) in spellings.into_iter().enumerate() {
            let name = format!("{tier}/{}", module("x"));
            let mut model = build();
            let mut entries = Vec::new();
            for target in targets {
                let (out, inp) = base_shape(&mut model, target);
                let key = module(target);
                entries.push((format!("{key}.hada_w1_a"), vec![out, 2], None));
                entries.push((format!("{key}.hada_w1_b"), vec![2, inp], None));
                entries.push((format!("{key}.hada_w2_a"), vec![out, 2], None));
                entries.push((format!("{key}.hada_w2_b"), vec![2, inp], None));
                entries.push((format!("{key}.alpha"), vec![], Some(2.0)));
            }
            let file = tmp.path().join(format!("{tier}-loha-{i}.safetensors"));
            write_safetensors_with(&file, &entries, &[], 2.0);
            let report = apply_qwen_image_2_1_adapters(&mut model, &[lora(&file, 1.0)])
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(report.applied, targets.len(), "{name}");
            assert!(report.unmatched_paths.is_empty(), "{name}");
            assert_moved(&name, &velocity(&model), &base);
        }
    }
}

/// The key / metadata layout the MLX trainer (sc-24159) writes — bare dotted keys; LoRA
/// `lora_A`/`lora_B` plus a `[1]` `.alpha` under `networkType=lora`; LoKr `lokr_w1` plus low-rank
/// `lokr_w2_a`/`lokr_w2_b` (the trainer's `factorization(dim, -1)` split) under `networkType=lokr`
/// with `rank`/`alpha`/`decomposeFactor` and the provenance stamp — applies on the dense DiT and
/// over a load-time Q4 one. The candle twin pins the same layout loading there unconverted.
#[test]
fn the_trainer_output_layout_applies_on_dense_and_packed() {
    let tmp = tempfile::tempdir().unwrap();
    let targets = [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.1.img_mlp.proj",
        "transformer_blocks.1.img_mlp.out",
    ];
    let provenance = [
        ("family", "qwen-image-2-1"),
        ("baseModel", "qwen_image_2_1"),
        ("ss_base_model_version", "qwen_image_2_1"),
    ];
    for (tier, build) in [("dense", dense as Build), ("q4", packed_q4 as Build)] {
        let base = velocity(&build());
        for network in ["lora", "lokr"] {
            let name = format!("{tier}/trainer-{network}");
            let mut model = build();
            let mut entries = Vec::new();
            for target in targets {
                let (out, inp) = base_shape(&mut model, target);
                if network == "lora" {
                    entries.push((format!("{target}.lora_A.weight"), vec![RANK, inp], None));
                    entries.push((format!("{target}.lora_B.weight"), vec![out, RANK], None));
                    entries.push((format!("{target}.alpha"), vec![1], Some(RANK as f32)));
                } else {
                    let (out_a, out_b) = mlx_gen::train::lora::factorization(out as i32, -1);
                    let (in_a, in_b) = mlx_gen::train::lora::factorization(inp as i32, -1);
                    let (out_a, out_b) = (out_a as usize, out_b as usize);
                    let (in_a, in_b) = (in_a as usize, in_b as usize);
                    entries.push((format!("{target}.lokr_w1"), vec![out_a, in_a], None));
                    entries.push((format!("{target}.lokr_w2_a"), vec![out_b, 2], None));
                    entries.push((format!("{target}.lokr_w2_b"), vec![2, in_b], None));
                }
            }
            let rank = RANK.to_string();
            let mut meta: Vec<(&str, &str)> = provenance.to_vec();
            meta.extend([
                ("networkType", network),
                ("rank", rank.as_str()),
                ("alpha", rank.as_str()),
            ]);
            if network == "lokr" {
                meta.push(("decomposeFactor", "-1"));
            }
            let file = tmp
                .path()
                .join(format!("{tier}-trainer-{network}.safetensors"));
            write_safetensors_with(&file, &entries, &meta, 1.0);
            let spec = if network == "lora" {
                lora(&file, 1.0)
            } else {
                lokr(&file, 1.0)
            };
            let report = apply_qwen_image_2_1_adapters(&mut model, &[spec])
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(report.applied, targets.len(), "{name}");
            assert!(report.unmatched_paths.is_empty(), "{name}");
            assert_moved(&name, &velocity(&model), &base);
        }
    }
}

/// Files the DiT cannot serve are refused, never silently skipped: a LyCORIS LoKr whose module
/// reaches no Linear (refused by name), and a spatial (conv) tucker LoKr, which has no Linear
/// reconstruction.
#[test]
fn unservable_lycoris_lokr_files_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let mut model = dense();
    let unmatched = lycoris_lokr_file(
        tmp.path(),
        "unmatched.safetensors",
        &mut model,
        lycoris,
        LycorisFactors::Full,
        &[
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.9.attn.to_q",
        ],
    );
    let err = apply_qwen_image_2_1_adapters(&mut model, &[lokr(&unmatched, 1.0)])
        .expect_err("an unmatched LyCORIS module must refuse the install")
        .to_string();
    assert!(
        err.contains("lycoris_transformer_blocks_9_attn_to_q"),
        "the refusal names the key: {err}"
    );

    let mut model = dense();
    let conv = lycoris_lokr_file(
        tmp.path(),
        "conv.safetensors",
        &mut model,
        lycoris,
        LycorisFactors::ConvTucker,
        &["transformer_blocks.0.attn.to_q"],
    );
    assert!(
        apply_qwen_image_2_1_adapters(&mut model, &[lokr(&conv, 1.0)]).is_err(),
        "a conv tucker LoKr has no Linear form"
    );
}

/// One LoRA factor spelled twice in a file (`lora_A` beside `lora_A.default`) is refused rather than
/// keeping whichever key the loader read last.
#[test]
fn a_lora_factor_spelled_twice_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let mut model = dense();
    let target = "transformer_blocks.0.attn.to_q";
    let (out, inp) = base_shape(&mut model, target);
    let file = tmp.path().join("twice.safetensors");
    write_safetensors_with(
        &file,
        &[
            (
                format!("transformer.{target}.lora_A.weight"),
                vec![RANK, inp],
                None,
            ),
            (
                format!("transformer.{target}.lora_A.default.weight"),
                vec![RANK, inp],
                None,
            ),
            (
                format!("transformer.{target}.lora_B.weight"),
                vec![out, RANK],
                None,
            ),
        ],
        &[],
        1.0,
    );
    let err = apply_qwen_image_2_1_adapters(&mut model, &[lora(&file, 1.0)])
        .expect_err("a doubly-spelled factor must refuse the install")
        .to_string();
    assert!(err.contains("spelled twice"), "{err}");
}

/// Two LyCORIS spellings of one Linear in one file (`lycoris_X` beside `lora_unet_X`) are refused
/// rather than both installing and double-applying the delta.
#[test]
fn two_lycoris_spellings_of_one_linear_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let mut model = dense();
    let target = "transformer_blocks.0.attn.to_q";
    let (out, inp) = base_shape(&mut model, target);
    let mut entries = Vec::new();
    for key in [lycoris(target), kohya(target)] {
        entries.push((format!("{key}.lokr_w1"), vec![2, 2], None));
        entries.push((format!("{key}.lokr_w2"), vec![out / 2, inp / 2], None));
    }
    let file = tmp.path().join("both.safetensors");
    write_safetensors_with(&file, &entries, &[], 1.0);
    let err = apply_qwen_image_2_1_adapters(&mut model, &[lokr(&file, 1.0)])
        .expect_err("two spellings of one Linear must refuse the install")
        .to_string();
    assert!(err.contains("resolve to the same module"), "{err}");
    assert_eq!(adapter_count(&mut model, target), 0, "nothing installed");
}
