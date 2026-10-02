//! LoRA / LoKr on the Qwen-Image 2.1 DiT (sc-24156), on the committed miniature snapshot.
//!
//! Two layers:
//!
//! * **The host** (`AdaptableHost for QwenImage21Transformer`) — the key→module surface, the kohya
//!   flattened table, and the residual itself: an installed adapter changes the DiT's velocity on
//!   the dense base and on a load-time **Q4-packed** base (where it must stay a residual — the base
//!   remains packed), LoRA and LoKr alike, stacked across files with their own strengths; a strength
//!   of zero is byte-identical to no adapter. Unmatched keys are a strict refusal that names them.
//! * **The load path** (`provider_registry().load(..)` with `LoadSpec::adapters`) — the refusal is
//!   gone, the descriptor advertises LoRA + LoKr, and the adapters reach the text-to-image route
//!   (Resident and Sequential), the reference/edit route, and a pre-quantized Q4 tier.
//!
//! The adapter files are synthesized at test time against the host's own base shapes (read through
//! the probe half, `adaptable_facts`), with deterministic bounded values — no RNG, no real weights.

use std::path::{Path, PathBuf};

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core::Conditioning;
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, WeightsSource};
use mlx_gen_qwen_image_2_1::convert::prequantize_turnkey;
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

/// On a Q4-packed DiT the adapter is a residual over the packed base: the base stays packed, the
/// adapter is installed on it, and the velocity moves relative to the same packed DiT without it.
#[test]
fn a_lora_residual_changes_the_packed_q4_velocity_and_leaves_the_base_packed() {
    let tmp = tempfile::tempdir().unwrap();
    let base = velocity(&packed_q4());

    let mut model = packed_q4();
    for t in TARGETS {
        assert!(is_packed(&mut model, t), "{t} packs at Q4 on the tiny DiT");
    }
    let file = peft_lora(tmp.path(), "lora.safetensors", &mut model, &TARGETS, 1.0);
    let report = apply_qwen_image_2_1_adapters(&mut model, &[lora(&file, 1.0)]).unwrap();
    assert_eq!(report.applied, TARGETS.len());
    for t in TARGETS {
        assert!(
            is_packed(&mut model, t),
            "{t}: the install must not unpack the base"
        );
        assert_eq!(adapter_count(&mut model, t), 1, "{t}");
    }
    assert_moved("q4/lora", &velocity(&model), &base);
}

/// LoKr on both bases: a materialized delta over the dense DiT, the structured (never-materialized)
/// Kronecker residual over the packed one — both move the velocity.
#[test]
fn a_lokr_residual_changes_the_velocity_on_dense_and_packed_q4() {
    let tmp = tempfile::tempdir().unwrap();
    let targets = [
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.1.attn.to_out.0",
    ];
    let builds: [(&str, fn() -> QwenImage21Transformer); 2] = [("dense", dense), ("q4", packed_q4)];
    for (name, build) in builds {
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

/// A pre-quantized Q4 tier takes the same adapters as residuals over its packed DiT.
#[test]
fn adapters_reach_a_packed_q4_tier() {
    let tmp = tempfile::tempdir().unwrap();
    let file = render_lora(tmp.path());
    let tier = tmp.path().join(Tier::Q4.dir_name());
    prequantize_turnkey(&tiny_snapshot(), &tier, Tier::Q4).expect("the tiny snapshot converts");
    let quant = Tier::Q4.selected_quant().expect("Q4 selects a quant");
    let req = t2i(32);

    let plain = render(&snapshot_spec(tier.clone()).with_quant(quant), &req);
    let adapted = render(
        &snapshot_spec(tier)
            .with_quant(quant)
            .with_adapters(vec![lora(&file, 1.0)]),
        &req,
    );
    assert_ne!(
        adapted.pixels, plain.pixels,
        "the LoRA must change the packed-tier render"
    );
}

/// The load path is strict too: an unmatched key fails the load (Resident builds the DiT eagerly)
/// and the error names it.
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
    let spec = snapshot_spec(tiny_snapshot()).with_adapters(vec![lora(&file, 1.0)]);
    let err = match mlx_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(ID, &spec)
    {
        Ok(_) => panic!("an unmatched adapter key must fail the load"),
        Err(err) => err.to_string(),
    };
    assert!(
        err.contains("transformer_blocks.0.attn.to_add_out"),
        "{err}"
    );
}
