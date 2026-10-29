//! The tiny-DiT fold fixtures: deterministic tensors, synthetic turbo / PEFT LoRA files and the
//! probe residual.

use super::*;

/// The published rank — every one of the seven turbo files is rank 128. Used verbatim in the tiny
/// fixtures so the folds asserted below are the *published* numbers (1.0 and 0.0625) and not a
/// scaled-down analogue of them.
pub(super) const PUBLISHED_RANK: i32 = 128;

/// The module the numeric arms probe. `attn.to_q` is `[96, 64]` at the fixture geometry, so its
/// factors are `A [128, 64]` / `B [96, 128]`.
pub(super) const PROBE: &str = "transformer_blocks.0.attn.to_q";

/// A deterministic host-built tensor — no GPU RNG stream, so every arm sees identical bytes.
pub(super) fn tensor(shape: &[i32], seed: f32) -> Array {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let v: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.013 + seed).sin()) * 0.25)
        .collect();
    Array::from_slice(&v, shape)
}

/// Cast to `bf16` — the dtype **every** tensor in **every** published turbo file carries.
pub(super) fn bf16(a: &Array) -> Array {
    a.as_dtype(Dtype::Bfloat16).expect("cast to bf16")
}

/// The `[out, in]` logical shape of each of the six adaptable leaves at `cfg`'s geometry.
pub(super) fn target_shape(cfg: &MiniMaxH3DitConfig, target: &str) -> (i32, i32) {
    match target {
        t if t.ends_with("attn.to_q") || t.ends_with("attn.to_k") || t.ends_with("attn.to_v") => {
            (cfg.inner_dim(), cfg.hidden_size)
        }
        t if t.ends_with("attn.to_out.0") => (cfg.hidden_size, cfg.inner_dim()),
        // SwiGLU input emits `[value | gate]`, so twice `ffn_dim`.
        t if t.ends_with("ff.net.0.proj") => (2 * cfg.ffn_dim, cfg.hidden_size),
        t if t.ends_with("ff.net.2") => (cfg.hidden_size, cfg.ffn_dim),
        other => panic!("unknown target {other}"),
    }
}

/// Write a tiny **diffusers-key-space** turbo LoRA covering every adaptable module at `cfg`'s
/// geometry, keyed exactly as the published files are — `.lora_A.default.weight` /
/// `.lora_B.default.weight`, no namespace prefix — with `alpha` stamped into the top-level
/// `__metadata__` only when `alpha` is `Some`.
///
/// **The factors are written `bf16`, not `f32`.** All 624 tensors in every published turbo file are
/// bf16, and that is the only dtype under which the install's dtype-matched fold scalar
/// (`scalar(fold).as_dtype(up.dtype())`) has any effect at all — with f32 fixtures the cast is a
/// no-op and the property is unguarded (sc-18724 review). Every fold asserted below is an exact
/// power of two, so bf16 storage costs the assertions no exactness: scaling a bf16 factor by `1.0`
/// or `0.0625` is representable without rounding.
pub(super) fn write_lora(
    dir: &Path,
    name: &str,
    cfg: &MiniMaxH3DitConfig,
    alpha: Option<&str>,
) -> PathBuf {
    write_lora_with_meta(dir, name, cfg, alpha, &[])
}

/// [`write_lora`] plus arbitrary extra `__metadata__` entries — the `lora_adapter_metadata` blob
/// arms need one, and everything else about the file must stay identical so the arms differ in
/// exactly the metadata under test.
pub(super) fn write_lora_with_meta(
    dir: &Path,
    name: &str,
    cfg: &MiniMaxH3DitConfig,
    alpha: Option<&str>,
    extra: &[(&str, &str)],
) -> PathBuf {
    let path = dir.join(name);
    let mut arrays: Vec<(String, Array)> = Vec::new();
    for target in adapter_target_paths(cfg) {
        let (out, inn) = target_shape(cfg, &target);
        // A distinct seed per module, so a routing bug that folds the right factors onto the wrong
        // module is observable rather than symmetric.
        let seed = target.len() as f32;
        arrays.push((
            format!("{target}.lora_A.default.weight"),
            bf16(&tensor(&[PUBLISHED_RANK, inn], seed)),
        ));
        arrays.push((
            format!("{target}.lora_B.default.weight"),
            bf16(&tensor(&[out, PUBLISHED_RANK], seed + 0.5)),
        ));
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".into(), "pt".into());
    meta.insert("floating_dtype".into(), "bfloat16".into());
    if let Some(a) = alpha {
        meta.insert("alpha".into(), a.into());
    }
    for (k, v) in extra {
        meta.insert((*k).into(), (*v).into());
    }
    let entries: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(entries, Some(&meta), &path).expect("write the tiny turbo LoRA");
    path
}

pub(super) fn spec(path: PathBuf, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path, scale, AdapterKind::Lora)
}

/// The committed tiny DiT — real crate types, real loader, 2 transformer blocks + 2 refiner blocks.
pub(super) fn tiny_dit(cfg: &MiniMaxH3DitConfig) -> MiniMaxH3Dit {
    let mut w = Weights::from_file(DIT_FIXTURE).expect("the committed DiT fixture");
    MiniMaxH3Dit::from_weights(&mut w, cfg, Dtype::Float32).expect("the whole tiny DiT")
}

/// `y_with_lora(x) − y_base(x)` at [`PROBE`] — the residual the install actually added.
pub(super) fn probe_residual(
    cfg: &MiniMaxH3DitConfig,
    lora: &Path,
    scale: f32,
    x: &Array,
) -> Array {
    let segs: Vec<&str> = PROBE.split('.').collect();
    let mut base = tiny_dit(cfg);
    let y0 = base
        .adaptable_mut(&segs)
        .expect("probe module")
        .forward(x)
        .expect("base forward");

    let mut adapted = tiny_dit(cfg);
    let report = apply_minimax_h3_adapters(&mut adapted, &[spec(lora.to_path_buf(), scale)])
        .expect("install");
    assert_eq!(report.applied, adapter_target_paths(cfg).len());
    assert!(report.unmatched_paths.is_empty());
    let y1 = adapted
        .adaptable_mut(&segs)
        .expect("probe module")
        .forward(x)
        .expect("adapted forward");
    subtract(&y1, &y0).expect("residual")
}
