//! sc-16956 — candle FLUX.1 per-step latent **preview** real-weight validation (epic 16948).
//!
//! Four things a shape-only smoke cannot establish, and which this epic requires of every wiring
//! story:
//!
//! 1. **The reused fit belongs to this latent space.** FLUX.1 adds no fit; `crate::preview` reuses the
//!    epic-16624 sixteen-channel constants `mlx-gen-flux` committed.
//!    [`the_flux1_family_ships_one_learned_vae_in_three_containers`] is the reuse gate, and it has to
//!    be a *tensor* comparison rather than a hash equality three times over: the fit donor is a
//!    q4-packed diffusers container, the shipped Chroma/FLUX.1 diffusers container is plain bf16, and
//!    the dense candle path loads a **BFL-named f32** file whose keys do not even spell the same
//!    modules.
//! 2. **Chroma and PuLID share it.** [`the_chroma_vaes_are_byte_identical_to_the_flux1_one`] is the
//!    Chroma half — a straight hash equality, because those really are one file. PuLID has no VAE row
//!    at all *by construction*: it composes this crate's own `FluxRefBackbone`, so it loads whichever
//!    container the rows here already cover.
//! 3. **The frames actually develop.** [`flux1_preview_frames_evolve_toward_the_final_image`] drives
//!    the registered route through the `Generator` seam with a live sink, checks numbering, checks
//!    seeded byte-identity against an inert render, and measures that every frame is closer to — and
//!    more like — the finished image than the one before it. The strip is written out for review.
//! 4. **One frame per OUTER step on a multi-eval solver.**
//!    [`a_multi_eval_solver_emits_one_frame_per_outer_step`] runs `heun` and proves the guard is
//!    non-vacuous first: the shared driver calls `on_progress` once per *evaluation*, so a two-eval
//!    solver must produce strictly more progress events than steps before "frames == steps" means
//!    anything.
//!
//! [`the_boogu_vae_is_the_flux1_one`] is the sc-17218 unblock: Boogu's 16-channel `AutoencoderKL` is
//! **this** learned autoencoder, not a second 16-channel space, so the fit in `crate::preview` is the
//! one Boogu should reuse — the opposite finding to sc-16955's, which correctly withheld Boogu from the
//! 32-channel FLUX.2 fit.
//!
//! [`the_flow_cohort_needs_no_sigma_correction`] is the row for this family's σ convention. It is the
//! **only non-`#[ignore]`d row in this file** and runs on the committed constants alone — deliberately,
//! because sc-16954 shipped a red row that hid behind `-- --ignored` excluding it. Run this file both
//! ways.
//!
//! ```sh
//! FLUX1_PREVIEW_SNAPSHOT=E:\huggingface\hub\models--SceneWorks--flux1-dev-mlx\snapshots\<rev>\q4 \
//! FLUX1_FIT_VAE=...\models--SceneWorks--flux1-dev-mlx\snapshots\<rev>\q4\vae\model.safetensors \
//! FLUX1_DIFFUSERS_VAE=...\models--black-forest-labs--FLUX.1-dev\snapshots\<rev>\vae\diffusion_pytorch_model.safetensors \
//! FLUX1_BFL_VAE=...\models--black-forest-labs--FLUX.1-dev\snapshots\<rev>\ae.safetensors \
//! FLUX1_CHROMA_VAES="<hd vae>;<base vae>;<flash vae>" \
//! FLUX1_BOOGU_VAE=...\models--Boogu--Boogu-Image-0.1-Turbo\snapshots\<rev>\vae\diffusion_pytorch_model.safetensors \
//! FLUX1_PREVIEW_ARTIFACT_DIR=E:\out\sc-16956 \
//!   cargo test -p candle-gen-flux --release --features cuda --test integration preview_real_weights:: \
//!     -- --ignored --nocapture
//! ```
//!
//! Every input is **required** by the row that uses it: a row that early-returns on an unset variable
//! still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran and
//! proved something. Asking for `--ignored` is already the opt-in.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame, PreviewSink, Progress,
    WeightsSource,
};

const PROMPT: &str =
    "A weathered lighthouse on a rocky headland at golden hour, warm sunlight, dramatic clouds, \
     highly detailed photograph.";
const SEED: u64 = 16956;

/// The SHA-256 of the container the epic-16624 sixteen-channel fit was measured against —
/// `SceneWorks/flux1-dev-mlx` @ `323fd12d79f78ad444e882e8d8e871914584f2b9`, **q4** tier,
/// `vae/model.safetensors`, 164,654,042 bytes, 260 tensors.
///
/// Diffusers layout: 244 learned bf16 tensors plus the 16 `scales`/`biases` arrays of the eight
/// mid-block attention linears the MLX packer quantized. `mlx-gen-flux/src/preview.rs` names exactly
/// this file, so it is the anchor every other container is measured against.
const FIT_VAE_SHA256: &str = "e510ed25d48de4a52d5e00189e7ad57f346c16a167ad0388fc8c90e0cf5e4823";

/// The SHA-256 of the plain **bf16 diffusers** container — 167,666,902 bytes, 244 tensors.
///
/// Published byte-identically by `black-forest-labs/FLUX.1-dev`
/// @ `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`, `black-forest-labs/FLUX.1-schnell`
/// @ `741f7c3ce8b383c54771c7003378a50191e9efe9`, and all three Chroma re-hosts (see
/// [`the_chroma_vaes_are_byte_identical_to_the_flux1_one`]). Four repos, one file.
const DIFFUSERS_VAE_SHA256: &str =
    "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3";

/// The SHA-256 of the **BFL-named f32** container — `ae.safetensors`, 335,304,388 bytes, 244 tensors,
/// byte-identical between `FLUX.1-dev` and `FLUX.1-schnell`.
///
/// This is what `crate::vae::native::AutoEncoder` loads on the dense path, and its keys are the BFL
/// spelling (`decoder.mid.attn_1.q`, `decoder.up.{i}.block.{j}`) rather than the diffusers one. A hash
/// equality is impossible here by construction, which is the whole reason for the key-mapped
/// comparison below.
const BFL_VAE_SHA256: &str = "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38";

/// The SHA-256 of `Boogu/Boogu-Image-0.1-Turbo` @ `7c475e94ddb10529daa9142942d297675dde1acc`'s
/// `vae/diffusion_pytorch_model.safetensors` — 244 **f32** tensors, no `bn.*` stats.
///
/// sc-16955 measured this file against the FLUX.2 32-channel fit and correctly refused it. sc-16956
/// measures it against the FLUX.1 sixteen-channel one and finds the opposite — see
/// [`the_boogu_vae_is_the_flux1_one`].
const BOOGU_VAE_SHA256: &str = "8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94";

/// The measured extent of each comparison, pinned so a partial one cannot pass as a full one.
const VAE_TENSORS: usize = 244;
const VAE_VALUES: usize = 83_819_683;

/// The eight tensors the MLX q4/q8 packer replaces with `(codes, scales, biases)` — the mid-block
/// spatial self-attention's Q/K/V/out projections, encoder and decoder. Everything else in the fit
/// donor is plain bf16 and byte-identical to the diffusers container.
const PACKED_ATTENTION_LINEARS: &[&str] = &[
    "decoder.mid_block.attentions.0.to_k.weight",
    "decoder.mid_block.attentions.0.to_out.0.weight",
    "decoder.mid_block.attentions.0.to_q.weight",
    "decoder.mid_block.attentions.0.to_v.weight",
    "encoder.mid_block.attentions.0.to_k.weight",
    "encoder.mid_block.attentions.0.to_out.0.weight",
    "encoder.mid_block.attentions.0.to_q.weight",
    "encoder.mid_block.attentions.0.to_v.weight",
];

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

/// An input a row cannot run without. Missing means **fail**, not skip.
fn required_path(name: &str) -> PathBuf {
    env_path(name).unwrap_or_else(|| {
        panic!(
            "{name} must be set for this row — skipping it would report success while proving \
             nothing"
        )
    })
}

fn env_u32(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn artifact_dir() -> PathBuf {
    env_path("FLUX1_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("flux1_preview_sc16956"))
}

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash the file");
    format!("{:x}", hasher.finalize())
}

/// Load a `.safetensors` file's tensors, keyed and ordered by name.
fn tensors_of(path: &Path) -> BTreeMap<String, Tensor> {
    candle_gen::candle_core::safetensors::load(path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
        .into_iter()
        .collect()
}

/// A tensor widened to f32, exactly.
///
/// Comparing widened values is equivalent to comparing the 16-bit patterns: bf16 → f32 is lossless and
/// injective, so two bf16 tensors widen to equal f32 vectors **iff** their bit patterns match (weights
/// carry no NaN, the only case where that equivalence would leak). Doing it this way keeps the
/// comparison inside candle rather than adding a `half` dependency for a test.
fn widened(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("widen a tensor to f32")
}

// ── Provenance: the reuse is grounded in tensor bytes ─────────────────────────────────────────────

/// **The reuse gate.** The FLUX.1 family ships one learned VAE in three containers, and the fit donor
/// is the q4-packed one.
///
/// This is the row a hash equality could not have replaced, in two independent ways:
///
/// * The fit donor is the **q4** tier, so eight of its 244 learned tensors are `U32` code blocks with
///   separate `scales`/`biases`. Those eight are named explicitly and excluded by name; the other 236
///   must be **byte-identical** to the plain diffusers container. Excluding them by name rather than by
///   "whatever failed" is what stops the row from degrading into "compare the tensors that happen to
///   agree".
/// * The dense candle path loads `ae.safetensors`, whose keys are the **BFL** spelling. Its 244 tensors
///   are mapped onto the diffusers naming — including the reversed `up`-block index order and the
///   1×1-conv-vs-Linear attention shapes — and each must round, round-to-nearest-even, onto the
///   diffusers container's bf16 bits.
///
/// Three containers, one learned autoencoder, so the fit's input domain is the same whichever tier a
/// FLUX.1 render loads.
#[test]
#[ignore = "needs FLUX1_FIT_VAE + FLUX1_DIFFUSERS_VAE + FLUX1_BFL_VAE; run with --ignored"]
fn the_flux1_family_ships_one_learned_vae_in_three_containers() {
    let fit_path = required_path("FLUX1_FIT_VAE");
    let diffusers_path = required_path("FLUX1_DIFFUSERS_VAE");
    let bfl_path = required_path("FLUX1_BFL_VAE");

    for (label, path, expected, bytes) in [
        ("fit donor (q4)", &fit_path, FIT_VAE_SHA256, 164_654_042u64),
        (
            "diffusers bf16",
            &diffusers_path,
            DIFFUSERS_VAE_SHA256,
            167_666_902,
        ),
        ("BFL f32", &bfl_path, BFL_VAE_SHA256, 335_304_388),
    ] {
        let sha = sha256_of(path);
        eprintln!("  {label:<15}: {sha}  {}", path.display());
        assert_eq!(
            sha, expected,
            "{label} moved — re-derive the fit with mlx-gen-flux/tests/fit_preview_rgb.rs before \
             reusing it"
        );
        assert_eq!(std::fs::metadata(path).expect("stat").len(), bytes);
    }
    assert_ne!(FIT_VAE_SHA256, DIFFUSERS_VAE_SHA256);
    assert_ne!(DIFFUSERS_VAE_SHA256, BFL_VAE_SHA256);

    // --- the q4 fit donor against the plain diffusers container ---------------------------------
    let (fit, diffusers) = (tensors_of(&fit_path), tensors_of(&diffusers_path));
    assert_eq!(diffusers.len(), VAE_TENSORS);
    assert_eq!(
        fit.len(),
        VAE_TENSORS + 2 * PACKED_ATTENTION_LINEARS.len(),
        "the q4 tier is the 244 learned tensors plus a scales+biases pair per packed linear"
    );
    for name in PACKED_ATTENTION_LINEARS {
        for suffix in ["scales", "biases"] {
            let key = name.replace(".weight", &format!(".{suffix}"));
            assert!(fit.contains_key(&key), "{key} missing from the q4 tier");
            assert!(!diffusers.contains_key(&key));
        }
    }

    let mut identical = 0usize;
    for (key, packed) in &diffusers {
        let donor = fit
            .get(key)
            .unwrap_or_else(|| panic!("{key} missing from the fit donor"));
        if PACKED_ATTENTION_LINEARS.contains(&key.as_str()) {
            // Quantized in the donor: a `[512, 512]` bf16 matrix becomes a `[512, 64]` U32 code block.
            assert_eq!(donor.dtype(), DType::U32, "{key} must be the packed form");
            assert_eq!(packed.dtype(), DType::BF16);
            continue;
        }
        assert_eq!(donor.dims(), packed.dims(), "{key}: shapes must match");
        assert_eq!(donor.dtype(), DType::BF16, "{key}");
        assert_eq!(
            widened(donor),
            widened(packed),
            "{key}: the fit donor and the shipped diffusers container must be bit-identical"
        );
        identical += 1;
    }
    assert_eq!(
        identical,
        VAE_TENSORS - PACKED_ATTENTION_LINEARS.len(),
        "every learned tensor outside the eight packed attention linears must be identical"
    );
    eprintln!(
        "  fit donor vs diffusers: {identical} of {VAE_TENSORS} tensors bit-identical, {} quantized",
        PACKED_ATTENTION_LINEARS.len()
    );

    // --- the BFL f32 container against the same diffusers container ------------------------------
    let bfl = tensors_of(&bfl_path);
    assert_eq!(bfl.len(), VAE_TENSORS);
    let mut values = 0usize;
    for (key, wide) in &bfl {
        let mapped = bfl_to_diffusers(key);
        let narrow = diffusers.get(&mapped).unwrap_or_else(|| {
            panic!("{key} maps to {mapped}, which the diffusers container lacks")
        });
        assert_eq!(wide.dtype(), DType::F32, "{key}");
        assert_eq!(narrow.dtype(), DType::BF16, "{mapped}");
        assert_eq!(
            squeezed(wide.dims()),
            squeezed(narrow.dims()),
            "{key} -> {mapped}: shapes must match once the attention 1x1 conv axes are squeezed"
        );
        // The comparison IS the bf16 cast: widening the bf16 side instead would silently accept an f32
        // value that merely rounds close, rather than one that rounds exactly onto these bits.
        let cast = wide.to_dtype(DType::BF16).expect("cast the f32 tensor");
        let (a, b) = (widened(&cast), widened(narrow));
        assert_eq!(
            a, b,
            "{key} -> {mapped}: the f32 tensor does not round onto the bf16 one"
        );
        values += a.len();
    }
    assert_eq!(
        values, VAE_VALUES,
        "the comparison must cover every learned value"
    );
    eprintln!(
        "  BFL f32 vs diffusers: {VAE_TENSORS} tensors, {values} values, bf16-round-identical"
    );

    assert_eq!(candle_gen_flux::preview::PREVIEW_LATENT_CHANNELS, 16);
    assert_eq!(candle_gen_flux::preview::PACKED_LATENT_CHANNELS, 64);
}

/// Dimensions with the attention modules' trailing 1×1 conv axes dropped, so a BFL `[C, C, 1, 1]`
/// conv and a diffusers `[C, C]` linear compare equal.
fn squeezed(dims: &[usize]) -> Vec<usize> {
    if dims.len() == 4 && dims[2] == 1 && dims[3] == 1 {
        return dims[..2].to_vec();
    }
    dims.to_vec()
}

/// Map a BFL `ae.safetensors` key onto its diffusers `AutoencoderKL` name.
///
/// Three renamings and one **reordering**, all of them load-bearing:
/// `mid.block_{1,2}` → `mid_block.resnets.{0,1}`; `mid.attn_1.{q,k,v,proj_out,norm}` →
/// `mid_block.attentions.0.{to_q,to_k,to_v,to_out.0,group_norm}`; `nin_shortcut` → `conv_shortcut`;
/// `norm_out` → `conv_norm_out`; and the decoder's `up.{i}` blocks run **lowest-resolution-first** in
/// BFL and highest-first in diffusers, hence the `3 - i`. The encoder's `down.{i}` blocks are in the
/// same order in both.
///
/// An unmapped or wrongly-mapped key is a hard failure at the call site rather than a skip, so a
/// renaming that silently dropped half the tensors could not read as a pass.
fn bfl_to_diffusers(key: &str) -> String {
    let attention = |side: &str, part: &str, suffix: &str| {
        let part = match part {
            "q" => "to_q",
            "k" => "to_k",
            "v" => "to_v",
            "proj_out" => "to_out.0",
            "norm" => "group_norm",
            other => panic!("unknown BFL attention part {other}"),
        };
        format!("{side}.mid_block.attentions.0.{part}.{suffix}")
    };
    let parts: Vec<&str> = key.split('.').collect();
    match parts.as_slice() {
        [side, "mid", "attn_1", part, suffix] => attention(side, part, suffix),
        [side, "mid", block, rest @ ..] if block.starts_with("block_") => {
            let index: usize = block
                .trim_start_matches("block_")
                .parse::<usize>()
                .expect("BFL mid block index")
                - 1;
            format!("{side}.mid_block.resnets.{index}.{}", rest.join("."))
        }
        ["decoder", "up", level, "block", block, rest @ ..] => {
            let level: usize = level.parse().expect("BFL up level");
            format!(
                "decoder.up_blocks.{}.resnets.{block}.{}",
                3 - level,
                rest.join(".").replace("nin_shortcut", "conv_shortcut")
            )
        }
        ["decoder", "up", level, "upsample", rest @ ..] => {
            let level: usize = level.parse().expect("BFL up level");
            format!(
                "decoder.up_blocks.{}.upsamplers.0.{}",
                3 - level,
                rest.join(".")
            )
        }
        ["encoder", "down", level, "block", block, rest @ ..] => format!(
            "encoder.down_blocks.{level}.resnets.{block}.{}",
            rest.join(".").replace("nin_shortcut", "conv_shortcut")
        ),
        ["encoder", "down", level, "downsample", rest @ ..] => {
            format!(
                "encoder.down_blocks.{level}.downsamplers.0.{}",
                rest.join(".")
            )
        }
        [side, "norm_out", suffix] => format!("{side}.conv_norm_out.{suffix}"),
        _ => key.to_string(),
    }
}

/// The Chroma half of the adjudication: all three shipped Chroma re-hosts publish a VAE
/// **byte-identical** to `black-forest-labs/FLUX.1-dev`'s.
///
/// A hash equality is the right instrument here, unlike everywhere else in this file, because these
/// genuinely are one file republished — and it is the strongest available statement, so weakening it to
/// a tensor comparison would prove less. `FLUX1_CHROMA_VAES` is a `;`-separated list so a re-host that
/// stopped matching is a named failure rather than an untested tier.
#[test]
#[ignore = "needs FLUX1_CHROMA_VAES + FLUX1_DIFFUSERS_VAE; run with --ignored"]
fn the_chroma_vaes_are_byte_identical_to_the_flux1_one() {
    let listed = std::env::var("FLUX1_CHROMA_VAES").unwrap_or_else(|_| {
        panic!(
            "FLUX1_CHROMA_VAES must name every shipped Chroma VAE (`;`-separated) — skipping it \
             would report success while proving nothing"
        )
    });
    let paths: Vec<PathBuf> = listed
        .split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    assert_eq!(
        paths.len(),
        3,
        "Chroma ships three variants (hd / base / flash); all three must be measured"
    );

    let reference = sha256_of(&required_path("FLUX1_DIFFUSERS_VAE"));
    assert_eq!(reference, DIFFUSERS_VAE_SHA256);
    for path in &paths {
        let sha = sha256_of(path);
        eprintln!("  chroma vae: {sha}  {}", path.display());
        assert_eq!(
            sha,
            reference,
            "{} is not the FLUX.1 VAE — Chroma may not reuse the FLUX.1 fit unless it is",
            path.display()
        );
        assert_eq!(std::fs::metadata(path).expect("stat").len(), 167_666_902);
    }
    // That `candle-gen-chroma` reuses THIS crate's constants rather than a copy is pinned in Chroma's
    // own suite (`the_reused_fit_is_the_sixteen_channel_flux1_one`); asserting it from here would need
    // a dev-dependency cycle back onto the crate that depends on this one.
}

/// **The sc-17218 unblock.** Boogu's 16-channel `AutoencoderKL` is this learned autoencoder.
///
/// sc-16955 measured this same file against the FLUX.2 **32-channel** fit and correctly refused it: a
/// different architecture (no `bn.*` stats), a different channel count, a different file. What that
/// story could not say is *which* 16-channel fit does apply — "16 channels" alone does not make two
/// latent spaces the same, and Z-Image (sc-16957) is the other candidate. This row answers it: all 244
/// of Boogu's f32 tensors round, round-to-nearest-even, exactly onto the FLUX.1 diffusers container's
/// bf16 bits, key for key, with the same `latent_channels` / `scaling_factor` / `shift_factor`. Same
/// weights, same scale, same space — so `candle_gen_flux::preview` is the seam sc-17218 should reuse.
///
/// Left as a row rather than as prose so that a snapshot swap which made them *stop* matching would be
/// noticed before a borrowed fit shipped.
#[test]
#[ignore = "needs FLUX1_BOOGU_VAE + FLUX1_DIFFUSERS_VAE; run with --ignored"]
fn the_boogu_vae_is_the_flux1_one() {
    let boogu_path = required_path("FLUX1_BOOGU_VAE");
    let diffusers_path = required_path("FLUX1_DIFFUSERS_VAE");
    let sha = sha256_of(&boogu_path);
    eprintln!("  boogu vae: {sha}  {}", boogu_path.display());
    assert_eq!(sha, BOOGU_VAE_SHA256, "the staged Boogu VAE moved");
    assert_ne!(
        sha, DIFFUSERS_VAE_SHA256,
        "a different container — which is why this row compares tensors, not hashes"
    );

    let (boogu, diffusers) = (tensors_of(&boogu_path), tensors_of(&diffusers_path));
    assert_eq!(boogu.len(), VAE_TENSORS);
    assert_eq!(
        boogu.keys().collect::<Vec<_>>(),
        diffusers.keys().collect::<Vec<_>>(),
        "the two containers must hold the same key set"
    );
    assert!(
        !boogu.keys().any(|k| k.starts_with("bn.")),
        "a plain AutoencoderKL, as sc-16955 recorded"
    );
    assert_eq!(
        boogu["decoder.conv_in.weight"].dims()[1],
        candle_gen_flux::preview::PREVIEW_LATENT_CHANNELS,
        "Boogu decodes a 16-channel latent"
    );

    let mut values = 0usize;
    for (key, wide) in &boogu {
        let narrow = &diffusers[key];
        assert_eq!(wide.dims(), narrow.dims(), "{key}");
        assert_eq!(wide.dtype(), DType::F32, "{key}");
        let cast = wide.to_dtype(DType::BF16).expect("cast");
        assert_eq!(
            widened(&cast),
            widened(narrow),
            "{key}: Boogu's tensor does not round onto FLUX.1's — the two 16-channel spaces would \
             then be different and sc-17218 must not reuse this fit"
        );
        values += narrow.elem_count();
    }
    assert_eq!(values, VAE_VALUES);
    eprintln!("  boogu vs flux1: {VAE_TENSORS} tensors, {values} values, bf16-round-identical");
}

// ── Frame analysis helpers ────────────────────────────────────────────────────────────────────────

fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "compared buffers must match in length");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn correlation(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| v as f64).sum::<f64>() / n,
        b.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let mut num = 0.0;
    let (mut da, mut db) = (0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    if da == 0.0 || db == 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

/// Nearest-neighbour box resample of an RGB8 buffer.
fn downsample_raw(pixels: &[u8], src_w: u32, src_h: u32, w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let sx = (x as u64 * src_w as u64 / w as u64) as u32;
            let sy = (y as u64 * src_h as u64 / h as u64) as u32;
            let idx = ((sy * src_w + sx) * 3) as usize;
            out.extend_from_slice(&pixels[idx..idx + 3]);
        }
    }
    out
}

fn downsample(img: &Image, w: u32, h: u32) -> Vec<u8> {
    downsample_raw(&img.pixels, img.width, img.height, w, h)
}

fn save_png(dir: &Path, pixels: &[u8], width: u32, height: u32, name: &str) {
    std::fs::create_dir_all(dir).expect("create the artifact dir");
    let path = dir.join(name);
    let buf: image::RgbImage = image::ImageBuffer::from_raw(width, height, pixels.to_vec())
        .expect("frame buffer matches its dimensions");
    buf.save(&path)
        .unwrap_or_else(|e| panic!("save {path:?}: {e}"));
    eprintln!("  wrote {}", path.display());
}

/// Lay the strip out as one horizontal contact sheet so a reviewer sees the progression at a glance.
fn save_strip(dir: &Path, frames: &[PreviewFrame], name: &str) {
    assert!(!frames.is_empty(), "an empty strip cannot be written");
    let (fw, fh) = (frames[0].image.width, frames[0].image.height);
    let total_w = fw * frames.len() as u32;
    let mut sheet = vec![0u8; (total_w * fh * 3) as usize];
    for (i, frame) in frames.iter().enumerate() {
        let x0 = i as u32 * fw;
        for y in 0..fh {
            for x in 0..fw {
                let src = ((y * fw + x) * 3) as usize;
                let dst = (((y * total_w) + x0 + x) * 3) as usize;
                sheet[dst..dst + 3].copy_from_slice(&frame.image.pixels[src..src + 3]);
            }
        }
    }
    save_png(dir, &sheet, total_w, fh, name);
}

fn one_image(out: GenerationOutput) -> Image {
    let GenerationOutput::Images(mut images) = out else {
        panic!("expected GenerationOutput::Images");
    };
    assert_eq!(images.len(), 1, "these rows render a single image");
    images.pop().expect("one image")
}

fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&frames);
    let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
    (sink, frames)
}

/// Consecutive frames must not be the same picture. A floor rather than a ratio, and a low one: on an
/// 8-bit scale 0.1 mean |Δ| is unmistakably "not identical" while leaving room for the FIRST pair of a
/// flow strip, which is the smallest step of the whole trajectory. The strong statement about movement
/// is the monotone acceleration beside it.
const MIN_FRAME_MOVEMENT: f64 = 0.1;

/// The strip must close a meaningful share of its distance to the finished image. Expressed as a
/// fraction of the distance travelled rather than as a ratio of the endpoints, because the endpoints
/// carry the fit's irreducible residual.
const MIN_DISTANCE_FALL: f64 = 0.25;
/// The terminal previewed step must still carry at least this share of the strip's PEAK movement. It
/// replaces a strict "the last step is the largest" assertion, which measures the model rather than the
/// wiring - see the comment at the assertion.
const MIN_TERMINAL_SHARE: f64 = 0.5;

/// The shared strip analysis: numbering, latent resolution, per-frame movement, falling distance to the
/// finished image, and rising resemblance to it.
///
/// ## What the correlation floor is, and what it is NOT
///
/// A projection cannot correlate with the decode better than the fit does, so the fit fixes a
/// **ceiling**: the FLUX.1 fit's in-sample R² is `0.98224` (`mlx-gen-flux/src/preview.rs`), a
/// correlation ceiling of √0.98224 ≈ `0.991`. The in-sample R² is the like-for-like statistic — the
/// 16-channel QwenVae families were held against an in-sample 0.9586 and sc-16954 matched that with
/// SDXL's in-sample 0.91849 — so the holdout 0.92176 is deliberately not used here. (sc-16954 was
/// caught comparing an in-sample number against a holdout one, which produced a floor that was too
/// loose; this is the same statistic in both places.)
///
/// What the ceiling does **not** fix is the floor. `min_r_last` also measures *how far the trajectory
/// has travelled one step from the end*, which is a property of the **schedule**: the hook emits BEFORE
/// each solver step (sc-16949), so the final advancement is never previewed. FLUX.1-dev's time-shifted
/// flow schedule is back-loaded, so the unpreviewed terminal step is a real share of the trajectory —
/// FLUX.2 legitimately reached only +0.556 for exactly this reason while Qwen-Image reached +0.994.
/// Each lane's floor is therefore passed in and held to the same fraction of *this* fit's ceiling that
/// the measurement supports, and the load-bearing assertions are the three monotonicities plus the
/// ≥ 0.30 total rise, none of which a stale, duplicated or wrongly-scaled latent could reproduce.
#[allow(clippy::too_many_arguments)]
fn assert_the_strip_converges(
    label: &str,
    frames: &[PreviewFrame],
    final_image: &Image,
    steps: u32,
    latent_w: u32,
    latent_h: u32,
    min_r_last: f64,
    min_acceleration: f64,
) {
    assert_eq!(
        frames
            .iter()
            .map(|f| (f.current, f.total))
            .collect::<Vec<_>>(),
        (1..=steps).map(|n| (n, steps)).collect::<Vec<_>>(),
        "{label}: a {steps}-step render must emit exactly {steps} frames numbered 1..={steps}"
    );

    // Native-latent resolution, and batch 1. A CFG-fused `[2, …]` latent fails the packed-layout
    // contract outright, so a strip that exists at all is already proof the preview never saw a fused
    // unconditional half — there would be no frames if it had.
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (latent_w, latent_h),
            "{label}: frames must be native-VAE-latent resolution"
        );
    }

    // Every metric is computed and printed BEFORE anything is asserted, so one run reports the entire
    // strip rather than stopping at the first failing pair.
    let movement: Vec<f64> = frames
        .windows(2)
        .map(|p| mean_abs_delta(&p[0].image.pixels, &p[1].image.pixels))
        .collect();
    for (pair, delta) in frames.windows(2).zip(&movement) {
        eprintln!(
            "  {label} frame {:>2} → {:>2}: mean |Δ| {delta:.3}",
            pair[0].current, pair[1].current
        );
    }

    let target = downsample(final_image, latent_w, latent_h);
    let distances: Vec<f64> = frames
        .iter()
        .map(|f| mean_abs_delta(&f.image.pixels, &target))
        .collect();
    for (frame, distance) in frames.iter().zip(&distances) {
        eprintln!(
            "  {label} frame {:>2}: mean |Δ| to final {distance:.2}",
            frame.current
        );
    }

    // Absolute distance can only ever say "closer", never "resembles": the projection is a global
    // linear approximation of the decode, so even a perfectly converged latent keeps an offset and gain
    // error against the true pixels. Correlation over a coarse thumbnail, which averages the residual
    // away and leaves subject placement and colour masses, is what "the preview looks like the image"
    // actually means for a decorative frame.
    let coarse = 16u32;
    let coarse_target = downsample(final_image, coarse, coarse);
    let correlations: Vec<f64> = frames
        .iter()
        .map(|f| {
            correlation(
                &downsample_raw(
                    &f.image.pixels,
                    f.image.width,
                    f.image.height,
                    coarse,
                    coarse,
                ),
                &coarse_target,
            )
        })
        .collect();
    for (frame, r) in frames.iter().zip(&correlations) {
        eprintln!(
            "  {label} frame {:>2}: coarse correlation with final {r:+.3}",
            frame.current
        );
    }

    // 1. No two consecutive frames are the same picture.
    assert!(
        movement.iter().all(|d| *d > MIN_FRAME_MOVEMENT),
        "{label}: some consecutive frames are effectively identical: {movement:?}"
    );
    // Frame-to-frame movement ACCELERATES through the strip - the flow-match time-shifted schedule's
    // signature, and a far stronger statement than a flat floor: a hook reading a stale, duplicated or
    // wrongly scaled latent would not reproduce it.
    //
    // Two exclusions, both measured rather than assumed. The OPENING frames are near-pure noise
    // projected through a global linear map, so the mean |delta| between two of them carries sampling
    // noise comparable to the sigma step itself - hence the second half rather than the whole strip.
    // And the TERMINAL pair is excluded because whether it rises is a property of the model, not of the
    // wiring: on the same nominal 1024^2 x 12-step flow schedule, FLUX.1-dev rises into it
    // (9.729 -> 12.288) while Chroma HD (9.496 -> 8.797) and PuLID (15.989 -> 14.507) dip. By the last
    // previewed step the latent is nearly converged, so the projection's mean |delta| saturates even as
    // the sigma interval grows. Asserting it would be asserting the model.
    //
    // What replaces it is a floor on the terminal step as a share of the strip's PEAK movement, so a
    // genuine collapse - a hook that froze, or one projecting a stale latent - still fails.
    let rising = &movement[..movement.len() - 1];
    let back_half = &rising[rising.len() / 2..];
    assert!(
        back_half.windows(2).all(|p| p[1] > p[0]),
        "{label}: movement must rise monotonically over the second half of the strip, up to but not \
         including the terminal pair: {movement:?}"
    );
    let (opening, closing) = (movement[0], movement[movement.len() - 1]);
    let peak = movement.iter().copied().fold(f64::MIN, f64::max);
    eprintln!(
        "  {label}: movement {opening:.3} -> {closing:.3} ({:.1}x), peak {peak:.3}",
        closing / opening
    );
    assert!(
        closing > opening * min_acceleration,
        "{label}: the terminal step must dominate the opening one by at least {min_acceleration}x \
         ({opening:.3} -> {closing:.3})"
    );
    assert!(
        closing > peak * MIN_TERMINAL_SHARE,
        "{label}: the terminal step must still carry at least {MIN_TERMINAL_SHARE} of the strip's \
         peak movement ({closing:.3} vs peak {peak:.3}) - below that the strip has stalled"
    );

    // 3. The strip approaches the finished image, at every step and by a meaningful margin.
    let (first, last) = (distances[0], distances[distances.len() - 1]);
    let fall = (first - last) / first;
    eprintln!(
        "  {label}: distance fell {:.1}% ({first:.2} → {last:.2})",
        fall * 100.0
    );
    assert!(
        fall > MIN_DISTANCE_FALL,
        "{label}: the strip must converge on the final image (first {first:.2} → last {last:.2}, \
         fall {fall:.3}, floor {MIN_DISTANCE_FALL})"
    );
    assert!(
        distances.windows(2).all(|p| p[1] < p[0]),
        "{label}: distance to the finished image must fall at every step: {distances:?}"
    );

    // 4. The strip actually comes to resemble the render, monotonically.
    let (r_first, r_last) = (correlations[0], correlations[correlations.len() - 1]);
    assert!(
        r_last > min_r_last,
        "{label}: the last preview frame must resemble the finished render \
         (r {r_last:+.3}, floor {min_r_last:+.3})"
    );
    assert!(
        correlations.windows(2).all(|p| p[1] > p[0]),
        "{label}: resemblance must increase at every step: {correlations:?}"
    );
    // "The strip develops" is asserted as a **rise**, not as an absolute floor on the first frame:
    // correlation is taken over flattened RGB triplets, so it carries channel-mean structure as well as
    // spatial structure, and this fit's intercept is a near-neutral grey — a frame of pre-denoise noise
    // starts at a non-zero, scene-dependent floor. sc-16950's `r_first < 0.35` ceiling is deliberately
    // not ported; the rise plus a loose ceiling is what cannot be faked, since a strip that opened on
    // the finished image would have nowhere to rise to.
    assert!(
        r_first < 0.75,
        "{label}: the first frame is pre-denoise noise and must not already BE the render \
         (r {r_first:+.3})"
    );
    assert!(
        r_last - r_first > 0.30,
        "{label}: resemblance must actually develop across the strip \
         (first {r_first:+.3} → last {r_last:+.3})"
    );
}

// ── Driving the registered route ──────────────────────────────────────────────────────────────────

fn base_request(steps: u32, size: u32, sampler: Option<&str>) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(SEED),
        steps: Some(steps),
        sampler: sampler.map(str::to_string),
        ..GenerationRequest::default()
    }
}

/// Render the registered `flux1_dev` route twice on one warmed generator at the same seed — once inert,
/// once live — and hold the strip to [`assert_the_strip_converges`]. Returns the live run's
/// `Progress::Step` count (which IS its evaluation count) and its frames.
fn assert_flux1_previews_converge(
    label: &str,
    sampler: Option<&str>,
    steps: u32,
    size: u32,
    min_r_last: f64,
    min_acceleration: f64,
) -> (usize, Vec<PreviewFrame>) {
    eprintln!("── {label}: {size}² × {steps} steps, sampler {sampler:?}");
    let spec = LoadSpec::new(WeightsSource::Dir(required_path("FLUX1_PREVIEW_SNAPSHOT")));
    let generator = candle_gen_flux::load_dev(&spec).unwrap_or_else(|e| panic!("load flux1: {e}"));

    // N1: the inert baseline. Same generator, same seed, no sink.
    let mut noop = |_: Progress| {};
    let inert = one_image(
        generator
            .generate(&base_request(steps, size, sampler), &mut noop)
            .unwrap_or_else(|e| panic!("{label}: inert render: {e}")),
    );

    let (sink, frames) = collecting_sink();
    let mut request = base_request(steps, size, sampler);
    request.preview = sink;
    let mut events = 0usize;
    let mut count_progress = |p: Progress| {
        if matches!(p, Progress::Step { .. }) {
            events += 1;
        }
    };
    let live = one_image(
        generator
            .generate(&request, &mut count_progress)
            .unwrap_or_else(|e| panic!("{label}: live render: {e}")),
    );

    // An active sink must not move a single bit of the render.
    assert_eq!(
        inert.pixels, live.pixels,
        "{label}: attaching a live preview sink changed the seeded render"
    );

    let frames = candle_gen::lock_recover(&frames).clone();
    let dir = artifact_dir();
    assert_the_strip_converges(
        label,
        &frames,
        &live,
        steps,
        size / 8,
        size / 8,
        min_r_last,
        min_acceleration,
    );
    save_strip(&dir, &frames, &format!("{label}-strip.png"));
    save_png(
        &dir,
        &live.pixels,
        live.width,
        live.height,
        &format!("{label}-final.png"),
    );
    (events, frames)
}

/// The registered route's shipped lane: `run_flow_sampler` over `FlowModelSampling`, which is what every
/// FLUX.1 request takes, including one that names no sampler at all.
#[test]
#[ignore = "needs FLUX1_PREVIEW_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn flux1_preview_frames_evolve_toward_the_final_image() {
    let steps = env_u32("FLUX1_PREVIEW_STEPS", 12);
    let size = env_u32("FLUX1_PREVIEW_SIZE", 1024);
    // 0.90 against a measured +0.970 on this lane — 91% of the fit-derived 0.991 ceiling, so the floor
    // is a real backstop rather than a formality, while leaving room for a prompt whose composition
    // resolves later. See `assert_the_strip_converges` for why the floor is per-lane.
    assert_flux1_previews_converge("flux1-dev-euler", None, steps, size, 0.90, 2.0);
}

/// Exactly one frame per **outer** solver step on a multi-eval solver.
///
/// The guard is made non-vacuous first, and in the strongest available way: the shared driver calls
/// `on_progress` once per *evaluation* (`sampler.rs` computes the step count on every eval and
/// deliberately repeats it), so counting `Progress::Step` events IS counting evaluations. If `heun` did
/// not evaluate twice per step the event count would equal the step count and "frames == steps" would
/// prove nothing — so that inequality is asserted before the frame count is.
#[test]
#[ignore = "needs FLUX1_PREVIEW_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn a_multi_eval_solver_emits_one_frame_per_outer_step() {
    let steps = 8u32;
    let (events, _) =
        // 0.90 against a measured +0.961 on this 8-step 768 lane.
        assert_flux1_previews_converge("flux1-dev-heun", Some("heun"), steps, 768, 0.90, 1.5);
    eprintln!("  heun: {events} evaluations for {steps} outer steps");
    assert!(
        events > steps as usize,
        "heun must evaluate more than once per outer step or this row proves nothing about the \
         preview counter's dedup ({events} events for {steps} steps)"
    );
    // `assert_flux1_previews_converge` already required exactly `steps` frames numbered 1..=steps, so
    // the dedup collapsed the extra evaluations. Stated here because that is the point of the row.
}

/// This family's σ-convention finding, measured rather than argued — and the counterpart of sc-16954's
/// VE-correction row, which found the **opposite** for the discrete ε cohort.
///
/// `run_flow_sampler` integrates a `FlowModelSampling` whose `input_scale` is exactly `1.0` at every σ,
/// so the running latent already *is* the tensor the fit was measured against and no `with_sigma`
/// correction is needed. The cheap decisive signal sc-16954 named is the first frame's rail-clipped
/// fraction: SDXL's uncorrected projection clipped 89.4% of pixels to 0/255, which is what a missing
/// input scaling looks like. Here the same measurement is taken on the latent this family's first
/// emission actually sees — flow priors are unit-normal, `σ_max = 1.0` — and it must come out readable.
///
/// Runs on the committed constants alone, no weights, and is deliberately **not** `#[ignore]`d: it is
/// the row that must appear in a plain `cargo test` of this file. sc-16954 shipped a red row that hid
/// because the only non-ignored row in its file was excluded by `-- --ignored`.
#[test]
fn the_flow_cohort_needs_no_sigma_correction() {
    use candle_gen::gen_core::sampling::{FlowModelSampling, ModelSampling, TimestepConvention};

    // The convention, first: the claim is about `input_scale`, so it is read off the very
    // `ModelSampling` the driver integrates rather than asserted about the family in prose.
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    for sigma in [0.0f32, 0.05, 0.25, 0.5, 0.75, 1.0] {
        assert_eq!(
            ms.input_scale(sigma),
            1.0,
            "FlowModelSampling::input_scale must be identically 1.0; at {sigma} it is not, and this \
             family would need PreviewHook::with_sigma"
        );
    }

    // The consequence, measured. A unit-normal packed latent at σ_max = 1.0 is what the first emission
    // sees — FLUX.1's `seeded_noise` is exactly that, 2×2-packed.
    let (width, height) = (256u32, 256u32);
    let (rows, cols) = candle_gen_flux::preview::token_grid(width, height);
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(SEED);
    let noise = candle_gen::seeded_normal_vec(
        &mut rng,
        rows * cols * candle_gen_flux::preview::PACKED_LATENT_CHANNELS,
    );
    let tokens = Tensor::from_vec(
        noise,
        (
            1,
            rows * cols,
            candle_gen_flux::preview::PACKED_LATENT_CHANNELS,
        ),
        &Device::Cpu,
    )
    .expect("latent");

    let frame = candle_gen_flux::preview::project_packed_tokens(&tokens, width, height)
        .expect("project the first-emission latent");
    let rails = frame.pixels.iter().filter(|&&v| v == 0 || v == 255).count() as f64
        / frame.pixels.len() as f64;
    eprintln!("  flow prior at sigma_max: rail-clipped fraction {rails:.4}");
    // The bound is loose enough that a rounding change cannot flip it and far below sc-16954's
    // uncorrected SDXL 0.894, which is the number it is being contrasted with.
    assert!(
        rails < 0.05,
        "an uncorrected flow-space projection must already be a readable noise field, not a clipped \
         one ({rails:.4}) — if this ever fails, the family needs PreviewHook::with_sigma"
    );
}
