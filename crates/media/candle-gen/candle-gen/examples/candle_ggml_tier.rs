//! Family-generic offline builder for Candle-native GGML tiers (SW-7285). Converts published
//! MLX-affine tier components into stored GGML block tiers (see `candle_gen::ggml_tier`). NOT part
//! of CI: it reads real weights.
//!
//! ```text
//! # build (the default subcommand)
//! cargo run --release --example candle_ggml_tier -p candle-gen -- [build] \
//!   --src-root <root holding <tier>/<component> templates/MLX tiers> --out-root <output root> \
//!   --tier q4=from-mlx \
//!   --tier q8=from-bf16@<bf16 root holding <component>> \
//!   --component transformer --component text_encoder   (or --components a,b,c) \
//!   [--map-prefix TEMPLATE_PREFIX=SOURCE_PREFIX]...
//!
//! # header-only q8 template from a local MLX q8 component (reads 8+N bytes per shard)
//! ... -- dump-headers --from <q8 component dir> --to <template dir>
//! # ... or straight from huggingface.co via HTTP Range (needs curl on PATH)
//! ... -- fetch-headers --repo SceneWorks/<repo> --rev main --subdir q8/transformer \
//!        --to <template dir> [--use-hf-token]
//! ```
//!
//! * `--tier NAME=from-mlx` -- Q4: lossless Q4_1 from the MLX triple (group size read from the
//!   component `config.json`, any multiple of 32). A q8 tier via from-mlx is the legacy lossy requant.
//!   Each triple converts at the width its own shapes imply: a q4 component's 8-bit precision-floor
//!   triples (Mage's `norm_out.linear`) become Q8_0 exactly as the load-time repack builds them,
//!   each recorded as an anchored `overrides` entry; any other width is refused.
//! * `--tier NAME=from-bf16@ROOT` -- Q8_0 quantized from the dense bf16/f16/f32 source at
//!   `ROOT/<component>`; the MLX q8 component at `<src-root>/NAME/<component>` is only the structural
//!   template and may be header-only (see dump-headers / fetch-headers). With a header-only template
//!   the non-quantized tensors come from the dense source, cast to the template's dtype (reported).
//! * `NAME` may be several path segments (`distilled/q4`); output mirrors `<out>/<NAME>/<component>`.
//! * **No `--component`** = each `--tier` is a **split tier** (a flat `split_model.json` directory:
//!   the LTX-2.3 / LTX-2.5 tiers): `<src-root>/<NAME>` -> `<out-root>/<NAME>`, and the from-bf16
//!   root is the bf16 split tier directory itself (same file names), e.g.
//!   `--tier distilled/q4=from-mlx --tier distilled/q8=from-bf16@<snap>/distilled/bf16`.
//! * Stacked MoE experts (rank-3 triples, Lens's text encoder) convert to rank-4 stored blocks; a
//!   from-bf16 source that ships them MXFP4 (`{base}_blocks` / `{base}_scales`) is dequantized and
//!   quantized to Q8_0 (reported as `mxfp4`).
//! * `--keep-q8 REGEX` (repeatable) + `--keep-src BF16_ROOT` -- a q4 **recipe override** for
//!   from-mlx component tiers: every packed projection whose **base name** (tensor name without
//!   `.weight`, e.g. `transformer_blocks.3.img_mod.1`) matches a REGEX (unanchored; use `^...$`)
//!   is stored Q8_0 quantized from `BF16_ROOT/<component>/{base}.weight` (same quantizer as
//!   from-bf16, `--map-prefix` applies); the rest stay lossless Q4_1. A pattern matching nothing is
//!   refused. config.json keeps `"bits": 4, "storage": "ggml"` and records
//!   `"overrides": [{"pattern", "bits": 8, "count"}]`. E.g. Qwen-Image:
//!   `--tier q4=from-mlx --keep-q8 '^transformer_blocks\.\d+\.img_mod\.1$' --keep-src <bf16 root>
//!   --components transformer`.
//! * An existing output component is a hard error, so a partial run resumes by naming only the
//!   components still missing.
use std::path::{Path, PathBuf};
use std::process::Command;

use candle_gen::ggml_tier::{
    dump_headers, plan_jobs_keep, write_header_dump, KeepQ8Spec, PrefixMap, TierSpec,
};

fn usage(msg: &str) -> ! {
    eprintln!("error: {msg}\nsee the module docs of examples/candle_ggml_tier.rs for usage");
    std::process::exit(2)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let sub = if args.first().is_some_and(|a| !a.starts_with("--")) {
        args.remove(0)
    } else {
        "build".to_string()
    };
    match sub.as_str() {
        "build" => build(args),
        "dump-headers" => dump(args),
        "fetch-headers" => fetch(args),
        other => usage(&format!("unknown subcommand {other}")),
    }
}

fn dump(args: Vec<String>) {
    let (mut from, mut to) = (None, None);
    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        let v = it
            .next()
            .unwrap_or_else(|| usage(&format!("{flag} needs a value")));
        match flag.as_str() {
            "--from" => from = Some(PathBuf::from(v)),
            "--to" => to = Some(PathBuf::from(v)),
            other => usage(&format!("unknown argument {other}")),
        }
    }
    let (from, to) = (
        from.unwrap_or_else(|| usage("--from is required")),
        to.unwrap_or_else(|| usage("--to is required")),
    );
    let n = dump_headers(&from, &to).unwrap_or_else(|e| panic!("{e}"));
    println!("dumped {n} shard headers (+ sidecars) to {}", to.display());
}

fn curl(url: &str, range: Option<&str>, token: Option<&str>, out: Option<&Path>) -> Vec<u8> {
    let mut cmd = Command::new("curl");
    cmd.args(["-sS", "-L", "--fail", url]);
    if let Some(r) = range {
        cmd.args(["-H", &format!("Range: bytes={r}")]);
    }
    if let Some(t) = token {
        cmd.args(["-H", &format!("Authorization: Bearer {t}")]);
    }
    if let Some(o) = out {
        cmd.arg("-o").arg(o);
    }
    let output = cmd.output().unwrap_or_else(|e| panic!("curl: {e}"));
    assert!(
        output.status.success(),
        "curl {url}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn hf_token() -> String {
    std::env::var("HF_TOKEN").unwrap_or_else(|_| {
        let home = std::env::var("HF_HOME").map_or_else(
            |_| {
                PathBuf::from(
                    std::env::var("USERPROFILE")
                        .or_else(|_| std::env::var("HOME"))
                        .unwrap_or_default(),
                )
                .join(".cache")
                .join("huggingface")
            },
            PathBuf::from,
        );
        std::fs::read_to_string(home.join("token"))
            .unwrap_or_else(|e| panic!("no HF token: {e}"))
            .trim()
            .to_string()
    })
}

/// Download just the safetensors headers (HTTP Range) and the small non-weight files of one
/// huggingface.co component directory into a header-only template directory.
fn fetch(args: Vec<String>) {
    let (mut repo, mut rev, mut subdir, mut to) = (None, "main".to_string(), None, None);
    let mut use_token = false;
    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        if flag == "--use-hf-token" {
            use_token = true;
            continue;
        }
        let v = it
            .next()
            .unwrap_or_else(|| usage(&format!("{flag} needs a value")));
        match flag.as_str() {
            "--repo" => repo = Some(v),
            "--rev" => rev = v,
            "--subdir" => subdir = Some(v.trim_matches('/').to_string()),
            "--to" => to = Some(PathBuf::from(v)),
            other => usage(&format!("unknown argument {other}")),
        }
    }
    let repo = repo.unwrap_or_else(|| usage("--repo is required"));
    let subdir = subdir.unwrap_or_else(|| usage("--subdir is required"));
    let to = to.unwrap_or_else(|| usage("--to is required"));
    // Public repos need no token; the standard token location is read only on request.
    let token = use_token.then(hf_token);
    let token = token.as_deref();
    std::fs::create_dir_all(&to).unwrap_or_else(|e| panic!("{}: {e}", to.display()));
    let listing = curl(
        &format!("https://huggingface.co/api/models/{repo}/tree/{rev}/{subdir}"),
        None,
        token,
        None,
    );
    let listing: serde_json::Value = serde_json::from_slice(&listing).expect("tree listing JSON");
    let resolve = |path: &str| format!("https://huggingface.co/{repo}/resolve/{rev}/{path}");
    let mut shards = 0;
    for item in listing.as_array().expect("tree listing array") {
        if item["type"] != "file" {
            continue;
        }
        let path = item["path"].as_str().expect("path");
        let name = path.rsplit('/').next().expect("file name");
        if name.ends_with(".safetensors") {
            let len = curl(&resolve(path), Some("0-7"), token, None);
            let len = u64::from_le_bytes(len[..8].try_into().expect("8 length bytes"));
            let header = curl(&resolve(path), Some(&format!("8-{}", 7 + len)), token, None);
            write_header_dump(&to, name, &header).unwrap_or_else(|e| panic!("{name}: {e}"));
            shards += 1;
            println!("[header] {name}: {len} B");
        } else {
            curl(&resolve(path), None, token, Some(&to.join(name)));
            println!("[file]   {name}");
        }
    }
    println!(
        "fetched {shards} shard headers into {} (header-only template)",
        to.display()
    );
}

fn build(args: Vec<String>) {
    let mut src_root = None;
    let mut out_root = None;
    let mut tiers = Vec::new();
    let mut components: Vec<String> = Vec::new();
    let mut maps = Vec::new();
    let mut keep_q8: Vec<String> = Vec::new();
    let mut keep_src: Option<PathBuf> = None;
    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next()
                .unwrap_or_else(|| usage(&format!("{flag} needs a value")))
        };
        match flag.as_str() {
            "--src-root" => src_root = Some(PathBuf::from(val())),
            "--out-root" => out_root = Some(PathBuf::from(val())),
            "--tier" => {
                tiers.push(TierSpec::parse(&val()).unwrap_or_else(|e| usage(&e.to_string())))
            }
            "--component" | "--components" => components.extend(
                val()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty()),
            ),
            "--map-prefix" => {
                maps.push(PrefixMap::parse(&val()).unwrap_or_else(|e| usage(&e.to_string())))
            }
            "--keep-q8" => keep_q8.push(val()),
            "--keep-src" => keep_src = Some(PathBuf::from(val())),
            other => usage(&format!("unknown argument {other}")),
        }
    }
    let src_root = src_root.unwrap_or_else(|| usage("--src-root is required"));
    let out_root = out_root.unwrap_or_else(|| usage("--out-root is required"));
    if tiers.is_empty() {
        usage("at least one --tier is required (no --component = split tiers)");
    }
    let keep = match (keep_q8.is_empty(), keep_src) {
        (true, None) => None,
        (false, Some(root)) => {
            Some(KeepQ8Spec::new(&keep_q8, root).unwrap_or_else(|e| usage(&e.to_string())))
        }
        (true, Some(_)) => usage("--keep-src needs at least one --keep-q8 pattern"),
        (false, None) => usage("--keep-q8 needs --keep-src <bf16 root holding <component>>"),
    };
    let jobs = plan_jobs_keep(
        &src_root,
        &out_root,
        &tiers,
        &components,
        &maps,
        keep.as_ref(),
    )
    .unwrap_or_else(|e| usage(&e.to_string()));
    for job in jobs {
        let started = std::time::Instant::now();
        println!(
            "[start] {}: {} -> {}",
            job.label,
            job.src.display(),
            job.dst.display()
        );
        let report = job
            .run(&mut |path, r| {
                println!(
                    "  [shard] {} ({} converted, {} copied, {} B out, {:.1}s)",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    r.converted,
                    r.passthrough,
                    r.tensor_bytes,
                    started.elapsed().as_secs_f64()
                );
                Ok(())
            })
            .unwrap_or_else(|e| panic!("{}: {e}", job.label));
        println!(
            "[done] {}: {} shards written, {} files copied, {} quantized ({} from mxfp4), {} copied, {} B in -> {} B out, {:.1}s",
            job.label,
            report.shards,
            report.copied_files,
            report.converted,
            report.mxfp4_sources,
            report.passthrough,
            report.bytes_in,
            report.tensor_bytes,
            started.elapsed().as_secs_f64()
        );
        if report.precision_floors > 0 {
            println!(
                "  [floor] {}: {} projections packed wider than the tier in the MLX source, kept \
                 at their own width (Q8_0, as the load-time repack builds them) and recorded in \
                 `overrides`",
                job.label, report.precision_floors
            );
        }
        if job.options.keep_q8.is_some() {
            println!(
                "  [recipe] {}: {} Q4_1 (lossless from MLX q4) + {} Q8_0 --keep-q8 overrides (from {})",
                job.label,
                report.converted - report.keep_q8 - report.precision_floors,
                report.keep_q8,
                job.options
                    .keep_q8
                    .as_ref()
                    .map(|k| k.source.dir.display().to_string())
                    .unwrap_or_default()
            );
        }
        let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
        for (_, from, to) in &report.casts {
            *kinds.entry(format!("{from}->{to}")).or_default() += 1;
        }
        for (kind, n) in &kinds {
            println!("  [cast] {n} tensors {kind}");
        }
        for (name, from, to) in report.casts.iter().take(50) {
            println!("    {name}: {from} -> {to}");
        }
    }
}
