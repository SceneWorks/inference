//! sc-17157: **which of the two 66 GB DiT checkpoints a task loads, and that a render maps exactly
//! one of them.**
//!
//! # Nothing structural can tell `transformer/` from `transformer_ref/`
//!
//! Measured on the published snapshot `MiniMaxAI/MiniMax-H3 @ 939557dc`:
//!
//! * their `config.json` files are **byte-identical**;
//! * both ship **the same 638 tensor names**, with the same shapes and dtypes;
//! * only the tensor *values* differ.
//!
//! So a loader that reads the wrong one produces a model that loads cleanly, runs at the same
//! speed, allocates the same memory and emits plausible video. This is exactly the class
//! [`candle_gen_minimax_h3::layout`] documents for the gated-FFN half-swap, and the answer is the
//! same: **only an explicit assertion against real bytes can pin it.** A shape check, a key-count
//! check or a config diff would all pass on the wrong checkpoint.
//!
//! `proj_in.bias` is the probe. It is a 5376-wide float32 tensor living in shard 1 of both
//! partitions, it differs between them, and it is reachable through
//! `MiniMaxH3Dit::projections()` — so the assertion can be made against *the loaded model* rather
//! than against a file the loader might not have read.
//!
//! # What is gated WITHOUT weights, and why that matters
//!
//! The real-bytes assertion is necessarily `#[ignore]`d: it needs 66 GB. An `#[ignore]`d test that
//! is never run reports nothing, so the load-bearing guards here are the **three weights-free**
//! ones, which run on every CI lane:
//!
//! * `the_task_mapping_is_not_a_constant` — the selection function itself;
//! * `a_ref2va_request_without_the_reference_partition_is_refused` — the sc-19517 hosting gap fails
//!   LOUD at the engine boundary rather than degrading to the base checkpoint, **and** does not take
//!   `t2va` / `fl2va` down with it;
//! * `every_dit_load_site_in_the_crate_is_driven_by_the_task` — a **whole-crate** source scan
//!   proving there is nowhere in the render path for a hardcoded partition string to be;
//! * `model.rs`'s own `a_reference_request_resolves_to_the_reference_partition` — the other end of
//!   the same chain, driving a real `GenerationRequest` through the render path's own
//!   `MiniMaxH3::resolve_task`.
//!
//! Those last two are what covers "never both checkpoints resident" structurally, and neither is
//! the claim on its own:
//!
//! * the scan proves a DiT is mapped at exactly two sites **in the whole crate** and that both take
//!   their partition from `task.partition()` — so the two 66 GB partitions cannot both be mapped by
//!   one render, and a third load site in `pipeline.rs`, `dit/` or a new module reds it;
//! * the request test proves the `task` those sites read is the one a **reference request**
//!   resolves to, through the single `MiniMaxH3Task::resolve` call the scan pins as unique.
//!
//! `model.rs`'s `every_heavy_component_is_released_before_the_next_one_is_mapped` covers the
//! release ordering of whichever partition it picked.

mod common;

use std::collections::BTreeSet;
use std::path::Path;

use common::{safetensors_keys, snapshot};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{Conditioning, GenerationRequest, Image, LoadSpec, WeightsSource};
use candle_gen_minimax_h3::{
    MiniMaxH3, MiniMaxH3Dit, MiniMaxH3Task, BASE_DIT_PARTITION, REFERENCE_DIT_PARTITION,
};

/// The tensor that distinguishes the two partitions. float32, `[inner_dim]`, in shard 1 of each.
const PROBE: &str = "proj_in.bias";

/// The component directories a snapshot must carry for `MiniMaxH3::load` to succeed.
///
/// [`REFERENCE_DIT_PARTITION`] is **not** among them — it is required of the `ref2va` *request*,
/// not of the snapshot. See `a_ref2va_request_without_the_reference_partition_is_refused`.
const COMPONENTS: [&str; 4] = ["text_encoder", BASE_DIT_PARTITION, "vae", "audio_vae"];

/// One component directory, with the `.safetensors` shard a component is only satisfied by.
fn stage(root: &Path, component: &str) {
    let dir = root.join(component);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model-00001-of-00001.safetensors"), []).unwrap();
}

// ---------------------------------------------------------------------------------------------
// Weights-free
// ---------------------------------------------------------------------------------------------

/// **The task mapping is not a constant, and only `ref2va` moves.**
///
/// The negative half is the one that matters: a mapping that returned `transformer_ref` for
/// everything, or `transformer` for everything, satisfies "ref2va loads transformer_ref" or "t2va
/// loads transformer" respectively. Both are asserted, plus that the two names differ at all.
#[test]
fn the_task_mapping_is_not_a_constant() {
    assert_eq!(MiniMaxH3Task::Ref2va.partition(), REFERENCE_DIT_PARTITION);
    assert_eq!(MiniMaxH3Task::T2va.partition(), BASE_DIT_PARTITION);
    assert_eq!(MiniMaxH3Task::Fl2va.partition(), BASE_DIT_PARTITION);
    assert_ne!(
        BASE_DIT_PARTITION, REFERENCE_DIT_PARTITION,
        "the two partitions must be different directories, or the selection is vacuous"
    );
    assert_ne!(
        MiniMaxH3Task::Ref2va.partition(),
        MiniMaxH3Task::T2va.partition()
    );
}

/// **A `ref2va` REQUEST against a snapshot carrying only `transformer/` is refused, naming the
/// missing partition — and the other two tasks keep working.**
///
/// This is the sc-19517 hosting gap made loud: `SceneWorks/minimax-h3-mlx` publishes no
/// `q4/transformer_ref`, so a pure-`q4` install has exactly this shape. The failure mode being
/// prevented is a `ref2va` request silently rendering off `transformer/` — which produces plausible
/// video and is wrong.
///
/// **The blast radius is asserted, not assumed.** Requiring the partition in `MiniMaxH3::load` — the
/// obvious implementation — would fail provider construction outright on such a snapshot and take
/// `t2va` and `fl2va` down with it, a whole model offline for a capability neither uses. So the
/// requirement is task-gated: this asserts the provider still **constructs**, and `model.rs`'s
/// `a_ref2va_request_needs_the_reference_partition_but_the_other_tasks_do_not` asserts the other two
/// tasks still validate.
///
/// The control is the point: the SAME root **with** the reference partition admits the very same
/// request, so the refusal is attributable to that directory and not to a broken fixture.
#[test]
fn a_ref2va_request_without_the_reference_partition_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    for c in COMPONENTS {
        stage(&root, c);
    }
    let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
    let model = MiniMaxH3::load(&spec)
        .expect("a base-only snapshot must still construct: t2va and fl2va are unaffected");

    let request = GenerationRequest {
        prompt: "a cellist on a rooftop".into(),
        width: 1344,
        height: 768,
        conditioning: vec![Conditioning::Reference {
            image: Image {
                width: 64,
                height: 64,
                pixels: vec![0u8; 64 * 64 * 3],
            },
            strength: Some(1.0),
        }],
        ..Default::default()
    };
    let e = match model.resolve_task(&request) {
        Ok((task, _)) => panic!(
            "a {task:?} request without `{REFERENCE_DIT_PARTITION}/` must NOT be admitted — it \
             would render off the base checkpoint"
        ),
        Err(e) => e.to_string(),
    };
    assert!(
        e.contains(REFERENCE_DIT_PARTITION),
        "the refusal must name the missing partition: {e}"
    );

    // Control: the same root with the partition present admits the same request, on the reference
    // partition.
    stage(&root, REFERENCE_DIT_PARTITION);
    let (task, references) = MiniMaxH3::load(&spec)
        .unwrap()
        .resolve_task(&request)
        .expect("the control must be admitted, else the refusal proves nothing");
    assert_eq!(task, MiniMaxH3Task::Ref2va);
    assert_eq!(task.partition(), REFERENCE_DIT_PARTITION);
    assert!(references.is_some());
}

// ---------------------------------------------------------------------------------------------
// The whole-crate load-site scan
// ---------------------------------------------------------------------------------------------

/// One `src/*.rs` file's **production** text: its `#[cfg(test)]` module removed, then whole-line
/// comments, then block comments.
///
/// All three exclusions are load-bearing:
///
/// * **the test module**, because `model.rs`'s own tests legitimately name both partitions — that
///   is where the mapping is asserted — so scanning them would make the gate unpassable for the
///   right reasons;
/// * **line comments**, because the prose narrates the load sites directly, and counting it would
///   make the gate track the documentation rather than the code;
/// * **block comments**, because `/* MiniMaxH3Dit::load( … task.partition() … ) */` would otherwise
///   satisfy the scan without executing anything. (`src/` contains exactly one `/*`, inside a doc
///   comment, so this strip is safe here — and the assertion below that the anchors still appear
///   catches a strip that ate real code.)
fn production_text(whole: &str) -> String {
    let src = whole
        .split_once("\n#[cfg(test)]\n")
        .map_or(whole, |(before, _)| before);
    let lines: String = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut out = String::with_capacity(lines.len());
    let mut rest = lines.as_str();
    while let Some(open) = rest.find("/*") {
        out.push_str(&rest[..open]);
        match rest[open..].find("*/") {
            Some(close) => rest = &rest[open + close + 2..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Every `.rs` file at or below `src/`, as `(path, production text)`.
///
/// **The whole crate, not one file.** The scan this feeds used to `include_str!("../src/model.rs")`
/// — a single file — so a third `MiniMaxH3Dit::load(` in `pipeline.rs`, `dit/` or a new module was
/// invisible to it, and nothing else looked. sc-17150 × sc-17149 is exactly that defect: `generate`
/// had two DiT load sites, only one was converted, and the unmarked arm silently mixed tiers across
/// one render.
fn production_sources() -> Vec<(String, String)> {
    fn walk(dir: &Path, out: &mut Vec<(String, String)>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        for entry in &entries {
            let path = entry.path();
            let kind = entry
                .file_type()
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            // Symlinks are skipped: `is_dir` resolves them, so a link would be followed into
            // whatever tree it points at and a broken `*.rs` link would panic on read.
            if kind.is_symlink() {
                continue;
            }
            if kind.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let whole = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                out.push((
                    path.display().to_string(),
                    production_text(&whole),
                ));
            }
        }
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    walk(&dir, &mut out);
    assert!(
        out.len() > 10,
        "{} yielded only {} .rs file(s); the scan would pass vacuously",
        dir.display(),
        out.len()
    );
    out
}

/// The `(…)` argument span of every occurrence of `anchor` in `src`, brace-… paren-matched, with
/// **all whitespace removed**.
///
/// Whitespace removal is what makes the check survive rustfmt. The previous version matched one
/// exact literal on one trimmed line, so a single extra indent level — enough to push
/// `MiniMaxH3Dit::load(&self.root, task.partition(), &self.device, self.dtype)` past 100 columns —
/// made rustfmt wrap the call and the guard went red on a purely cosmetic change. Matching the
/// whitespace-free span instead accepts any formatting of the same call.
fn call_spans(src: &str, anchor: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(at) = src[from..].find(anchor) {
        let open = from + at + anchor.len() - 1;
        let mut depth = 0usize;
        let mut end = None;
        for (i, c) in src[open..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(open + i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end.unwrap_or_else(|| panic!("unbalanced parens after `{anchor}`"));
        out.push(src[open..end].chars().filter(|c| !c.is_whitespace()).collect());
        from = end;
    }
    out
}

/// **The whole crate maps a DiT at exactly two sites, and BOTH take their partition from the
/// task.**
///
/// The structural half of the never-both-resident property, and the half a memory measurement
/// cannot reach: measuring one load says nothing about a second load elsewhere. This reads **every**
/// `.rs` file under `src/` and pins that there is nowhere else for a load site to be.
///
/// Source-scanning is a blunt instrument, so it is bounded and stated:
///
/// * it asserts a whole-crate **count**, and where those sites are;
/// * it asserts each site's argument span contains `task.partition()`, matched with whitespace
///   removed so **rustfmt cannot red it** and after block comments are stripped so a commented-out
///   call cannot satisfy it;
/// * it asserts `MiniMaxH3Dit` is never **aliased** on import, because `use … as D; D::load(…)`
///   would otherwise be a load site the anchor cannot see;
/// * it asserts `MiniMaxH3Task::resolve` is called from exactly **one** place in the crate, so the
///   `task` those sites read cannot be a second, divergent derivation;
/// * and it asserts the bare partition literals never appear outside their `const` declarations.
#[test]
fn every_dit_load_site_in_the_crate_is_driven_by_the_task() {
    const ANCHOR: &str = "MiniMaxH3Dit::load(";
    let sources = production_sources();

    let mut sites: Vec<(String, String)> = Vec::new();
    let mut resolve_calls = 0usize;
    for (path, src) in &sources {
        // `use candle_gen_minimax_h3::dit::model::MiniMaxH3Dit as D;` then `D::load(` is a load
        // site with no anchor in it. Ban the alias rather than pretend the anchor is exhaustive.
        assert!(
            !src.contains("MiniMaxH3Dit as "),
            "{path} aliases MiniMaxH3Dit on import; the load-site anchor cannot see `D::load(`"
        );
        for span in call_spans(src, ANCHOR) {
            sites.push((path.clone(), span));
        }
        resolve_calls += src.matches("MiniMaxH3Task::resolve(").count();
    }

    assert_eq!(
        sites.len(),
        2,
        "the crate has {} `{ANCHOR}` call site(s); the render path has exactly two (the t2va/fl2va \
         arm and the ref2va arm), which are mutually exclusive branches: {sites:?}",
        sites.len()
    );
    for (path, span) in &sites {
        assert!(
            path.ends_with("model.rs"),
            "a DiT load site appeared in {path}; the render path's two sites are both in model.rs, \
             and a third one elsewhere is how a ref2va render reads the wrong 66 GB"
        );
        assert!(
            span.contains("task.partition()"),
            "{path}: a DiT load site does not take its partition from `task.partition()` — a \
             hardcoded partition is how a ref2va render reads the wrong 66 GB: {span}"
        );
    }

    // One derivation of the task, crate-wide. Two would let the load sites read a `task` that a
    // request never produced.
    assert_eq!(
        resolve_calls, 1,
        "`MiniMaxH3Task::resolve(` is called {resolve_calls} time(s) in the crate; it must be \
         exactly once (inside `MiniMaxH3::resolve_task`), or the load sites' `task` can diverge \
         from the one the request resolved to"
    );

    // ...and neither partition literal appears anywhere outside its own `const` declaration.
    for literal in ["\"transformer\"", "\"transformer_ref\""] {
        let uses: Vec<String> = sources
            .iter()
            .flat_map(|(path, src)| {
                src.lines()
                    // A `const` DECLARATION of the literal is the one place it may appear. The
                    // filter read `!l.contains("pub const")`, which missed `memory_strategy.rs`'s
                    // private `const DIT_COMPONENT: &str = "transformer";` — invisible while this
                    // scan read only `model.rs`, and a guaranteed red the moment it was broadened
                    // to the whole crate. Visibility is not what makes a line a declaration.
                    .filter(move |l| {
                        let decl = l.trim_start();
                        l.contains(literal)
                            && !decl.starts_with("const ")
                            && !decl.starts_with("pub const ")
                            && !decl.starts_with("pub(crate) const ")
                    })
                    .map(move |l| format!("{path}: {}", l.trim()))
            })
            .collect();
        assert!(
            uses.is_empty(),
            "the bare literal {literal} appears outside its const declaration: {uses:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Real weights
// ---------------------------------------------------------------------------------------------

/// The `[out]` float32 values of `PROBE` in one partition, read straight off the shards.
fn probe_bytes(root: &Path, partition: &str) -> Vec<f32> {
    let dir = root.join(partition);
    let shards = candle_gen::loader::sorted_safetensors(&dir, "minimax-h3 dit").unwrap();
    let w = candle_gen::Weights::from_files_filtered(&shards, &Device::Cpu, DType::F32, &[PROBE])
        .unwrap();
    w.require(PROBE)
        .unwrap_or_else(|e| panic!("{partition}/{PROBE}: {e}"))
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// **The two partitions really are byte-different and structurally identical.**
///
/// This is the premise every other real-weight assertion rests on, so it is measured rather than
/// assumed. If a future snapshot made them identical, the wrong-checkpoint test below would become
/// vacuous — and this fails first, saying so.
#[test]
#[ignore = "reads shard headers plus one tensor from each 66 GB DiT partition (MINIMAX_H3_SNAPSHOT)"]
fn the_two_partitions_are_indistinguishable_except_by_value() {
    let root = snapshot();
    let base = root.join(BASE_DIT_PARTITION);
    let reference = root.join(REFERENCE_DIT_PARTITION);

    let base_cfg = std::fs::read(base.join("config.json")).unwrap();
    let ref_cfg = std::fs::read(reference.join("config.json")).unwrap();
    assert_eq!(
        base_cfg, ref_cfg,
        "the two partitions' config.json are supposed to be byte-identical; if they diverged, the \
         selection could be made structurally and this whole file is over-engineered"
    );

    let base_keys: BTreeSet<String> = candle_gen::loader::sorted_safetensors(&base, "dit")
        .unwrap()
        .iter()
        .flat_map(|p| safetensors_keys(p))
        .collect();
    let ref_keys: BTreeSet<String> = candle_gen::loader::sorted_safetensors(&reference, "dit")
        .unwrap()
        .iter()
        .flat_map(|p| safetensors_keys(p))
        .collect();
    assert_eq!(
        base_keys, ref_keys,
        "the two partitions ship the same tensor names"
    );
    println!("  both partitions carry {} tensors", base_keys.len());

    // …and only the VALUES differ.
    let a = probe_bytes(&root, BASE_DIT_PARTITION);
    let b = probe_bytes(&root, REFERENCE_DIT_PARTITION);
    assert_eq!(a.len(), b.len(), "{PROBE} has the same shape in both");
    assert_ne!(
        a, b,
        "{PROBE} is identical in both partitions, so it cannot discriminate them; pick another \
         probe before trusting the wrong-checkpoint test"
    );
}

/// **The wrong-checkpoint test: the DiT loader handed `Ref2va`'s partition really reads
/// `transformer_ref`, and this fails if it reads `transformer`.**
///
/// Named for what it proves. It does **not** construct a request and does not call `MiniMaxH3::load`
/// or `generate_impl` — it computes `MiniMaxH3Task::Ref2va.partition()` and calls
/// `MiniMaxH3Dit::load` directly, so what it pins is that *the loader honours the partition it is
/// handed*. The other two links in the chain are weights-free and run on every lane:
/// `a_reference_request_resolves_to_the_reference_partition` (in `model.rs`) proves a reference
/// **request** resolves to that task, and `every_dit_load_site_in_the_crate_is_driven_by_the_task`
/// proves both load sites take their partition from it and from nothing else.
///
/// Asserted against **the loaded model's own weights**, not against the path string — a test that
/// only compared partition names would pass on a loader that built the path correctly and then read
/// somewhere else.
///
/// The `assert_ne!` against the base checkpoint is the half that matters. Without it, a loader that
/// returned the *same* tensor for every partition would satisfy the positive assertion.
#[test]
#[ignore = "loads the 66 GB transformer_ref (MINIMAX_H3_SNAPSHOT)"]
fn the_dit_loader_honours_the_ref2va_partition_and_reads_transformer_ref() {
    let root = snapshot();
    let expected = probe_bytes(&root, REFERENCE_DIT_PARTITION);
    let base = probe_bytes(&root, BASE_DIT_PARTITION);
    assert_ne!(expected, base, "premise: the probe discriminates");

    // The task -> partition mapping is what the render path uses; it is NOT hardcoded here.
    let partition = MiniMaxH3Task::Ref2va.partition();
    assert_eq!(partition, REFERENCE_DIT_PARTITION);

    let dit = MiniMaxH3Dit::load(&root, partition, &Device::Cpu, DType::BF16).unwrap();
    let loaded = dit
        .projections()
        .proj_in
        .bias()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(
        loaded, expected,
        "a ref2va load must carry transformer_ref's weights"
    );
    assert_ne!(
        loaded, base,
        "a ref2va load must NOT carry transformer/'s weights — this is the assertion that fails \
         if the wrong 66 GB checkpoint was read"
    );
}
