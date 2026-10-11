//! sc-25689 — **real multi-rank** training on one host, the Candle twin of the MLX
//! `train_distributed`: every scenario launches each rank as its own OS process (this test binary
//! re-entered through [`rank_entry`]) running the Candle trainer on the miniature snapshot on the
//! build's device (CPU here; on a CUDA build rank r on `cuda:r`, ranks sharing a card where the box
//! has fewer), over the host TCP transport (the Candle trainer's transport: Windows has no NCCL).
//!
//! * replication — DDP ranks reproduce bit for bit across launches, the MLX transports are typed
//!   refusals here, and FSDP — whose ranks cannot diverge by construction (every step re-gathers
//!   one set of rows) — reproduces the DDP replicas;
//! * sharding — FSDP `full`, `grad_op` (2 ranks) and `hybrid` (2 × 2 = 4 ranks) reproduce the DDP
//!   run of the same world (global batch = Σ local batches; optimizer + EMA semantics preserved,
//!   Muon orthogonalizing the gathered matrix);
//! * sharded checkpoints — a `dcp` checkpoint resumes exactly on the same mesh, and reshards to a
//!   single process;
//! * failure — a rank that exits, or stalls, mid-run surfaces as a typed `RankLost` on the
//!   survivor within the transport timeout (no hang); invalid meshes and a global-batch mismatch
//!   are typed refusals before any rank connects.

use std::collections::HashMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use candle_gen::candle_core::DType;
use candle_gen::gen_core::iris::train::dist::{dist_error, DistError};
use candle_gen::gen_core::iris::train::{
    checkpoint_root, latest_checkpoint, read_checkpoint_state, OPTIONS_KEY,
};
use candle_gen::gen_core::iris::TEXT_ENCODER_COMPONENT;
use candle_gen::gen_core::{
    LoadSpec, LrSchedule, Trainer, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use serde_json::{json, Value as Json};

use crate::common::{cpu, tiny_backbone, tiny_text_encoder};

fn on_cpu<T>(f: impl FnOnce() -> T) -> T {
    f()
}

const ID: &str = "iris_3b";
const CHILD_ENV: &str = "IRIS_DIST_CHILD";
const NAME: &str = "dist.safetensors";
/// The bound on any one rank process (a hang fails the test instead of stalling CI).
const RANK_DEADLINE: Duration = Duration::from_secs(300);

fn trainer() -> Box<dyn Trainer> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    candle_gen_iris::provider_registry()
        .unwrap()
        .load_trainer(ID, &spec)
        .expect("the tiny snapshot loads as a trainer")
}

const CAPTIONS: [&str; 4] = [
    "a red fox in the snow",
    "golden hour",
    "a fox , snow",
    "snow , golden fox",
];

fn write_dataset(dir: &Path) {
    for i in 0..CAPTIONS.len() {
        let img = image::RgbImage::from_fn(18 + i as u32 * 2, 16, |x, y| {
            image::Rgb([
                (x * 9 + i as u32 * 40) as u8,
                (y * 7 + 30) as u8,
                ((x + y) * 5 + i as u32 * 70) as u8,
            ])
        });
        img.save(dir.join(format!("img{i}.png"))).unwrap();
    }
}

fn items(dir: &Path) -> Vec<TrainingItem> {
    CAPTIONS
        .iter()
        .enumerate()
        .map(|(i, c)| TrainingItem::captioned(dir.join(format!("img{i}.png")), c.to_string()))
        .collect()
}

/// One rank's job, handed to the child process as JSON.
fn job(
    data: &Path,
    out: &Path,
    rank: usize,
    world: Json,
    dist: Json,
    steps: u32,
    cancel_at: Option<u32>,
) -> Json {
    let mut d = dist;
    let mut w = world;
    w["rank"] = json!(rank);
    d["world"] = w;
    json!({
        "data": data, "out": out, "steps": steps, "dist": d,
        "cancel_at": cancel_at, "die_at": null, "stall_at": null, "resume": false,
    })
}

fn config(steps: u32) -> TrainingConfig {
    TrainingConfig {
        learning_rate: 1e-3,
        steps,
        batch_size: 1,
        resolution: 16,
        save_every: 2,
        seed: 9,
        train_dtype: "f32".into(),
        lr_scheduler: LrSchedule::Cosine,
        lr_warmup_steps: 1,
        full_finetune: true,
        optimizer: "muon".into(),
        weight_decay: 0.01,
        ..Default::default()
    }
}

/// The child side: run one rank's job and write `{ok}` or `{err, dist}` to `result`.
fn run_job(j: &Json) -> Json {
    let data = PathBuf::from(j["data"].as_str().unwrap());
    let out = PathBuf::from(j["out"].as_str().unwrap());
    let mut cfg = config(j["steps"].as_u64().unwrap() as u32);
    cfg.resume = j["resume"].as_bool().unwrap_or(false);
    let mut opts = json!({"ema_decay": 0.9, "text_dropout": 0.3});
    match j["optimizer"].as_str() {
        // Muon with the clip out of reach: the orthogonalization input is then independent of
        // how the global norm was summed.
        Some(UNCLIPPED_MUON) => {
            cfg.optimizer = "muon".into();
            opts["gradient_clip"] = json!(1e6);
        }
        Some(o) => cfg.optimizer = o.into(),
        None => {}
    }
    let dist = j["dist"].clone();
    let world_size = dist["world"]["world_size"].as_u64().unwrap_or(1);
    if world_size > 1 {
        opts["dist"] = dist.clone();
        if dist["world"]["num_nodes"].as_u64().unwrap_or(1) > 1 {
            opts["perf"] = json!({"lazy_sync": true});
        }
    }
    cfg.model_options.insert(OPTIONS_KEY.into(), opts);
    let req = TrainingRequest {
        items: items(&data),
        config: cfg,
        output_dir: out,
        file_name: NAME.into(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    };
    let flag = req.cancel.clone();
    let cancel_at = j["cancel_at"].as_u64().map(|v| v as u32);
    let die_at = j["die_at"].as_u64().map(|v| v as u32);
    let stall_at = j["stall_at"].as_u64().map(|v| v as u32);
    let started = Instant::now();
    let res = on_cpu(|| {
        trainer().train(&req, &mut |p| {
            if let TrainingProgress::Training { step, .. } = p {
                if Some(step) == cancel_at {
                    flag.cancel();
                }
                if Some(step) == die_at {
                    std::process::exit(17);
                }
                if Some(step) == stall_at {
                    std::thread::sleep(Duration::from_secs(120));
                }
            }
        })
    });
    let secs = started.elapsed().as_secs_f64();
    match res {
        Ok(o) => json!({"ok": o.adapter_path, "steps": o.steps, "secs": secs}),
        Err(e) => json!({
            "err": e.to_string(),
            "dist": dist_error(&e).map(|d| format!("{d:?}")),
            "secs": secs,
        }),
    }
}

/// The re-entry point of a rank process (a no-op when run as an ordinary test).
#[test]
#[ignore = "the rank-process entry of train_distributed (driven by the scenarios)"]
fn rank_entry() {
    let Ok(raw) = std::env::var(CHILD_ENV) else {
        return;
    };
    let j: Json = serde_json::from_str(&raw).unwrap();
    let result = run_job(&j);
    std::fs::write(
        j["result"].as_str().unwrap(),
        serde_json::to_vec(&result).unwrap(),
    )
    .unwrap();
}

fn spawn(mut j: Json, result: &Path) -> Child {
    j["result"] = json!(result);
    Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "train_distributed::rank_entry",
            "--ignored",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .env(CHILD_ENV, j.to_string())
        .spawn()
        .expect("spawn a rank process")
}

/// Reap `child`, killing it (only this PID) if it is still running at `deadline` — a hung rank
/// then writes no result, and reading that result fails the test.
fn reap_or_kill(child: &mut Child, deadline: Instant) {
    loop {
        if child.try_wait().unwrap().is_some() {
            return;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn read_result(path: &Path, rank: usize) -> Json {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!("rank {rank} wrote no result (it crashed, or hung and was killed): {e}")
    });
    serde_json::from_slice(&bytes).unwrap()
}

/// Launch every job as a rank process and collect their results.
fn launch(jobs: Vec<Json>, scratch: &Path) -> Vec<Json> {
    let deadline = Instant::now() + RANK_DEADLINE;
    let results: Vec<PathBuf> = (0..jobs.len())
        .map(|r| scratch.join(format!("result{r}.{}.json", unique())))
        .collect();
    let mut children: Vec<Child> = jobs
        .into_iter()
        .zip(&results)
        .map(|(j, r)| spawn(j, r))
        .collect();
    for c in children.iter_mut() {
        reap_or_kill(c, deadline);
    }
    results
        .iter()
        .enumerate()
        .map(|(r, p)| read_result(p, r))
        .collect()
}

fn unique() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn tcp_world(n: usize, nodes: usize) -> Json {
    json!({
        "world_size": n, "local_world_size": n / nodes, "num_nodes": nodes,
        "master_addr": "127.0.0.1", "master_port": free_port(), "transport": "tcp",
        "timeout_secs": 20,
    })
}

fn ok(r: &Json) -> PathBuf {
    PathBuf::from(
        r["ok"]
            .as_str()
            .unwrap_or_else(|| panic!("rank failed: {r}")),
    )
}

fn tensors(path: &Path) -> HashMap<String, Vec<f32>> {
    candle_gen::candle_core::safetensors::load(path, &cpu())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .into_iter()
        .map(|(k, v)| {
            let v = v
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            (k, v)
        })
        .collect()
}

/// The largest |a − b| over two tensor files with the same keys.
fn max_diff(a: &Path, b: &Path) -> f32 {
    let (ta, tb) = (tensors(a), tensors(b));
    assert_eq!(
        ta.keys().collect::<std::collections::BTreeSet<_>>(),
        tb.keys().collect::<std::collections::BTreeSet<_>>(),
        "{} vs {}",
        a.display(),
        b.display()
    );
    let mut worst = 0f32;
    for (k, va) in &ta {
        let vb = &tb[k];
        assert_eq!(va.len(), vb.len(), "{k}");
        for (x, y) in va.iter().zip(vb) {
            let d = (x - y).abs();
            assert!(d.is_finite(), "{k}: non-finite");
            worst = worst.max(d);
        }
    }
    worst
}

fn latest(out: &Path) -> PathBuf {
    latest_checkpoint(&checkpoint_root(out, NAME)).expect("a published checkpoint")
}

fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

/// Run `dist` over `n` ranks on `world` with `optimizer`, sharing one output dir; returns it.
fn run_on(
    world: Json,
    n: usize,
    dist: Json,
    optimizer: &str,
    data: &Path,
    scratch: &Path,
) -> tempfile::TempDir {
    let out = tmp();
    let jobs = (0..n)
        .map(|r| {
            let mut j = job(data, out.path(), r, world.clone(), dist.clone(), 4, None);
            j["optimizer"] = json!(optimizer);
            j
        })
        .collect();
    let res = launch(jobs, scratch);
    assert!(
        res.iter().all(|r| r["steps"] == 4),
        "every rank completes: {res:#?}"
    );
    for r in &res {
        assert_eq!(ok(r), ok(&res[0]), "every rank returns the same artifact");
    }
    out
}

fn shared_run(
    n: usize,
    dist: Json,
    optimizer: &str,
    data: &Path,
    scratch: &Path,
) -> tempfile::TempDir {
    run_on(tcp_world(n, 1), n, dist, optimizer, data, scratch)
}

/// Two launches of the same 2-rank DDP job reproduce each other bit for bit (rank-order
/// reductions; positional randomness keyed on the rank).
#[test]
fn ddp_runs_reproduce_bit_for_bit() {
    let (data, scratch) = (tmp(), tmp());
    write_dataset(data.path());
    let ddp = json!({"strategy": "ddp"});
    let a = shared_run(2, ddp.clone(), "muon", data.path(), scratch.path());
    let b = shared_run(2, ddp, "muon", data.path(), scratch.path());
    let (la, lb) = (latest(a.path()), latest(b.path()));
    for f in [
        "trainable.safetensors",
        "ema.safetensors",
        "optimizer.safetensors",
    ] {
        assert_eq!(max_diff(&la.join(f), &lb.join(f)), 0.0, "{f}");
    }
    assert_eq!(max_diff(&export(a.path()), &export(b.path())), 0.0);
    assert_eq!(read_checkpoint_state(&la).unwrap().step, 4);
}

/// The MLX transports are MLX's distributed layer: refused typed on the Candle trainer before any
/// rank connects.
#[test]
fn the_mlx_transports_are_typed_refusals_on_candle() {
    let data = tmp();
    write_dataset(data.path());
    let mut world = tcp_world(2, 1);
    world["transport"] = json!("mlx_ring");
    let mut cfg = config(4);
    cfg.model_options.insert(
        OPTIONS_KEY.into(),
        json!({"dist": {"strategy": "ddp", "world": world}}),
    );
    let req = TrainingRequest {
        items: items(data.path()),
        config: cfg,
        output_dir: data.path().join("out"),
        file_name: NAME.into(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    };
    let e = trainer()
        .train(&req, &mut |_| {})
        .expect_err("mlx_ring is refused");
    assert!(
        matches!(dist_error(&e), Some(DistError::Unsupported(_))),
        "{e}"
    );
}

fn export(out: &Path) -> PathBuf {
    out.join("dist").join("model.safetensors")
}

/// A sharded step equals the replicated one up to the f32 rounding of the global gradient norm
/// (summed per shard, then clipped): AdamW is elementwise, ≤ 1e-6. Muon orthogonalizes in bf16
/// (Dion's Newton–Schulz), which turns that last-ulp clip difference into bf16-sized update
/// differences — so Muon is compared with the clip out of reach ([`UNCLIPPED_MUON`]), where the
/// gathered matrix it orthogonalizes is the replicated one exactly.
const ADAMW_TOL: f32 = 1e-6;
const UNCLIPPED_MUON: &str = "muon_unclipped";

fn assert_matches_ddp(ddp: &Path, other: &Path, what: &str, tol: f32) {
    let d = max_diff(&export(ddp), &export(other));
    eprintln!("{what} vs ddp export: max |Δ| = {d:.3e} (tol {tol:.1e})");
    assert!(d <= tol, "{what}: {d}");
    let (a, b) = (latest(ddp), latest(other));
    for f in ["ema.safetensors", "optimizer.safetensors"] {
        if !a.join(f).is_file() || !b.join(f).is_file() {
            continue;
        }
        let d = max_diff(&a.join(f), &b.join(f));
        eprintln!("{what} vs ddp {f}: max |Δ| = {d:.3e}");
        assert!(d <= tol.max(1e-5), "{what} {f}: {d}");
    }
}

#[test]
fn fsdp_full_and_grad_op_sharding_reproduce_ddp() {
    let (data, scratch) = (tmp(), tmp());
    write_dataset(data.path());
    for (optimizer, tol) in [("adamw", ADAMW_TOL), (UNCLIPPED_MUON, ADAMW_TOL)] {
        let ddp = shared_run(
            2,
            json!({"strategy": "ddp"}),
            optimizer,
            data.path(),
            scratch.path(),
        );
        for sharding in ["full", "grad_op"] {
            let fsdp = shared_run(
                2,
                json!({"strategy": "fsdp2", "sharding": sharding, "param_dtype": "fp32",
                       "checkpoint_io": "rank0"}),
                optimizer,
                data.path(),
                scratch.path(),
            );
            assert_matches_ddp(
                ddp.path(),
                fsdp.path(),
                &format!("fsdp {sharding} {optimizer}"),
                tol,
            );
            let s = read_checkpoint_state(&latest(fsdp.path())).unwrap();
            assert_eq!(s.step, 4);
            assert_eq!(s.dist.as_ref().unwrap()["dp_shard"], 2, "{:?}", s.dist);
        }
        if optimizer == "adamw" {
            // The default param_dtype (bf16) is honored: the gathered forward copy is rounded.
            let bf16 = shared_run(
                2,
                json!({"strategy": "fsdp2", "sharding": "full"}),
                optimizer,
                data.path(),
                scratch.path(),
            );
            let d = max_diff(&export(ddp.path()), &export(bf16.path()));
            eprintln!("fsdp full param_dtype=bf16 vs ddp: max |Δ| = {d:.3e}");
            assert!(
                d > 0.0 && d < 1e-2,
                "param_dtype=bf16 rounds the forward copy: {d}"
            );
        }
    }
}

#[test]
fn hybrid_sharding_over_four_ranks_reproduces_ddp() {
    let (data, scratch) = (tmp(), tmp());
    write_dataset(data.path());
    let ddp = shared_run(
        4,
        json!({"strategy": "ddp"}),
        "adamw",
        data.path(),
        scratch.path(),
    );
    let hybrid = shared_run(
        4,
        json!({"strategy": "fsdp2", "sharding": "hybrid", "param_dtype": "fp32",
               "mesh": {"dp_replicate": 2, "dp_shard": 2}, "checkpoint_io": "dcp"}),
        "adamw",
        data.path(),
        scratch.path(),
    );
    let d = max_diff(&export(ddp.path()), &export(hybrid.path()));
    eprintln!("hybrid 2x2 vs ddp export: max |Δ| = {d:.3e}");
    assert!(d <= ADAMW_TOL, "hybrid: {d}");
}

#[test]
fn a_sharded_checkpoint_resumes_exactly_and_reshards() {
    let (data, scratch) = (tmp(), tmp());
    write_dataset(data.path());
    let dist = json!({"strategy": "fsdp2", "sharding": "full", "checkpoint_io": "dcp"});
    let whole = shared_run(2, dist.clone(), "muon", data.path(), scratch.path());

    // Interrupted after step 2 (a published dcp checkpoint), then resumed to 4 on the same mesh.
    let cut = tmp();
    let world = tcp_world(2, 1);
    let jobs = (0..2)
        .map(|r| {
            job(
                data.path(),
                cut.path(),
                r,
                world.clone(),
                dist.clone(),
                4,
                Some(2),
            )
        })
        .collect();
    for r in launch(jobs, scratch.path()) {
        assert_eq!(r["steps"], 2, "{r}");
    }
    let at2 = latest(cut.path());
    assert!(
        at2.join("shard_00000.safetensors").is_file()
            && at2.join("shard_00001.safetensors").is_file(),
        "dcp writes one file per shard"
    );
    // Reshard: the 2-shard checkpoint resumes in one process (copied so the mesh run stays clean).
    let single = tmp();
    copy_dir(
        &checkpoint_root(cut.path(), NAME),
        &checkpoint_root(single.path(), NAME),
    );
    let world = tcp_world(2, 1);
    let jobs = (0..2)
        .map(|r| {
            let mut j = job(
                data.path(),
                cut.path(),
                r,
                world.clone(),
                dist.clone(),
                4,
                None,
            );
            j["resume"] = json!(true);
            j
        })
        .collect();
    for r in launch(jobs, scratch.path()) {
        assert_eq!(r["steps"], 4, "{r}");
    }
    for f in ["shard_00000.safetensors", "shard_00001.safetensors"] {
        assert_eq!(
            max_diff(&latest(whole.path()).join(f), &latest(cut.path()).join(f)),
            0.0,
            "{f}: the resumed mesh run equals the uninterrupted one"
        );
    }
    assert_eq!(
        max_diff(&export(whole.path()), &export(cut.path())),
        0.0,
        "exports agree"
    );

    let mut j = job(
        data.path(),
        single.path(),
        0,
        json!({"world_size": 1}),
        json!({}),
        4,
        None,
    );
    j["resume"] = json!(true);
    let r = launch(vec![j], scratch.path());
    assert_eq!(r[0]["steps"], 4, "{}", r[0]);
    let s = read_checkpoint_state(&latest(single.path())).unwrap();
    assert_eq!(s.step, 4, "the resharded run continued from step 2");
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dst = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &dst);
        } else {
            std::fs::copy(e.path(), dst).unwrap();
        }
    }
}

fn lost_rank_case(field: &str) {
    let (data, scratch, out) = (tmp(), tmp(), tmp());
    write_dataset(data.path());
    let mut world = tcp_world(2, 1);
    world["timeout_secs"] = json!(5);
    let mut jobs: Vec<Json> = (0..2)
        .map(|r| {
            job(
                data.path(),
                out.path(),
                r,
                world.clone(),
                json!({"strategy": "ddp"}),
                4,
                None,
            )
        })
        .collect();
    jobs[1][field] = json!(1);
    let deadline = Instant::now() + Duration::from_secs(120);
    let results: Vec<PathBuf> = (0..2)
        .map(|r| scratch.path().join(format!("lost{r}.json")))
        .collect();
    let mut kids: Vec<Child> = jobs
        .into_iter()
        .zip(&results)
        .map(|(j, r)| spawn(j, r))
        .collect();
    // A survivor that hung instead of failing is killed at the deadline and writes no result.
    reap_or_kill(&mut kids[0], deadline);
    // The stalled rank is ours to stop; the exited one is already gone.
    let _ = kids[1].kill();
    let _ = kids[1].wait();
    let r = read_result(&results[0], 0);
    eprintln!("survivor ({field}): {r}");
    let typed = r["dist"]
        .as_str()
        .unwrap_or_else(|| panic!("untyped failure: {r}"));
    assert!(typed.starts_with("RankLost"), "{typed}");
    if field == "stall_at" {
        // The bound that ended the wait is the configured transport timeout (clock-free reading:
        // the typed detail names it).
        assert!(typed.contains("within the 5s transport timeout"), "{typed}");
    }
}

#[test]
fn a_rank_that_exits_mid_run_fails_the_survivor_typed() {
    lost_rank_case("die_at");
}

#[test]
fn a_rank_that_stalls_mid_run_fails_the_survivor_typed_within_the_timeout() {
    lost_rank_case("stall_at");
}

#[test]
fn invalid_meshes_and_global_batch_mismatches_are_typed_refusals() {
    let data = tmp();
    write_dataset(data.path());
    let refusal = |dist: Json, extra: Json| -> DistError {
        let mut cfg = config(4);
        let mut opts = json!({"dist": dist});
        for (k, v) in extra.as_object().unwrap() {
            opts[k] = v.clone();
        }
        cfg.model_options.insert(OPTIONS_KEY.into(), opts);
        let req = TrainingRequest {
            items: items(data.path()),
            config: cfg,
            output_dir: data.path().join("out"),
            file_name: NAME.into(),
            trigger_words: Vec::new(),
            cancel: Default::default(),
        };
        let e = trainer().validate(&req).expect_err("refused");
        dist_error(&e)
            .unwrap_or_else(|| panic!("untyped refusal: {e}"))
            .clone()
    };
    let w = |n: usize| json!({"world_size": n, "rank": 0, "master_port": 1});
    assert!(matches!(
        refusal(
            json!({"strategy": "fsdp2", "mesh": {"dp_shard": 3}, "world": w(2)}),
            json!({})
        ),
        DistError::InvalidMesh(_)
    ));
    assert!(matches!(
        refusal(
            json!({"strategy": "fsdp2", "sharding": "hybrid",
                   "mesh": {"dp_replicate": 3, "dp_shard": 2}, "world": w(4)}),
            json!({})
        ),
        DistError::InvalidMesh(_)
    ));
    assert!(matches!(
        refusal(
            json!({"strategy": "ddp", "mesh": {"dp_shard": 2}, "world": w(2)}),
            json!({})
        ),
        DistError::InvalidMesh(_)
    ));
    assert!(matches!(
        refusal(
            json!({"strategy": "ddp", "world": w(2)}),
            json!({"expected_global_batch": 3})
        ),
        DistError::GlobalBatchMismatch { .. }
    ));
    assert!(matches!(
        refusal(
            json!({"strategy": "ddp", "world": {"world_size": 2, "rank": 2}}),
            json!({})
        ),
        DistError::InvalidMesh(_) | DistError::SplitWorld(_)
    ));
}
