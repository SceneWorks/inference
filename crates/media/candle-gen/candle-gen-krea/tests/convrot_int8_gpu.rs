//! **sc-12301 — the per-projection cuBLASLt workspace in the INT8-ConvRot lane, measured.**
//!
//! `QLinear::convrot_int8` built a fresh `CublasLt` handle for **every** int8 projection, and
//! `CublasLt::new` eagerly allocates a 32 MiB workspace it holds for life. `linear_detect` reaches that
//! constructor once per projection, so a ConvRot DiT's ~224 int8 projections each carried their own —
//! duplicated scratch that no weights-only footprint sum can see. This is the int8 half of the defect
//! sc-12274 fixed (and measured at 32.00 MiB/handle) on the NVFP4 lane.
//!
//! **Weights-free**, so it runs in the normal CUDA lane rather than needing the community ConvRot
//! checkpoint: it grows the projection count on a fixed, tiny synthetic weight and reads the driver's
//! free-memory response. That measures the *mechanism* and its real per-projection cost. What it
//! deliberately does **not** claim is a reading on a real ConvRot trunk — see
//! [`convrot_int8_shares_one_cublaslt_workspace_across_projections`]'s extrapolation note.
//!
//! CUDA-only (needs a live cuBLASLt handle); the sm_89 int8 floor (locked decision 7) is checked and the
//! gate no-ops below it, exactly as the lane itself would refuse.
#![cfg(feature = "cuda")]

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::quant::{CublasLt, Int8Context};
use candle_gen_krea::quant::QLinear;

/// A deterministic pseudo-random vector in `[-1, 1)` (mirrors the sc-12274 gate's generator — no `rand`
/// dev-dep, and a fixed seed keeps the bit-identity leg reproducible).
fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64)
                .wrapping_add(seed)
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// A CUDA device meeting the INT8-ConvRot **sm_89** floor (locked decision 7 / sc-9300), or `None` with
/// a reason. Note the floor is sm_89 — *not* NVFP4's sm_120: this lane is offered on RTX 40-series and
/// up, and gating this test at sm_120 would silently skip it on exactly the cards sc-12301 cares about.
fn int8_device() -> Option<Device> {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-12301] no CUDA device; skipping the INT8-ConvRot workspace gate");
            return None;
        }
    };
    let lt = CublasLt::new(&dev).expect("cuBLASLt handle");
    match lt.meets_fp8_floor() {
        Ok(true) => {
            eprintln!(
                "[sc-12301] device compute cap = {:?} (>= sm_89, INT8-ConvRot eligible)",
                lt.compute_cap().unwrap()
            );
            Some(dev)
        }
        _ => {
            eprintln!(
                "[sc-12301] device cap {:?} < 8.9; the INT8-ConvRot lane is not offered here, skipping",
                lt.compute_cap().ok()
            );
            None
        }
    }
}

/// The int8 parts of one synthetic ConvRot projection: `(N, K)` codes as `I64` **on the CPU** (exactly
/// how `Weights::get_int8_codes` materializes them, to avoid holding them at 8× on the GPU) plus an
/// `[N]` per-output-row scale.
fn convrot_parts(out_dim: usize, in_dim: usize, seed: u64) -> (Tensor, Vec<f32>) {
    let codes: Vec<i64> = pseudo_random(out_dim * in_dim, seed)
        .into_iter()
        .map(|v| (v * 127.0).round().clamp(-127.0, 127.0) as i64)
        .collect();
    let w_i8 = Tensor::from_vec(codes, (out_dim, in_dim), &Device::Cpu).unwrap();
    let scale = vec![0.01f32; out_dim];
    (w_i8, scale)
}

/// **sc-12301 measurement + regression gate: one `CublasLt` handle costs ~32 MiB, and
/// `QLinear::convrot_int8` paid it per projection.**
///
/// The story's ~7 GiB headline is arithmetic — sc-12274's measured 32.00 MiB/handle × the ~224
/// projections `eight_bit_linear.rs` documents for the ConvRot DiT. **This test is the reading** for the
/// int8 lane, isolated from any checkpoint. Four legs:
///
/// 1. **Bare handles** — `N × CublasLt::new`, no projections at all. Isolates the handle's own cost;
///    this slope is the quantity the story multiplies by 224.
/// 2. **`convrot_int8` (private handle)** — the real constructor, on a deliberately TINY weight
///    (`[512, 512]` → 256 KiB of int8 codes). If per-projection VRAM lands near 32 MiB, the only thing
///    it can be is the workspace: the codes are <1% of it. Pins the by-design private cost.
/// 3. **`convrot_int8_in` (THE FIX)** — the same projections against **one** shared [`Int8Context`],
///    measured as **two** readings: the context's own cost (must be ~one workspace) and the N
///    projections' (must carry none). Two readings rather than one because inferring the workspace as
///    `used − logical_code_bytes` silently folds in the driver's allocation granularity — a 0.25 MiB
///    staged buffer costs a whole 1.00 MiB granule, and that rounding is not workspace. **Leg 3 is the
///    assertion that fails if a per-projection `CublasLt::new` returns.**
/// 4. **Bit-identity** — every shared-workspace projection's forward must be bit-identical to a
///    private-handle reference. The 32 MiB scratch is the one genuinely shared *mutable* resource, so
///    "sharing is safe because a stream serializes its kernels" is a reasoning chain; this is the check.
///
/// Legs 2 and 3 together are the regression gate: the private constructor *should* still cost a whole
/// workspace (it is for one-off projections and tests), and the constructor the loader uses must not.
#[test]
fn convrot_int8_shares_one_cublaslt_workspace_across_projections() {
    use candle_gen::candle_core::cuda_backend::cudarc::driver::result as cuda;
    let Some(dev) = int8_device() else { return };

    /// `CublasLt::WORKSPACE` — private to the handle, restated here as the *predicted* per-handle cost
    /// this test exists to confirm or refute against the driver.
    const WORKSPACE: usize = 32 * 1024 * 1024;
    /// The ConvRot DiT's int8 projection count, per `eight_bit_linear.rs` — the multiplier in the
    /// story's ~7 GiB claim. Used only for the extrapolation print, not measured here.
    const CONVROT_PROJECTIONS: f64 = 224.0;
    const N: usize = 32;
    /// The ConvRot regular-Hadamard order (`convrot_groupsize`), a power of four dividing K.
    const G: usize = 256;
    let mib = |b: f64| b / (1024.0 * 1024.0);
    let gib = |b: f64| b / (1024.0 * 1024.0 * 1024.0);

    let free_at = || {
        dev.synchronize().unwrap();
        let (free, _total) = cuda::mem_get_info().unwrap();
        free
    };

    eprintln!(
        "\n[sc-12301] ===== INT8-ConvRot PER-PROJECTION cuBLASLt WORKSPACE (weights-free) ====="
    );
    eprintln!("[sc-12301] predicted CublasLt::WORKSPACE = {WORKSPACE} B (32 MiB)");

    // --- Leg 1: bare handles ------------------------------------------------------------------
    let base = free_at();
    let mut handles = Vec::with_capacity(N);
    for _ in 0..N {
        handles.push(CublasLt::new(&dev).expect("cuBLASLt handle"));
    }
    let leg1_used = base.saturating_sub(free_at()) as f64;
    let leg1_per_handle = leg1_used / N as f64;
    drop(handles);
    eprintln!(
        "[sc-12301] leg1 TOTAL: {N} bare handles = {:.2} MiB → {:.2} MiB/handle (predicted {:.2})",
        mib(leg1_used),
        mib(leg1_per_handle),
        mib(WORKSPACE as f64)
    );

    // --- Leg 2: convrot_int8 — the private-handle constructor ----------------------------------
    // K=512 divides G=256; the codes stay CPU-resident (as the loader materializes them) and
    // `from_per_channel_parts` stages them to a 1 byte/elem device buffer.
    let (out_dim, in_dim) = (512usize, 512usize);
    let (w_i8, scale) = convrot_parts(out_dim, in_dim, 77);
    let staged_codes = (out_dim * in_dim) as f64; // 1 byte/elem once staged to i8

    let base2 = free_at();
    let mut private = Vec::with_capacity(N);
    for _ in 0..N {
        let lin = QLinear::convrot_int8(w_i8.clone(), scale.clone(), G, None, &dev)
            .expect("convrot_int8 on a >= sm_89 CUDA device");
        assert!(
            lin.is_convrot_int8(),
            "the measurement is only meaningful if these are really int8 projections"
        );
        private.push(lin);
    }
    let leg2_used = base2.saturating_sub(free_at()) as f64;
    let leg2_per_layer = leg2_used / N as f64;

    eprintln!(
        "[sc-12301] leg2 convrot_int8 (PRIVATE) x{N} ([{out_dim},{in_dim}]): {:.2} MiB → {:.2} MiB/proj",
        mib(leg2_used),
        mib(leg2_per_layer)
    );
    eprintln!(
        "[sc-12301] leg2   a weights-only footprint sum would report {:.2} MiB/proj of int8 codes \
         ({:.2} MiB total) — everything above that is invisible to it",
        mib(staged_codes),
        mib(staged_codes * N as f64)
    );
    drop(private);

    // --- Leg 3: THE FIX — the same projections against ONE shared context ----------------------
    // Measured in TWO readings, deliberately. Inferring the workspace as `used - logical_code_bytes`
    // conflates it with the driver's allocation granularity: a 0.25 MiB staged buffer actually costs a
    // whole 1.00 MiB granule, so 32 projections carry ~24 MiB of rounding that is NOT workspace and
    // must not be counted as it. Reading the context and the projections separately measures each for
    // what it is, and needs no assumption about either.
    let base_ctx = free_at();
    let ctx = Int8Context::new(&dev).expect("shared cuBLASLt context");
    assert!(
        ctx.is_int8(),
        "a CUDA device must yield a live int8 context, else leg 3 proves nothing"
    );
    let ctx_cost = base_ctx.saturating_sub(free_at()) as f64;

    let base_projs = free_at();
    let mut shared = Vec::with_capacity(N);
    for _ in 0..N {
        let lin = QLinear::convrot_int8_in(w_i8.clone(), scale.clone(), G, None, &dev, &ctx)
            .expect("convrot_int8_in against a live shared context");
        assert!(
            lin.is_convrot_int8(),
            "the shared-context projections must still be int8 — a 'fix' that silently dropped every \
             projection to the dequant-dense fallback would also 'save' the workspace"
        );
        shared.push(lin);
    }
    let projs_cost = base_projs.saturating_sub(free_at()) as f64;
    // The real resident cost of ONE projection once the handle is shared: its staged codes, rounded up
    // to the driver's allocation granule. No workspace in it — that is the whole point.
    let per_proj_alloc = projs_cost / N as f64;
    let leg3_used = ctx_cost + projs_cost;
    let leg3_per_layer = leg3_used / N as f64;
    // How many 32 MiB workspaces the whole shared set actually paid for. Must be ~1.
    let leg3_workspaces = (leg3_used - projs_cost) / WORKSPACE as f64;
    // What leg 2's projections spent on duplicated workspace, now that a projection's own resident cost
    // is measured rather than assumed: 32 handles x 32 MiB.
    let leg2_workspace = leg2_used - per_proj_alloc * N as f64;

    eprintln!(
        "[sc-12301] leg3 convrot_int8_in (SHARED) x{N}: {:.2} MiB → {:.2} MiB/proj (vs {:.2} private)",
        mib(leg3_used),
        mib(leg3_per_layer),
        mib(leg2_per_layer)
    );
    eprintln!(
        "[sc-12301] leg3   split: ONE shared context = {:.2} MiB ({:.2} workspaces) + {N} projections \
         = {:.2} MiB ({:.2} MiB/proj of staged codes, granule-rounded from {:.2} MiB logical)",
        mib(ctx_cost),
        ctx_cost / WORKSPACE as f64,
        mib(projs_cost),
        mib(per_proj_alloc),
        mib(staged_codes)
    );
    eprintln!(
        "[sc-12301] leg3   duplicated workspace: private {:.2} MiB ({:.1} handles) → shared {:.2} MiB \
         ({:.2} handles). SAVED {:.2} MiB over {N} projections",
        mib(leg2_workspace),
        leg2_workspace / WORKSPACE as f64,
        mib(leg3_used - projs_cost),
        leg3_workspaces,
        mib(leg2_used - leg3_used)
    );
    // The story's headline, restated with THIS box's measured per-projection workspace. Weights-free and
    // synthetic: it is a measured per-projection overhead × a documented count, NOT a reading on a real
    // ConvRot trunk (whose projections differ in shape and whose codes dominate the weight side). It
    // confirms the ORDER of the claim; sc-12381 carries the trunk reading.
    eprintln!(
        "[sc-12301] EXTRAPOLATION: at the measured {:.2} MiB of workspace per projection, a \
         {CONVROT_PROJECTIONS:.0}-projection ConvRot DiT carried {:.2} GiB of duplicated scratch (story \
         claims ~7 GiB) — now ONE {:.0} MiB workspace. Synthetic shapes: the order, not a trunk reading.",
        mib(leg2_workspace / N as f64),
        gib(leg2_workspace / N as f64 * CONVROT_PROJECTIONS),
        mib(WORKSPACE as f64)
    );

    // --- Assertions: the mechanism, not the exact byte count -----------------------------------
    // Tolerance is generous (±25%): the handle also allocates cuBLASLt-internal state and the driver
    // rounds allocations. The claim under test is "each handle costs a 32 MiB-scale buffer".
    assert!(
        leg1_per_handle > 0.75 * WORKSPACE as f64 && leg1_per_handle < 1.25 * WORKSPACE as f64,
        "a bare CublasLt handle measured {:.2} MiB, expected ~{:.2} MiB (CublasLt::WORKSPACE). \
         sc-12301's arithmetic rests on this being ~32 MiB — if it is not, the story's headline is wrong.",
        mib(leg1_per_handle),
        mib(WORKSPACE as f64)
    );
    // Leg 2: the private-handle constructor still costs a whole workspace per projection. BY DESIGN
    // (`convrot_int8` is for a one-off projection / a test) — pinned so the cost stays visible and
    // nobody reaches for it in a loader.
    assert!(
        leg2_per_layer > 0.75 * WORKSPACE as f64,
        "`convrot_int8` builds a private handle per projection, so it should still measure ~32 MiB \
         each; got {:.2} MiB. If this dropped, the private/shared distinction has been lost.",
        mib(leg2_per_layer)
    );
    assert!(
        leg2_workspace > 10.0 * per_proj_alloc * N as f64,
        "sc-12301's claim is that the per-projection workspace DWARFS the int8 codes it serves; \
         measured {:.2} MiB of workspace vs {:.2} MiB of actually-resident codes",
        mib(leg2_workspace),
        mib(per_proj_alloc * N as f64)
    );
    // Leg 3: THE FIX. ONE workspace for the whole set, however many projections — this is the assertion
    // that fails if anyone reintroduces a per-projection `CublasLt::new` in `linear_detect`'s path.
    // Measured as the context's own cost, so the driver's per-buffer rounding cannot leak into it.
    assert!(
        leg3_workspaces < 1.5,
        "a shared Int8Context must cost ONE {:.2} MiB workspace for all {N} projections, but the \
         non-projection VRAM was {:.2} MiB (≈{:.2} workspaces) — the handle is not actually being \
         shared (sc-12301 regression).",
        mib(WORKSPACE as f64),
        mib(leg3_used - projs_cost),
        leg3_workspaces
    );
    // ...and a shared projection must carry NO workspace of its own: its resident cost is its staged
    // codes (granule-rounded), orders below the private constructor's.
    assert!(
        per_proj_alloc < 0.25 * WORKSPACE as f64,
        "a projection built against a shared context must not carry a workspace: measured \
         {:.2} MiB/proj, which is workspace-scale (~{:.2} MiB) rather than code-scale (~{:.2} MiB)",
        mib(per_proj_alloc),
        mib(WORKSPACE as f64),
        mib(staged_codes)
    );
    assert!(
        leg3_per_layer < 0.25 * leg2_per_layer,
        "sharing the handle must collapse per-projection VRAM toward the codes alone: shared \
         {:.2} MiB/proj vs private {:.2} MiB/proj is not a real saving",
        mib(leg3_per_layer),
        mib(leg2_per_layer)
    );

    // --- Leg 4: sharing must not change a single bit -------------------------------------------
    // Drive ALL N shared projections through a forward so the 32 MiB scratch is genuinely reused across
    // them, then require every output to be bit-identical to a fresh private-handle projection holding
    // the same weight. Activations live on the compute device; the codes stayed on the CPU.
    let x = Tensor::from_vec(pseudo_random(64 * in_dim, 5150), (64, in_dim), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let bits = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };

    let reference = QLinear::convrot_int8(w_i8.clone(), scale.clone(), G, None, &dev)
        .expect("private-handle reference projection");
    let want = bits(&reference.forward(&x).expect("reference forward"));

    for (i, lin) in shared.iter().enumerate() {
        let got = bits(&lin.forward(&x).expect("shared-context forward"));
        assert_eq!(
            got.len(),
            want.len(),
            "shared projection {i} produced a different shape than the private-handle reference"
        );
        assert!(
            got == want,
            "shared projection {i}/{N} is NOT bit-identical to a private-handle reference. The 32 MiB \
             cuBLASLt scratch is the one genuinely shared MUTABLE resource, so this is the check that \
             sharing it across projections is safe — if it fails, sc-12301's fix is unsound, not just \
             slow.",
        );
    }
    eprintln!(
        "[sc-12301] leg4 BIT-IDENTITY: all {N} shared-workspace projections == private-handle \
         reference, bit for bit ({} outputs each)",
        want.len()
    );
}

/// **The device-identity guard — a hazard that only exists once handles are shared.**
///
/// A per-projection handle could never be bound to the wrong device; a shared one can. A context built
/// on the CPU (i.e. empty) handed to a projection on CUDA must be a **typed error at load**, not a
/// silent collapse into a cross-device dequant-dense matmul — that is the F-121 / sc-11208 property
/// sc-12301 had to carry over from the eager per-projection build.
#[test]
fn convrot_int8_in_rejects_a_context_that_is_not_bound_to_the_compute_device() {
    let Some(dev) = int8_device() else { return };
    let (out_dim, in_dim) = (512usize, 512usize);
    let (w_i8, scale) = convrot_parts(out_dim, in_dim, 11);

    // An empty context (what `Int8Context::new` yields off CUDA) is NOT a valid handle source for a
    // projection whose compute device is CUDA.
    let cpu_ctx = Int8Context::new(&Device::Cpu).expect("cpu context");
    assert!(
        !cpu_ctx.is_int8(),
        "a CPU device must yield an empty context — there is no handle to bind"
    );
    // `match`, not `expect_err`: the latter needs `QLinear: Debug`, and a production type should not
    // grow a derive just to widen a test's error reporting.
    let err = match QLinear::convrot_int8_in(w_i8.clone(), scale.clone(), 256, None, &dev, &cpu_ctx)
    {
        Ok(_) => panic!(
            "an empty context on a CUDA projection must be a typed error, not a silent \
             dequant-dense fallback (F-121)"
        ),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("sc-12301"),
        "the error must name the story that introduced sharing, so the next reader knows why an \
         empty context is fatal here; got: {msg}"
    );

    // The same context IS correct for a CPU projection — there the dequant-dense fallback is the path.
    let cpu_lin = QLinear::convrot_int8_in(w_i8, scale, 256, None, &Device::Cpu, &cpu_ctx)
        .expect("an empty context is the honest state for a CPU projection");
    assert!(cpu_lin.is_convrot_int8());
}
