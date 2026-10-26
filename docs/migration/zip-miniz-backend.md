# ZIP 9: preserve the consumer's default flate2 backend

## Decision

Candle and Kokoro do not require zlib-rs. Both use ZIP 9 with default features
disabled, `deflate-flate2`, and an explicit `flate2` dependency enabling only
`rust_backend` (miniz). The direct backend dependency is intentional: ZIP's flate2
dependency disables defaults, so a standalone consumer must not depend on an
unrelated PNG crate enabling a backend for it.

ZIP 9's plain `deflate` feature is not an alternative: it enables both
`deflate-zopfli` and `deflate-flate2-zlib-rs`. The former explicit
`deflate-flate2-zlib-rs` selection from inference commit `70fba92c68` unified
`flate2/zlib-rs` into downstream consumers, overriding the normal miniz encoder.
This correction retains ZIP 9 and its security fixes; it changes no tensor math,
kernels, model weights, or decoded archive contents. It no longer forces zlib-rs;
a consumer can still select another backend independently.

## Required archive capabilities

- Vendored Candle `src/npy.rs` reads STORED and DEFLATE NPZ files (eager, named,
  lazy, and header-only paths). Its public `Tensor::write_npz` explicitly writes
  STORED entries.
- Vendored Candle `src/pickle.rs` reads torch ZIP metadata and tensor storage,
  including lazy storage reads. It does not write archives.
- Kokoro `src/weights.rs` loads checkpoint sections through Candle and directly
  reads voice ZIP byte-order markers and f32 storage. Production is read-only.
- DEFLATE writing remains available through flate2 and is exercised by the
  existing test fixture writers; no zopfli or backend-specific writing is needed.

The weights-free `candle-audio-kokoro` tests already cover STORED/DEFLATE voices,
STORED NPZ writing and eager/named/lazy NPZ reads, and STORED/DEFLATE torch state
dictionaries. They exercise the actual patched Candle dependency from a workspace
member. Vendored Candle is excluded from workspace membership, so its upstream
suite must be run from a standalone copy rather than with workspace `-p` testing.

## Verification

Run from the inference workspace:

```bash
cargo tree --locked -p candle-audio-kokoro -i flate2 -e features
```

```bash
cargo tree --locked -p candle-core -i flate2 -e features
```

```bash
cargo test --locked -p candle-audio-kokoro --lib --tests
```

```bash
cargo clippy --locked -p candle-audio-kokoro --all-targets -- -D warnings
```

The feature trees must show `rust_backend` / `miniz_oxide`, not `zlib-rs` or
`any_zlib`. The corrected lockfile removes zlib-rs entirely. Targeted Kokoro trees
were also checked for `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`,
`x86_64-pc-windows-msvc`, and `aarch64-apple-ios`; these are feature-resolution
checks, not cross-target compilation or GPU tests.

## SceneWorks: next inference pin

No SceneWorks pin or fixture is changed by this inference PR. When adopting it:

1. Run the exact-byte film guards
   `checked_in_fixture_plates_match_the_generator_byte_for_byte` and
   `the_checked_in_labeled_frames_match_the_generator_byte_for_byte`. The fixtures
   are seven PNGs under `config/film-harness/courier-workshop/references/` and 44
   under `config/film-harness/review-eval/frames/`. If the adopting branch already
   took the zlib-rs re-encoding (`18d720cb0`), regenerate with the corrected final
   consumer graph. If it still has the original miniz fixtures, avoid gratuitous
   re-encoding. For example, `sh020_correct_1.png` grew from 3,860 to 5,575 bytes
   with zlib-rs (IDAT `0x0edb` to `0x158e`); the pixels did not change.
2. Run `a_zip_bomb_after_the_image_data_refuses_instead_of_inflating` and
   `a_zip_bomb_refuses_instead_of_inflating` in `workflow_png.rs`. Branch commit
   `3940904e2` relaxed the compressed-size precondition from 64 KiB to payload/64
   because of zlib-rs output. Do not change production decompression limits
   (1 MiB per text payload, approximately 8 MiB cumulative ancillary metadata).
3. Inspect the final application feature graph, not just the lockfile.
   SceneWorks `apps/desktop/Cargo.toml` independently requests ZIP 9 `deflate` on
   Linux/Windows. A graph including that package still selects zlib-rs and zopfli,
   even with this inference fix. This inference change restores miniz only where
   no other dependency requests a higher-priority backend. API/worker-only and
   macOS graphs must be checked separately from Linux/Windows desktop graphs.
