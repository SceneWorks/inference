# Iris HOST source boundaries (sc-26270)

Checked on 2026-10-26 against inference `2697693a981f69bb042a171520e096589a6146e7`.
This records a closed raw-layout gap and the remaining complete-HOST obligations;
it is not phone acceptance, a physical-footprint law, or an approval of a premise.

## Apple mutex requested layout

The exact [libc 0.2.183 package](https://static.crates.io/crates/libc/libc-0.2.183.crate)
was retrieved and checked against SHA-256
`b5b646652bf6661599e1da8901b3b9522896f01e736bad5f723fe7a3a27f899d`.
That is the libc version/checksum in Rust
`ac68faa20c58cbccd01ee7208bf3b6e93a7d7f96/library/Cargo.lock`, even though the
application workspace currently resolves libc 0.2.186. Substituting the
workspace version for the standard library's version would not prove this term.

The package's migrated definition lives in
`src/new/apple/libpthread/sys/_pthread/_pthread_types.rs`, rather than only the
older `src/unix/bsd/apple/mod.rs` path. The selection/export chain is:

- `src/lib.rs` includes/reexports `new` and selects `unix`.
- `src/new/mod.rs` selects `apple` by `target_vendor = "apple"`.
- `src/new/apple/mod.rs` exports `libpthread::pthread_` and its `pthread` module.
- `src/new/apple/libpthread/pthread_/pthread.rs` exports the `_pthread_types` definitions.
- `_pthread_types.rs` selects 56 opaque bytes for 64-bit pointers; its `s!`
  struct contains `c_long` followed by those bytes.
- `src/primitives.rs` makes non-Windows 64-bit `c_long` an `i64`; `s!` in
  `src/macros.rs` supplies `repr(C)`.

The C layout is therefore size **64**, alignment **8**. A native Rust 1.96.0
(`ac68faa20`) pure-layout executable linked against the retrieved package
confirmed both `Layout::new::<libc::pthread_mutex_t>()` and its `UnsafeCell`
wrapper have that layout. No mutex lock, model, tensor, allocator probe, or GPU
work was executed. The package was compiled directly with:

```sh
rustc +1.96.0 --crate-name libc --crate-type rlib --edition=2021 \
  --cfg 'feature="std"' -C debuginfo=0 libc-0.2.183/src/lib.rs \
  -o liblibc_0_2_183.rlib
```

The layout executable used the resulting rlib with `--extern libc=...` and
asserted `(size, align) == (64, 8)`. Output:

```text
libc=0.2.183 target=Apple64 pthread_mutex_t.size=64 pthread_mutex_t.align=8 unsafe_cell.size=64 status=checked-layout-only
```

Rust's exact source selects `sys/sync/mutex/pthread.rs` on Apple. Its `get`
allocates `Box::pin(pal::Mutex::new())` through `OnceBox`; the Unix pal mutex's
sole field is `UnsafeCell<libc::pthread_mutex_t>`. The existing, separately
unapproved `rust-toolchain/requested-host-layout-v1` premise covers the private
`repr(Rust)` wrapper. Source file SHA-256 values at the pinned Rust commit:

| Source below `library/std/src/` | SHA-256 |
| --- | --- |
| `sys/sync/mutex/mod.rs` | `1c715675a2bc34471b96de58f7aaaf2db411289cf838d17edc61cb2d6b006aa2` |
| `sys/sync/mutex/pthread.rs` | `1aa80435780abe34a365179635e92fb5935894fb9b4f615eb45b336958c28035` |
| `sys/pal/unix/sync/mutex.rs` | `1df3c112f84ccfa7ff048aa88a604dc6029b405f59e0aa96d35381b589aa9b17` |

`host_mutex.rs` gates this term on Apple64 and the actual build compiler commit.
The source-bound depth/restoration analysis charges 64 requested bytes per
`pixel_embed_body -> pixel_pos` lock: one for depth and the actual restoration
plan's `forward_count()` for restoration. Counting every lock overprices warm
reuse and avoids assuming a successful `OnceBox` race. Checked multiplication
rejects overflow. Other targets/compilers retain the named ABI gap.

`KnownHostFragments.mutexes` is diagnostic source arithmetic, conditional on
the layout premise; it is never a complete `MetadataHostBound`. Removing the
now-recovered ABI gap on the matching compiler/target does not remove
`CapturedMetadata`. pthread internals, malloc usable size, driver charge and
physical footprint are outside this Rust requested-layout term.

## Production tokenizer identity and remaining boundary

SceneWorks' checked download manifest names
`Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17`.
The [tokenizer at that immutable revision](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/ebb281ec70b05090aa6165b016eac8ec08e71b17/tokenizer.json)
was retrieved as source data: **7,032,403 bytes**, SHA-256
`a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7`.
Its actual graph is NFC normalization; a Sequence of regex Split and ByteLevel
with `use_regex=false`; ByteLevel processor/decoder; and BPE with 151,643 vocab
entries, 151,387 merges and 26 added tokens. Dropout, truncation and padding are
absent. The Split regex includes negative lookahead.

Iris uses `gen_core_tokenizer::TextTokenizer`, resolved to tokenizers 0.21.4
with Oniguruma. `IrisTextEncoder::assemble` loads the tokenizer and encodes the
prefix/suffix; `encode_in` encodes each full caption before the final token
window is truncated. A token-window bound alone therefore cannot bound cold
construction or temporary full-caption normalization/tokenization.

The vendored CLIP certificate is for a different exact literal, regex engine
and source graph. Its checked envelopes must not be passed as
`TokenizerHostBound`. The native loader also accepts other supported HF
tokenizer graphs; no source rule currently restricts it to this frozen JSON.
A model-specific estimate may bind this actual retained source identity
without changing native accepted inputs, but only under an explicitly approved
engineering policy with distinct evidence/limitations. It cannot manufacture
the existing source-certificate type.

Complete source-certified HOST integration still needs the execution-time
metadata call accounting and multiplicities (including streamed header/key
parses and model construction), plus the actual tokenizer construction/search
graph where generation reaches it. Existing JSON/YAML/table/payload fragments
are useful inputs, not evidence that this composition is complete. Source
retrieval succeeded: no access, credential or missing-package wall was found.
The remaining source work must not be described as an external blocker.

## Local validation of the ABI repair

Native Apple64 Rust 1.96.0 checks passed using four build jobs, zero debug info,
incremental compilation disabled, the MacOSX26.5 SDK and rust-lld. The existing
verified MLX prebuilt was reused; no native MLX rebuild or model execution was
needed.

- `cargo check --locked --offline -p mlx-gen-iris --no-default-features`: passed.
- `cargo clippy --locked --offline -p mlx-gen-iris --no-default-features --lib --tests -- -D warnings`: passed.
- The pure HOST tests `cold_mutex_requests_are_counted_without_reuse_credit`,
  `complete_host_bound_refuses_named_gaps_and_adds_test_only_terms`, and
  `no_facts_host_refusal_never_opens_the_supplied_source`: 3 passed, 0 ignored,
  167 filtered out. Runtime 0.01 seconds; test compilation/linking 40.66 seconds.
- Standalone helper builds with authenticated and absent compiler markers each
  passed their target/provenance/overflow check. The separate actual-libc layout
  check above passed independently of the helper's formula.

The new ordinary test is pure layout/arithmetic and remains in the existing
library test target. Build provenance adds one `rustc -vV` invocation when the
Iris build script runs. No new CI lane, inference test, measurement matrix or
physical proof is introduced.

## Explicit engineering preparation route

The approved engineering contract permits source/checkpoint/device/OS-scoped
pressure estimates with measured spread, a minimum 32 MiB margin, and live
headroom checks. It does not assert a mathematical physical-memory guarantee.
`EstimatedSourcePreparation` is a distinct input to
`pricing_facts_with_estimated_preparation` and
`check_replay_with_estimated_preparation`; no existing default caller changes
policy merely because these APIs exist. Its constructor checks that bytes are
nonzero; it does not attest the capture or approve an arbitrary byte value.
The catalog policy owns validation of exact artifact/source, device, OS,
execution, and measured-margin provenance before constructing this input.

The named provenance is `BeforeRequestConstructionPeakWithMargin`: actual
whole-process growth above the first baseline before request preparation in a
fresh process, plus the approved policy margin. This whole-process peak is an
empirical preparation upper estimate; it is not inferred from post-load demand
or an unrelated lifetime peak. Changes to bound identities invalidate reuse.
Missing or mismatched evidence must not admit a request. The catalog must check
its selected preparation envelope before its own first source verification or
capture, including token/key preparation preceding Iris pricing.

Iris checks that envelope before retained-source verification and again just
before compatibility parsing. The existing inventory still verifies configured
paths and source identity before and after the read, and preserves capacity
callback errors. The pricing API supports generation, restoration, and depth,
including their replay tokens. The replay API preserves wrong-prefix rejection
ordering, and checks capacity before JSON preflight for valid prefixes. These
checks cover the aggregate empirical preparation estimate; the compatibility
readers are not individually instrumented allocation proofs.

Returned HOST analysis remains incomplete. Neither API constructs a
`MetadataHostBound` or `TokenizerHostBound`; the source-certified generation
route still refuses at aa011. General native tokenizer/Array input support is
unchanged, and this API supplies no phone-fit certification for arbitrary
inputs outside the measured catalog/model scope.

Six focused pure tests passed (0 failed, 0 ignored; runtime 0.04 seconds):
estimated replay cancellation/prefix/default behavior; estimated generation
pricing with retained inventory and incomplete HOST analysis; cancellation at
the second envelope check and after-read source mutation rejection; and the
existing replay-gap, complete-HOST-gap, and no-source-open refusal regressions.
The fixtures use synthetic preparation bytes explicitly marked as test data,
not phone measurements.
