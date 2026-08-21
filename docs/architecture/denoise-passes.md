# Chained denoise passes

> Epic 20414 · story sc-20415 (the contract) · sc-20416/20417 (solver + scheduler math) ·
> sc-20418 (chained execution)

The `advanced.denoisePasses` request block lets one job run **several complete denoise runs in
sequence**, each with its own step count, sampler, scheduler, `denoise` fraction, guidance and
adapter weights. The typed contract lives in
[`gen_core::denoise_passes`](../../crates/contracts/gen-core/src/denoise_passes.rs); this document
is the prose statement of the same thing, plus the runtime invariant execution must satisfy.

## The shape

```jsonc
{
  "seed": 1234567890, "steps": 20, "sampler": "euler", "scheduler": "normal", "guidance": 3.5,
  "advanced": {
    "denoisePasses": [
      { "steps": 10, "sampler": "euler",    "scheduler": "normal", "denoise": 1.0 },
      { "steps": 4,  "sampler": "dpmpp_2m", "scheduler": "karras", "denoise": 0.2, "guidance": 2.0 }
    ]
  }
}
```

Every per-pass field is optional. Each one resolves down a three-rung ladder:

```
pass value  →  top-level request value  →  model default
```

`denoise` is the exception: it has no top-level analogue and defaults to `1.0` (the full
trajectory). `GenerationRequest::strength` is *not* its top-level form — `strength` is the img2img
fidelity of a conditioning image, a whole-request concept, while `denoise` is one pass's entry point
into its own schedule.

## Resolution happens at the request boundary

`gen_core::resolve_denoise_plan` turns a request into a `ResolvedDenoisePlan` where **every** field
is explicit and the per-pass seed is already derived. Execution and the replay metadata read the
plan, never the request — so nothing downstream re-derives a default and no rendered image depends
on implicit UI state.

A request with **no** `denoisePasses` resolves through the same function to a **one-pass** plan
carrying the top-level request's own values. The legacy path and the chained path are therefore one
code path, not two that can drift, and a provider can adopt the plan unconditionally instead of
branching on whether the field is present.

## The runtime invariant

For pass *n+1*:

1. **It starts from the latent pass *n* produced.** There is **no VAE round trip** at a pass
   boundary — no decode to pixels, no re-encode — so no reconstruction loss and no resample is
   introduced between passes.
2. **It builds a fresh schedule.** A new sigma schedule from that pass's own scheduler and step
   count, entered at the point its `denoise` fraction names. Schedules are never shared or spliced
   across passes.
3. **It starts with empty solver history.** Multistep solvers (`dpmpp_2m`, `uni_pc`, …) accumulate
   derivatives that belong to the schedule they were taken on; carrying them across a schedule reset
   would mix quantities sampled at incomparable sigmas.
4. **It draws from its own noise stream.** The pass seed is `denoise_pass_seed(job_seed, index)` and
   is recorded in the plan, so a replay reproduces the render exactly.

Pass 0's seed is the job seed **verbatim**. That is deliberate: it makes a one-pass plan bit-identical
to the legacy single-trajectory render, and every already-persisted replay payload names only the job
seed. Later passes mix a golden-ratio salt in at the pass index and run a splitmix64 finalizer, so
adjacent passes are not merely offset seeds.

### Plan seeds are JSON strings

Because the finalizer is splitmix64, a derived pass seed is uniform over the whole `u64` range and
sits above 2^53 in ~99.9% of cases (`default_seed()` — nanoseconds since the epoch — already does).
The other end of this contract is JavaScript, where `JSON.parse` rounds such a number to the nearest
`f64` **without raising anything**: `5145724004617983535` reads back as `5145724004617983000`, and the
replay silently produces a different image. So in a resolved plan, `jobSeed` and every per-pass `seed`
are written as **decimal strings**, and `ResolvedDenoisePlan::from_json` refuses a bare JSON number
rather than accepting the exact shape a truncating producer emits. Consumers parse them with a
full-width integer type (`BigInt`, `u64`).

The request's own top-level `seed` keeps its pre-existing bare-number form — that wire format predates
this contract and is not this contract's to change. The asymmetry is documented in the fixture
README, which is what the SceneWorks-side implementation reads.

## Denoise passes are not Krea RAW multi-phase denoise

`advanced.phases` (epic 13879, sc-13884) predates this and is a different mechanism. **The two are
mutually exclusive in one request**; a request that sets both is rejected before execution, because
there is no defined composition of "slice one global schedule" with "build a fresh schedule per
pass".

| | `phases` — RAW **multi-phase denoise** | `denoisePasses` — **denoise passes** |
|---|---|---|
| Schedule | ONE global sigma schedule, shared | a **fresh** schedule per pass |
| A phase/pass is | a contiguous **slice** of that schedule | a complete denoise run of its own |
| Boundary | resumes at the sigma the prior phase reached — no reset | full reset: new schedule, new solver history |
| Total steps | the **sum** of the phases' steps | each pass runs its own `steps` |
| Sampler / scheduler | one, request-level | per pass |
| `denoise` fraction | none — the trajectory is continuous | per pass, in `(0, 1]` |
| Adapters, empty list | the **bare base model** | the load-time stack at its load-time scales |
| Adapters, non-empty | the exact active set for that phase | **weight overrides** for the named adapters |
| Seeds | one trajectory, one stream | one derived seed per pass |
| Scope | the Krea RAW family | model-agnostic |

The adapter row is the trap worth restating: a phase's `adapters` list *selects* what is active, so
an empty list means "run bare base". A pass's `adapters` list *overrides weights*, so an empty list
means "leave the load-time stack exactly as loaded". Both reuse the same `PhaseAdapter`
`(load-time index, optional weight)` reference type; only the meaning of the empty list differs.

## Validation

Everything below is checked **before execution**, and every error names the offending **pass index**
and **field** (`DenoisePassError::pass_index()` / `::field()`, rendered as e.g.
`advanced.denoisePasses[1].denoise`):

- `phases` and `denoisePasses` both set;
- the model does not advertise `Capabilities::supports_denoise_passes`;
- an empty pass array, or more than `MAX_DENOISE_PASSES`;
- `steps <= 0`, or above the shared step sanity cap;
- `denoise` outside `(0, 1]`, or non-finite;
- non-finite guidance or adapter weight (including a JSON number that is finite as `f64` but
  overflows to `±Inf` when narrowed to the contract's `f32`);
- an unknown sampler or scheduler id — validated against the model's advertised menu when it has
  one, else the curated `Solver` / `Scheduler` registries;
- an **advertised but unhonorable** sampler or scheduler id (see *What a model honors per pass*);
- a per-pass `guidance` on a model with no guidance axis, or a per-pass `adapters` override on a
  model whose adapters cannot be re-weighted per pass (same section);
- an adapter index past the end of the load-time stack, or two overrides for the same adapter in one
  pass;
- structurally malformed JSON, including an **unknown key**.

Capability gaps (`supports_denoise_passes`, unknown ids, unhonorable ids, unsupported per-pass
guidance/adapters) surface as the typed `Error::Unsupported`; malformed values surface as
`Error::Msg` — the same split the rest of the shared request floor uses.

The adapter *index bound* is the one check the model-free request floor cannot make: it holds a
`Capabilities`, never a `LoadSpec`, so it does not know how many adapters were provisioned. That
check belongs to the model resolving the plan against its real stack, and the fixtures mark it
`checkedBy: "model"`.

## What a model honors per pass

`Capabilities::supports_denoise_passes` says *whether* a family runs a chain.
`Capabilities::denoise_pass_surface` (`DenoisePassSurface`) says *what* it honors, and
`Capabilities::denoise_pass_capability()` projects the two into the explicit
`DenoisePassCapability` a consumer reads — pass-count cap, per-pass step cap, the per-pass **fields**
this model honors, and the per-pass sampler/scheduler menus. It is derived from the descriptor
alone: there is no model-id table anywhere in the derivation, so a family that edits its menu edits
its published surface in the same commit. Studio and the worker read it instead of maintaining a
family table.

**Advertised is not the same as honorable.** A family's flat `samplers` / `schedulers` menus
legitimately carry native ids beyond the curated registries, and those ids are meaningful on the
single-pass path. On a chain they are not automatically meaningful, and the difference is what the
intersection rules encode:

| axis | per-pass set | why |
|---|---|---|
| sampler | advertised ∩ curated `Solver` | The executor boxes a curated `Solver` per pass and has **no hook** a family could use to run a native sampler name. An uncurated id used to validate and then integrate as Euler — the wrong algorithm, reported as success. |
| scheduler | advertised ∩ (curated `Scheduler` ∪ `native_schedulers`) | The family owns `build_schedule`, so a native alias *can* be honorable — but only the family knows which, and an undeclared one falls through a resolver's native-default branch just as silently. |

Adopters resolve a pass through `gen_core::resolve_pass_solver` / `resolve_pass_scheduler`, which
reject rather than fall back, so the advertisement and the honoring cannot drift.

### The global per-pass adapter policy

A per-pass `adapters` entry re-scales a **live, revertible** adapter residual at a pass boundary.
Nearly every family instead folds LoRA/LoKr into its dense weights once at load and keeps no residual
to re-scale, so "adapter off for pass 1, on for pass 2" is physically unavailable there — and
accepting the field anyway would render the wrong image and report success.

So `DenoisePassSurface::per_pass_adapters` defaults to `false` and the shared floor **rejects** a
non-empty per-pass `adapters` list with a pass-indexed `PerPassAdaptersUnsupported`. Reject, never
accept-and-ignore. A family whose adapter seam is a real apply/revert sets it `true`; a family that
advertises no adapters at all (`supports_lora` and `supports_lokr` both `false`) gets the rejection
from the shared floor for free. An **empty** list is not an override — it means "the load-time stack
at its load-time scales" — and stays legal everywhere.

### Descriptor conformance

`registry::model_descriptor_errors` cross-checks the surface against the rest of the descriptor, so
the pairing cannot be got wrong by *inheritance* — the way `krea_2_turbo_control` inherited
`supports_denoise_passes: true` from the Turbo descriptor it derives from while having no chained
execution at all. It rejects: a control route (`control_kinds: Some(..)`) advertising the capability;
a menu with no curated sampler or no honorable scheduler (a capability no request could exercise); a
declared native scheduler that is unadvertised or shadows a curated id; `per_pass_adapters` with no
adapter support; and a non-empty surface on a descriptor that does not set the capability.

## Versioning

A **resolved plan** is a persisted replay artifact, so its JSON is stamped with
`"contractVersion": 1` (`gen_core::DENOISE_PASS_CONTRACT_VERSION`). A reader refuses a version it
does not implement rather than interpreting unfamiliar fields as familiar ones — the failure mode
that stamp exists to prevent is a silently *different* replayed image, not a parse error. Bump the
version when a change would make an older build mis-read a newer plan: a new seed derivation, a new
meaning for `denoise`, a changed resolution ladder, a changed JSON type for an existing field. A new
*optional* per-pass field that older builds may ignore is not such a change.

The **request** side carries no version, deliberately. `advanced.denoisePasses` is a bare array with
nowhere natural to put one, its decoding is strict (an unknown key is an error, not a silent drop),
and its compatibility rule needs no negotiation: absent means the ordinary single-pass render, for
every build that has ever existed.

## Compatibility

A request serialized before this contract carries no `denoisePasses` key. It decodes to the absent
state, resolves to the one-pass plan described above, and renders exactly as it did — the field is
purely additive and `None` is the default. No model advertises `supports_denoise_passes` yet, so
every existing provider rejects a chained request with a typed `Error::Unsupported` rather than
silently running only its first pass.

## Cross-language fixtures

`crates/contracts/gen-core/tests/fixtures/denoise_passes/` holds the shared valid / invalid /
resolved-plan fixtures. SceneWorks reads the **same files** to prove its encoder round-trips
losslessly against this contract; see that directory's `README.md` for the envelope and the
regeneration rules.
