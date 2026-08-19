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
- an adapter index past the end of the load-time stack, or two overrides for the same adapter in one
  pass;
- structurally malformed JSON, including an **unknown key**.

Capability gaps (`supports_denoise_passes`, unknown ids) surface as the typed `Error::Unsupported`;
malformed values surface as `Error::Msg` — the same split the rest of the shared request floor uses.

The adapter *index bound* is the one check the model-free request floor cannot make: it holds a
`Capabilities`, never a `LoadSpec`, so it does not know how many adapters were provisioned. That
check belongs to the model resolving the plan against its real stack, and the fixtures mark it
`checkedBy: "model"`.

## Versioning

A **resolved plan** is a persisted replay artifact, so its JSON is stamped with
`"contractVersion": 1` (`gen_core::DENOISE_PASS_CONTRACT_VERSION`). A reader refuses a version it
does not implement rather than interpreting unfamiliar fields as familiar ones — the failure mode
that stamp exists to prevent is a silently *different* replayed image, not a parse error. Bump the
version when a change would make an older build mis-read a newer plan: a new seed derivation, a new
meaning for `denoise`, a changed resolution ladder. A new *optional* per-pass field that older
builds may ignore is not such a change.

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
