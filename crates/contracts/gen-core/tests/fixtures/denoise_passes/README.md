# Denoise-pass cross-language fixtures (sc-20415, epic 20414)

These `.json` files are the **shared** contract fixtures for `advanced.denoisePasses`. Both
repositories read the *same* files:

- **inference** — `crates/contracts/gen-core/tests/denoise_pass_fixtures.rs` decodes every fixture
  into `gen_core::DenoisePass`, re-encodes it, resolves the plan, and compares against the
  `resolved/` expectation.
- **SceneWorks** — the worker-side PR of sc-20415 consumes the identical files to prove its own
  encoder/decoder is lossless against this contract. Do **not** fork or paraphrase them; a
  SceneWorks-side copy that drifts is exactly the defect they exist to prevent.

## Layout

| Path | Contents |
|---|---|
| `valid/*.json` | Requests that must be accepted, with the model context needed to resolve them. |
| `resolved/<name>.plan.json` | The exact `ResolvedDenoisePlan` each `valid/<name>.json` must resolve to, per-pass seeds included. |

Each `resolved/` file opens with `"contractVersion": 1`. A plan is a persisted replay artifact, so it
carries the contract version it was written under, and a reader refuses a version it does not
implement rather than mis-reading it. The request side carries no version — an absent
`denoisePasses` is the single-pass render for every build there has ever been.
| `invalid/*.json` | Requests that must be rejected, each naming the pass index, field and issue code. |

## Envelope

Every fixture is an object:

```jsonc
{
  "fixture": "two_pass_reference_recipe",   // must equal the file stem
  "story": "sc-20415",
  "description": "prose, for the human reading a failure",
  "request": {                              // the SceneWorks request, camelCase
    "seed": 1234567890,                     // the RESOLVED job seed
    "steps": 20, "sampler": "euler", "scheduler": "normal", "guidance": 3.5,
    "advanced": { "denoisePasses": [ /* ... */ ] }
  },
  "modelDefaults": {                        // the last rung of the resolution ladder
    "steps": 20, "sampler": "euler", "scheduler": "normal", "guidance": 3.5
  },
  "capabilities": {                         // optional; absent ⇒ the curated registry, no menus
    "supportsDenoisePasses": true,
    "samplers": ["euler", "dpmpp_2m"],
    "schedulers": ["normal", "karras"],
    "loadedAdapters": 2
  },
  "canonicalDenoisePasses": [ /* ... */ ],  // optional; the re-encoded form, when it differs
  "expectedError": {                        // invalid/ only
    "passIndex": 1, "field": "denoise", "issue": "outOfRange",
    "checkedBy": "floor"                    // optional; "floor" (default) or "model"
  }
}
```

`expectedError.checkedBy` says **where** the rejection is enforceable. `"floor"` — the default —
means the shared, model-free request floor (`Capabilities::validate_request`) rejects it, so a
consumer can pre-check the request without loading anything. `"model"` means the check needs state
only the loaded model has: today that is exactly the per-pass adapter index bound, which requires
the load-time adapter count and therefore belongs to the model resolving the plan against its real
stack. A consumer replicating this contract should mirror the same split rather than pretending a
request-boundary check can see the load spec.

`request.advanced.phases` is only ever inspected for **presence** here — the Krea RAW multi-phase
payload has its own contract and is not decoded by this fixture set. Its presence alongside
`denoisePasses` is what the mutual-exclusion fixture exercises.

`expectedError.passIndex` is `null` when the whole array is at fault (mutual exclusion, empty,
arity, capability). `field` is a `DenoisePassField::name()` value and `issue` is a
`DenoisePassIssue::code()` value; both are deliberately stable, machine-readable strings so a
consumer asserts the same rejection without matching English prose.

## Algorithm ids

The fixtures use algorithms that exist in the registry **today** (`euler`, `dpmpp_2m`, `ddim`,
`normal`, `karras`, `simple`). Epic 20414's reference recipe names `rk6_7s` / `abnorsett_4m` /
`linear_quadratic` / `bong_tangent`, which two sibling stories are adding to the registries; the
recipe *shape* (10 steps at denoise 1.0, then 4 steps at denoise 0.2, each with its own
sampler+scheduler) is what `two_pass_reference_recipe.json` pins, and the ids can be swapped in
once they land without touching the harness. `invalid/unknown_sampler.json` deliberately uses an id
no registry will ever contain, so it stays a genuine negative regardless of what the siblings add.

## Regenerating `resolved/`

The per-pass seeds come from `gen_core::denoise_pass_seed(job_seed, index)` — pass 0 is the job seed
verbatim, later passes are a salted splitmix64 of it. They are checked in as literals on purpose: a
change to the derivation must show up as a fixture diff, because it silently breaks replay of every
persisted multi-pass render. Do not "fix" a seed mismatch by editing the fixture without deciding
that the replay break is intended.
