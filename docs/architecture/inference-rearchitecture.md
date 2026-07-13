# Inference Rearchitecture Rationale

> **Status:** Accepted and implemented
> **Decision date:** 2026-07-13
> **Scope:** Repository consolidation, dependency ownership, and generative-media
> provider composition

## Decision summary

SceneWorks inference is maintained as one history-preserving repository with one
Cargo workspace, one lockfile, and explicit ownership boundaries for contracts,
engines, provider families, and platform catalogs.

Generative-media providers are composed explicitly. Provider crates publish named
registration constants, each family exposes a `register_providers` function, and
the MLX and Candle catalog crates enumerate the families shipped by their
platform. Applications construct a `ProviderRegistry` from one of those platform
catalogs and route discovery and loading through that value.

This replaces the former media `inventory` registry, global loading facade, and
force-link compatibility anchors. The LLM engines also expose explicit catalogs,
but their inventory-based compatibility path is intentionally outside this
decision and remains until its consumers are migrated in a separate change.

## Context

The inference stack had grown across five repositories:

- `core-llm`, containing backend-neutral LLM contracts;
- `mlx-llm` and `candle-llm`, containing the two LLM engines;
- `mlx-gen` and `candle-gen`, containing the media contract, engines, provider
  families, tests, and examples.

Those repositories were separate deployment units in name but one change unit in
practice. A contract change in `gen-core` or `core-llm` routinely required a
sequence of downstream SHA updates, lockfile changes, and compatibility edits.
The imported history contains many explicit “re-pin,” “de-skew,” and “lockstep”
commits documenting this coordination cost. At any moment, a green source
repository could still be incompatible with the exact revisions selected by a
consumer or sibling repository.

The old media registry created a second, independent source of uncertainty.
Provider crates submitted registrations into `inventory`; the effective catalog
was therefore determined by which crates and object code happened to participate
in the final link. Some Candle paths required `force_link` anchors to make that
participation reliable. Tests and examples could observe different catalogs
based on their dependency graph, and the supported platform surface could not be
read from one composition root. A global `load(...)` API concealed that implicit
composition from callers.

Neither problem was primarily about the number of repositories or the syntax of
the registration macros. The underlying issue was hidden integration state:

- the source revision set was distributed across Git dependencies and lockfiles;
- the runtime provider set was distributed across link-time submissions;
- compatibility was proven repository by repository rather than for the shipped
  graph;
- ownership of “which providers does this platform ship?” was not represented in
  ordinary source code.

## Goals

The rearchitecture was intended to:

1. Make a contract change and its backend adaptations atomic and reviewable.
2. Give a checkout exactly one internal dependency graph and one lockfile.
3. Preserve backend-neutral contracts and independent MLX/Candle implementations.
4. Make each shipped media provider surface explicit, deterministic, inspectable,
   and testable without loading model weights.
5. Detect duplicate IDs and malformed descriptors at registry construction or in
   weights-free conformance tests.
6. Let tests construct minimal registries without inheriting unrelated providers
   from the test binary's link graph.
7. Preserve source history, provider IDs, package names, serialized contracts,
   and platform-specific build constraints throughout the migration.

## Non-goals

The change was not intended to:

- collapse 69 packages into a single crate;
- merge MLX and Candle implementations behind a lowest-common-denominator engine;
- introduce a stable dynamic-plugin ABI or load third-party code at runtime;
- move every provider into a new model-first directory layout;
- upgrade backend revisions, tokenizer versions, or public request schemas as a
  side effect of consolidation;
- make `--workspace --all-features` a universal cross-platform build mode;
- remove the LLM inventory registry before its own consumers and compatibility
  requirements have been evaluated.

The monorepo is a shared change and validation boundary, not a monolith.

## Chosen architecture

The dependency and composition direction is:

```text
backend-neutral contracts
          |
          v
MLX/Candle engines and shared runtime primitives
          |
          v
provider-family crates (named registrations + register_providers)
          |
          v
MLX/Candle platform catalog (the shipped surface)
          |
          v
application composition root (one immutable ProviderRegistry)
```

### One repository, one resolved graph

The source repositories were not rewritten or force-pushed. Their histories were
filtered into namespaced imports with committed old-to-new commit maps and tree
equivalence checks, then relocated beneath ownership paths:

```text
crates/contracts/  backend-neutral contracts and conformance suites
crates/llm/        MLX and Candle LLM engines
crates/media/      MLX and Candle media engines and provider families
```

Internal Git dependencies became workspace paths. The root owns the member list,
lockfile, Rust toolchain, backend revision pins, Cargo configuration, CI lane
selection, supply-chain policy, and release artifacts. Package boundaries remain
because they still express useful compile, ownership, and provider-family seams.

This makes the committed workspace the integration set. A contract and all of
its in-repository consumers are reviewed and tested at the same revision rather
than being made compatible through a later series of downstream re-pins.

### Provider-owned declarations

The media contract exposes small registration records containing a descriptor
function and a load function. Registration macros define named constants; they
do not mutate or submit into a global registry. A provider family owns the list
of its constants and exposes a normal Rust function:

```rust,ignore
pub fn register_providers(
    registry: ProviderRegistryBuilder,
) -> ProviderRegistryBuilder {
    registry.register_generator(MY_MODEL)
}
```

This keeps implementation knowledge with the provider while making inclusion a
normal dependency and function-call relationship.

### Platform-owned composition

`mlx-gen-catalog` and `candle-gen-catalog` are the composition roots for their
respective media platforms. Each explicitly calls the family registration
functions in stable order and exposes `provider_registry()`.

The catalog crates own only selection and ordering. They do not own model
implementations, duplicate descriptors, or infer membership by scanning the
workspace. A provider crate existing in the repository does not automatically
mean it ships in a platform bundle; catalog inclusion is a deliberate decision.

### Immutable, value-scoped registries

`ProviderRegistryBuilder::build()` rejects duplicate IDs per provider kind and
produces an immutable registry. Callers can iterate descriptors and load
generators, trainers, captioners, image embedders, text embedders, and transforms
through that value.

Production applications normally construct one complete platform registry.
Tests may construct the same complete registry or a deliberately small registry.
The set under test no longer changes because an unrelated crate was added to the
test binary.

### Named runtime bundles

Media composition alone is not the supported product boundary. `runtime-macos`,
`runtime-cuda`, and `runtime-cpu` each assemble one media registry, one LLM
registry, and one snapshot-preparer registry. `runtime-catalog` validates that
every descriptor and preparer belongs to the bundle's declared backend, then
exposes a stable machine-readable snapshot.

The LLM engines export ordinary builder functions for provider and preparer
registrations. This gives model-first loading and snapshot preparation the same
explicit ownership model as media generation. The older process-global LLM APIs
remain only as temporary cutover adapters.

The bundle names are also CI and release profile names. MLX, CUDA, and CPU are
not additive features of one universal build, so `--workspace --all-features`
is intentionally not a supported validation strategy.

### Executable catalog contracts

Both platform catalog crates pin their complete, ordered ID surfaces in tests.
Those tests also assert the expected backend and run the weights-free descriptor
conformance sweep. Provider-family and consumer tests resolve through explicit
registries, so the test path matches the production composition model.

Stable ordering is retained because descriptor iteration is used for diagnostics,
capability reporting, and reproducible tests. It is not an accidental promise
made by linker iteration.

## Why the alternatives were rejected

### Keep the five repositories and automate revision bumps

Bots could reduce the manual work of re-pinning, but they would not make a
cross-repository change atomic. Intermediate revision sets, duplicated lockfiles,
and delayed integration failures would remain. The repositories already behaved
as one release graph, so preserving separate integration boundaries had little
architectural value.

### Merge everything into one crate

This would produce a simpler-looking filesystem at the cost of much worse
ownership, platform isolation, compile behavior, and provider testability. The
problem was not package boundaries; it was dependency and composition state that
lived outside a single authoritative graph.

### Retain media inventory and document the force-link rules

That would keep provider membership dependent on final-link participation and
would leave tests unable to state their catalog precisely. More documentation
could explain the implicit behavior but could not make it locally inspectable or
deterministic. Force-link anchors were symptoms of the model, not isolated bugs.

### Generate a catalog in `build.rs`

Filesystem scanning or code generation would replace linker magic with build
magic and introduce a second source of truth. Explicit Rust composition is small,
type-checked, searchable, and reviewable in an ordinary diff.

### Use runtime dynamic plugins

Dynamic loading would solve a different requirement. It would add ABI stability,
version negotiation, packaging, signing, and failure-mode complexity. SceneWorks
currently ships statically selected first-party providers, so compile-time
composition is the appropriate boundary.

## Accepted tradeoffs

The new design intentionally adds a small amount of wiring:

- adding a provider requires exporting its registration and including its family
  in the appropriate platform catalog;
- applications must construct or receive a registry instead of calling a global
  loader;
- catalog crates depend on all providers in the platform bundle and therefore
  represent a relatively broad compile target;
- ordered catalog tests require an intentional update when the shipped surface
  changes;
- a single repository is larger and requires dependency-aware CI rather than
  running every platform for every documentation change.

These costs are features of explicit ownership. The extra catalog edit is the
review point where platform inclusion becomes visible. Dependency-aware CI and
platform-specific lanes address repository scale without reintroducing separate
source-of-truth graphs.

## Migration strategy

The registry change was staged so compatibility remained available until every
consumer had an explicit replacement:

1. Define and validate explicit registration and family-catalog primitives.
2. Migrate asymmetric providers whose MLX and Candle surfaces are not identical.
3. Compose complete MLX and Candle platform catalogs with exact-surface tests.
4. Cut examples, conformance suites, provider tests, and other consumers over to
   explicit registries.
5. Remove media inventory submissions, global loaders, force-link anchors, and
   unused dependency edges only after the explicit path was complete.

The implementation begins with the explicit media registry in `4f81dbcf`,
composes the platform catalogs in `5401a056`, and completes compatibility removal
in `62069989`. Keeping those stages separate made omissions observable before the
implicit fallback disappeared.

## Invariants for future changes

Future inference work should preserve these rules:

- backend-neutral contracts do not depend on MLX or Candle tensor types;
- internal SceneWorks dependencies use workspace paths, not repository SHA pins;
- media provider registration is explicit—do not add `inventory` submissions or
  `force_link` anchors as a shortcut;
- every shipped family is named in its platform catalog;
- every supported product consumes one named runtime bundle rather than assembling
  backend crates;
- LLM providers and snapshot preparers are added through explicit builders, not
  new linker side effects;
- duplicate provider IDs fail registry construction;
- complete platform surfaces and descriptor conformance remain weights-free test
  gates;
- consumers load through a registry value, not a process-global media facade;
- differences between MLX and Candle catalogs are allowed when they represent
  real implementation differences and are pinned explicitly in tests;
- the LLM inventory compatibility adapter is temporary cutover infrastructure,
  not a supported composition boundary or precedent for new registration.

## Validation and outcome

The consolidated workspace and explicit catalogs were validated at the final
migration revision with:

- strict Clippy checks for the contract/testkit, both platform catalogs, and all
  MLX/Candle media provider library and test targets;
- exact platform-catalog surface and descriptor-conformance tests;
- `cargo test --locked --workspace --lib --tests`;
- source audits confirming no legacy global media loaders, anonymous media
  registrations, or force-link paths for the media registry remain; the LLM
  inventory submissions are tracked compatibility adapters pending consumer
  cutover;
- a clean lockfile diff removing the obsolete media `inventory` dependency
  edges.

The resulting architecture does not promise that all provider implementations
are identical or that platform work is cheap. It makes the shipped graph and the
remaining differences explicit, which is the prerequisite for maintaining them
safely.

## Revisit criteria

Revisit this decision if SceneWorks needs third-party providers to be installed
without recompiling, if platform catalogs become independently versioned products,
or if workspace scale can no longer be handled by dependency-aware CI. Any such
change should preserve an inspectable composition root, deterministic catalog
membership, duplicate-ID validation, and weights-free conformance checks.

Remove the LLM inventory adapter only after SceneWorks and ChatWorks consume the
explicit registries from an immutable runtime release and the documented rollback
point no longer depends on link-time discovery.
