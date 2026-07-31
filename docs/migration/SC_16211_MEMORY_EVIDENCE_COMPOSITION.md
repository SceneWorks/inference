# SC-16211 memory-evidence composition identity

Memory evidence now keys the exact provider-intersected strategy composition that was active during
measurement. `MemoryProviderContract::engaged_composition` walks the canonical rung order and routes
each decision through `MemoryProviderContract::engages`, so additional provider prerequisite edges
are part of the identity without giving providers a second engagement policy.

`MemoryEvidenceKey::engaged_composition` is a non-empty, strictly ordered set. Before any evidence
can authorize a fit, `MemoryEvidence::optimized_eligibility` compares that measured set with the
composition derived from the current provider contract. Any difference returns
`MemoryEvidenceVerdict::CompositionMismatch`; it never falls through to a predicted or guessed fit.

This is an additive Rust source contract but a required consumer migration: every `MemoryEvidenceKey`
constructor must supply the composition from its measurement record. SceneWorks owns the serialized
calibration-schema bump and the migration of promoted evidence and manifest rows. A non-default
prerequisite-edge mutation test proves that evidence which was eligible before the contract change
becomes ineligible afterward.
