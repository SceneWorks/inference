# gen-core-train

> Package: `sceneworks-gen-core-train` · library: `gen_core_train`

The **family-agnostic trainer helpers** of the backend-neutral `Trainer` contract: request
validation against a `TrainerDescriptor` (`validate_training_techniques`, `validate_edit_request`,
…), rectangular aspect-ratio assignment, cropping and microbatch scheduling, subject-mask decode,
auxiliary-loss band scheduling, resume identity, and the body/face loss geometry. The contract
itself — `Trainer`, `TrainingRequest`, `TrainingConfig` and the shapes they carry — stays in
`gen_core::train` (the registry's `TrainerRegistration` names it) and is re-exported here, so a
trainer reaches the whole surface from `gen_core_train::…`.

Split out of [`gen-core`](../gen-core/README.md) in sc-27316. It depends on gen-core (gen-core
never depends on it) and, like gen-core, on no tensor library. It holds gen-core's former `train`
helper modules; with them goes the subject-mask `image` decode.

## License

Apache-2.0.
