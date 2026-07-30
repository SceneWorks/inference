# SC-15804: `image_memory` → `memory_strategy` vocabulary rename

A source-compatibility break for consumers of `gen_core`. No behaviour, serialized field, provider
id, weight key, or wire format changes; every edit is an identifier, module path, or doc-comment.

## Why

Nothing in the SC-15448 ladder is image-specific. Rung 1 sheds a conditioning component, rung 2
bounds decoder scratch, rung 3 bounds attention, rung 4 bounds transformer residency — video and
audio have all four. The `Image` prefix recorded which lane adopted the contract first, not what the
contract covers, so it is dropped before any other lane adopts it and inherits a second contract.

## Why not the bare name `memory`

Five crates in this workspace already carry a `memory` module (`mlx-gen`, `mlx-gen-sam2`,
`mlx-gen-krea`, `mlx-gen-mage`, `candle-gen-sd3`), meaning two different things:

- `mlx_gen::memory` — the MLX budget interface (`SAFE_FRAC`, `safe_budget_gib`,
  `clamp_budget_to_cap`, `apply_memory_cap_env`, `MEMORY_CAP_ENV`), imported directly by MLX
  provider files, including the z-image adoption this contract already drives;
- `mlx-gen-sam2::memory` — SAM2's *model* memory bank (`MemoryEncoder`, `MemoryAttention`).

A third sits one crate away: `mlx_rs::memory` is the allocator itself (`clear_cache`,
`get_peak_memory`, `reset_peak_memory`).

A `gen_core::memory` would sit in the same `use` block as the first and read as the second.
`memory_strategy` collides with neither. The **types** stay bare — the existing bare `Memory*` names
belong to the SAM2/model-bank concepts above and never appear alongside the contract types.

## Mapping

| before | after |
| --- | --- |
| `gen_core::image_memory` | `gen_core::memory_strategy` |
| `gen_core_testkit::image_memory` | `gen_core_testkit::memory_strategy` |
| `IMAGE_MEMORY_CALIBRATION_ABI` | `MEMORY_CALIBRATION_ABI` |
| `ImageMemory*` (36 module items) | `Memory*` |
| `ImageMemoryRegistration` (`gen_core::registry`) | `MemoryRegistration` |
| `ProviderRegistryBuilder::register_image_memory` | `register_memory_strategy` |
| `Generator::image_memory_contract` | `memory_strategy_contract` |
| `Generator::begin_image_memory_request` | `begin_memory_strategy_request` |
| `Generator::image_memory_safety_check` | `memory_strategy_safety_check` |
| `ProviderRegistry::image_memory_contract` | `memory_strategy_contract` |
| provider-local `IMAGE_MEMORY_REGISTRATION` / `IMAGE_MEMORY_CALIBRATION_FINGERPRINT` | `MEMORY_REGISTRATION` / `MEMORY_CALIBRATION_FINGERPRINT` |
| `{Krea,Mage,ZImage}ImageMemoryScope` | `{Krea,Mage,ZImage}MemoryScope` |
| `mlx_gen_z_image::image_memory` | `mlx_gen_z_image::memory_strategy` |

`GenerationMemory` and `TransformerComponent` were already lane-neutral and are unchanged. The
calibration ABI value stays `1`: the handshake did not change, only what it is called.

## Consumer impact

SceneWorks is the only consumer. Its paired change renames
`sceneworks_worker::image_memory` → `sceneworks_worker::memory_strategy`, the
`sceneworks-image-memory-adapter` crate → `sceneworks-memory-adapter`, and bumps the calibration
harness version to `sceneworks-memory-v3`, which invalidates every calibration record captured under
the prior vocabulary. That is the stale-evidence gate working as designed; the promoted evidence
bundle was empty at the time of the rename, so re-capture cost was zero.
