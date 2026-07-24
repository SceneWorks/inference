# Mage-Flow config fixtures

Verbatim, unmodified copies of three component configs published by **`microsoft/Mage-Flow`**
(revision `9f46d09dce8a6211a5aaf157cc99754ac402a2fc`), committed so `config_conformance.rs` can
check every constant in `mlx_gen_mage::config` against the real file instead of against itself.

| file | upstream path | SHA-256 |
| --- | --- | --- |
| `transformer_config.json` | `transformer/config.json` | `8493c3b2722738c2a824ac82b1fd9c89fefb4e354fc88363207193db7fe702de` |
| `vae_config.json` | `vae/config.json` | `abd124d603d6c6a03e9d0f2aa6d113b8c4afda0738400bdf2f99240aeaeaff76` |
| `scheduler_config.json` | `scheduler/scheduler_config.json` | `438fd8bcf254740e5d3f3e9800bbd9c571e342ab87885388d1505b7531c69c02` |

All **six** Mage-Flow repositories — `Mage-Flow`, `-Base`, `-Turbo`, `-Edit`, `-Edit-Base`,
`-Edit-Turbo` — ship these three files byte-identically (verified by hashing every local snapshot),
so one copy pins the family. Only the transformer *weights* and the model cards' default
`steps`/`cfg` differ between variants.

**Licence:** the Mage-Flow model repositories and `github.com/microsoft/Mage` are MIT, which permits
redistribution. These are metadata files, not weights; the frozen reference implementation is
vendored separately under `crates/media/mlx-gen/_vendor/mage_flow/` with its own `LICENSE`,
per-file checksums and re-vendor policy in `../../../_vendor/VENDORED.md`. The weight-licence
manifest row (`release/model-weight-licenses.json`) lands with the catalog surface in sc-14047.

**Re-fetching:** copy the files out of a fresh snapshot and re-run
`cargo test -p mlx-gen-mage --test config_conformance`. A SHA-256 mismatch means the published
config changed — treat every constant transcribed from it as suspect and re-read
`_vendor/MAGE_FLOW_GAPS.md` before touching `src/config.rs`.
