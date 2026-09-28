# Image generator sweep fixes (sc-23187)

## Unicode prompts

The shared SDXL tokenizer ran BPE over Unicode characters directly. CLIP runs BPE
on UTF-8 bytes mapped through its 256-entry byte encoder. The installed SDXL q8
vocabulary maps the reported em dash to token 2005 after that encoding. Tests cover
ASCII, merged Unicode bytes, multilingual text, emoji, and special tokens.

## True-V2 conversion identity

The installed True-V2 conversion borrowed the previously shipped dense Klein base:
`SceneWorks/flux2-klein-9b-mlx@acf05e8d5103838baba6a5e32dc91d6997a56023/bf16`.
The historical dense layout passes the existing header and HF-blob validation.
Historical packed q4/q8 variants remain excluded.

The converter saves a HashMap of tensors, so safetensors header and payload ordering
can vary without changing tensor content. A whole-file checksum rejected this valid
conversion. Validation now hashes sorted tensor names, dtypes, rank, dimensions, and
exact payload bytes; it retains complete-layout validation and file identity checks.
Metadata and serialization order are not model content.

Independent conversion verification compared all 233 tensors against the original
checkpoint using the converter's rename table, QKV splits, and final adaLN half swap:

- Source: `wikeeyang/Flux2-Klein-9B-True-V2@9c9fe9880029a4e0c4af5ca7d86e83cdb83eea83/Flux2-Klein-9B-True-v2-bf16.safetensors`.
- Source SHA-256 (also the HF blob identity): `ad0da30a2efc1a0baf583628ac50062d033f2135b7717dac118b234eb55eabad`.
- Observed converted file SHA-256: `4c9a8fa1eaf0c2fa741351474bb5b95a47e7a5db79e8e847703b0dc8730aa5b9`.
- Canonical tensor SHA-256: `c7834f2e36e9de053384dd07bd409911bb1d66bd75ddc4f4c6b23cd846aec3db`.

The digest domain is `flux2-true-v2-tensors-v1` plus NUL. Each tensor contributes
UTF-8 name plus NUL, dtype plus NUL, little-endian u64 rank and dimensions, then its
payload. Tests vary header formatting and tensor order, and reject changed names,
dtypes, shapes, and bytes. The ignored installed-inventory test exercises the exact
component validation without loading GPU tensors.

## Flux2 Dev request staging

Dev/Edit retain phase loaders under either load policy and honor the request's
selected staging mode. The default still applies when no memory request exists.
Klein keeps its existing policy; control remains a separate lifecycle and only
advertises staging under Sequential. Deferred Dev/Edit loads do not borrow the eager
staging contract. Scope tests cover configuration and cancellation, and the shared
residency suite covers warm/staged/warm component release and repopulation.

These are source-level and focused installed-artifact checks. They do not claim a
new full-resolution render or a rebuilt desktop app.
