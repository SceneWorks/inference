# Frozen MOSS edit CPU oracles

These are synthetic inputs, weights and outputs, not downloaded trained weights or vendored
GPL source. The independent Python generators execute definitions extracted unchanged from
externally supplied, hash-verified upstream source. `receipt.json` and `resize_receipt.json`
record source/tool versions, fixture hashes and deliberately incorrect controls.

Frozen ComfyUI source commit: f70d118a814e4a7e9b834597408af2f470eddd6c.
Frozen standalone project commit: 75f95426d7b55bb29cb8dbf123eb7c2257ccbf0d.
Official MOSS source and its SHA256 are recorded in fixture.json. The generator scripts belong
beside an external `official-sft/pipeline_moss_image.py` and
`ComfyUI-Moss-Image-Edit/moss_runtime/{model,weights,processing}.py` plus `nodes.py` tree;
copy them to that external staging directory and run Python with Torch CPU, Diffusers0.38
and Pillow12.3. They write `tiny_edit/` underneath that directory. Production never invokes Python.

Transformer fixture dimensions: dim16, ffn24, heads2, layers2, text12, freq8, channels4;
BF16 base and nonzero BF16 rank2 adapters (production rank64). One/four mixed-aspect
references, positive/negative masks, t0 modulation, exact reference coordinates, per-layer
K/V, cached/uncached predictions and two-step FP32 CFG4/Euler trace. Controls alter ref time,
remove reference visibility, reverse reference order, widen edit residual to FP32, or merge
adapters into BF16 base. These are validation controls, never accepted alternative semantics.

Resize fixtures compare complete RGB byte arrays with Pillow's separable 22-bit coefficient
Lanczos path, including upsampling to the /16 lattice, nonmultiple maximum edge and no resize.

The original metadata provenance recorded
chinoll/Moss-Image-Edit@e19b6ab0fb8fd010cb6f1c7f5b9bfe152252df11. That revision now
returns RevisionNotFound. The fixture bytes and historical numerical receipts remain unchanged.
`upstream_identity.json` records the accessible immutable revision
3be9c40037e1b89b17aa85ff6f8ee805872aa558 with the same weight SHA256 and byte-identical
adapter_config.json; it is a metadata correction, not a numerical rerun.
adapter_config.json and adapter_header.json prove all252
rank64 module configurations and504 BF16 A/B headers. Runtime also authenticates the complete
297382216-byte weights payload with SHA256
17a48b8c605dcffdedb92a2b887d18cf316505be90802717669fb6556098dcdf.
The publisher has not declared a weights license; GPL source licensing is not a weight license.
