# Strengthened independent edit oracle v2

This version addresses an inadequate negative-control margin in the original synthetic fixture. Original `/upstream/tiny_edit/fixture.json` and its receipt/generator are untouched; SHA256 remains `4a326adbd707bf11ce45f0e0d62d315690e295e067bac12f3c486e932f51475f`.

New `fixture.json` SHA256 `398b6b2260c15b670655700ed552d4ee935d493af5c9f2a72c85570cd7d121f1`; generator SHA256 `d464fcac7bf3b10989ec43f71f77a6f38340655ad59f618adc4a2745c0ed61f7`. `receipt.json` binds source/generator/output; `validation.json` verifies original preservation, unchanged source math, exact cached/uncached equality and diagnostic identity.

Synthetic changes only: non-norm backbone weight elements use decorrelated SHA256 of `name:index`, first4 bytes little-endian divided by2^31 minus1, scaled by0.8/sqrt(fanin); all norm weights, including final_norm_video, use1+0.03*sin from the original recipe. Before BF16 conversion, time MLP weight gain2, shared modulation weight gain4/bias gain2, head weight gain2, final projection weight gain4. Reference amplitude1.2 rather than0.6. Adapters remain the original nonzero BF16 rank2/alpha2 recipe. These choices remove accidental low-rank correlation in the original sinusoidal backbone and avoid tiny final normalization weights suppressing reference effects. Source transformer/wrapper/LoRA AST math is unchanged and hash-verified; no native output informed expected values.

Outside this repository, the authoring receipt directory preserves two rejected synthetic tuning attempts as `tiny_edit_v2_attempt_sine_gains/` and `tiny_edit_v2_attempt_xorshift_gains/`, each with generator, fixture and receipt. They still had controls close to BF16 rounding; final v2 uses the decorrelated weight recipe above. Their generators retain their original output-directory text, so do not rerun them into the finalized directory.

| case | wrong reference time max abs | missing reference visibility max abs | reversed order max abs |
| --- | --- | --- | --- |
| one reference |0.453125|0.951171875|not applicable|
| four references |0.703125|1.541015625|0.28515625|

All controls exceed0.01. Both correct cached/uncached outputs remain bit-identical on CPU. This supports discriminating native tests; it does not authorize arbitrary loose tolerances or establish native parity.

`time_probes` retains normalized timesteps0,.73,.31,1,.005. Each has `time1`, `silu`, `time2`, `modulation_flat`, `head_flat`, the corresponding `_input` objects, `time_output_fp32`, reshaped `modulation` and `head`. Values are **actual forward hooks on unchanged source modules**. `sinusoid` aliases actual first-linear input and is BF16 after the source cast; it is not the pre-cast FP32 sinusoid. Tensor shape/dtype/row-major values are recorded explicitly.

`bias-rounding-diagnostic.json` retains actual-source outputs alongside deliberately split BF16 `matmul` then bias outputs. Regenerator is `../generate_edit_bias_diagnostic.py`; stage it externally beside the source trees before use. For v2 t0, the split probe differs by0.00390625(time1),0.001953125(time2),0.00390625(modulation),0.001953125(head). These are arithmetic-discrimination measurements, not a claim that this causes the native mismatch. The correct expected graph remains the unchanged frozen module output.

Original-weight localization fixture is sibling `tiny_edit_v1_time_probe/fixture.json`, SHA256 `c2af600c1c966ced4f8485081fc469d4b206a91205db49a7f5400e66204c5985`, with its own generator/receipt/validation/rounding diagnostic. Its weights, adapters, inputs, complete correct/control cases and projection probe were checked exactly equal to original v1. Only `time_probes` were added. At original t0 the split-bias discrepancies are0.00390625(time1),0.0009765625(time2),0.000244140625(shared modulation/head).

All refinement executed Torch2.9.1 CPU, one thread, no GPU initialization, builds, trained weights, native calls or repository edits. Original source authorities remain official SFT pipeline `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`, Comfy edit wrapper `f0f5d3011c1f689547d290231a8a50a1c4d16f58c561556b53db60a0edbe9885`, LoRA `692f9df723323a95cdb24bbf612c7b6d2a98709cefcbee479e2d7dc57aab0705`.

Native comparisons retain exact adapter projections and coordinates, and bit-exact cached/uncached and canceled-run reentry checks. Cross-backend BF16 graph comparisons allow `max(abs(expected))/64 + 1/1024`, covering two BF16 rounding units at the tensor maximum plus a small cancellation floor. Every incorrect time, visibility and order prediction must differ from the native prediction by more than three times that same bound. Time stage inputs are injected independently from source hooks to localize backend rounding; the full time chain is also compared.
