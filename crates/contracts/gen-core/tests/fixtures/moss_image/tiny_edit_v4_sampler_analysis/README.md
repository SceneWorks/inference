# Independent sampler composition evidence

Confidence: **high** for the frozen-source arithmetic, replay, independent CPU controls, and stated perturbation measurements. This supplies no new native/hardware pass and no real-weight acceptance. Parent v2 fixture, source code/math, expected outputs and receipts are unchanged.

`analysis.json` SHA256 `1e89a8c037569b988a89703c86894c6907da8c9097c92176d94b461663dcf6da`; `generate.py` SHA256 `b1079205b9d4676ef9d7bcd8c0fb931a3d4e9b43c603cb26467563156a6045e3`. Parent `../tiny_edit_v2/fixture.json` SHA256 remains `398b6b2260c15b670655700ed552d4ee935d493af5c9f2a72c85570cd7d121f1`. The generator checks its own frozen construction source and every base/adapter value, and asserts parent bytes unchanged afterward. Replaying each original source input, c/u prediction, guided velocity, Euler output and final returns exactly zero error, both 1- and 4-reference cases.

`fp32_sampler.json` SHA256 `905bef23f86f6bf08e116c716a03051ebbfa4f886d7b3dd9786e68e0d064cf56`; generator `generate_fp32.py` SHA256 `5610fda57b31d8fbcba6867c84e49d0c803f4e4fdb75d4be929d315c8e9f118b`; independent `fp32_receipt.json`. This casts **the exact already-represented BF16 base and adapter values** to FP32, asserts equality to the parent's represented values, and executes unchanged external source/model/edit/scheduler definitions. Every sampler tensor is FP32. No native output is read, no source arithmetic is replaced, and original BF16 expected values are not overwritten.

Both files contain `cases[].reference_count`; `correct.{final,steps,sigmas}`. Each step records `index,timestep,sigma,next_sigma,dt,input,conditional,unconditional,guided,output,manual_euler_max_abs`. Every tensor has explicit `shape,dtype,values`. Each semantic control includes the full corresponding step/final trace and final max absolute delta. BF16 analysis additionally retains 20 fixed source-output perturbation trials per case and per-evaluation perturbation magnitudes.

## Exact sampler contract

The actual author MOSS scheduler uses inference shift5/training shift1, two nearest-grid steps: sigmas `[1.0,0.024000000208616257,0.0]`, timecodes `[1000,24]`, dt approximately `[-.976,-.024]`. Network outputs are converted FP32; CFG is `u + 4*(c-u)`, then FP32 Euler `x_next = x + dt*guided`. Literal manual evaluation and unchanged Diffusers scheduler output have **zero discrepancy** on all correct-source steps in both BF16-network and FP32-network cases.

The sampler source used by retained v2 is installed Diffusers **0.36.0**, Torch2.9.1 CPU one thread (the separate VAE oracle is frozen Diffusers0.38.0). The imported FlowMatch Euler scheduler file SHA is `c2f47ed75801a7bd11f43aa793671986b7d382461b8d6e9d60786ca77fa61bca`; frozen official MOSS scheduler wrapper remains under pipeline SHA `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`. Complete original sampler replay proves this dependency has not drifted relative to retained v2 expected values.

For same input x, if conditional and unconditional network errors satisfy infinity-norm bounds Ec and Eu, the guided velocity error is bounded by `4*Ec + 3*Eu`, and one Euler step error by `abs(dt)*(4*Ec+3*Eu)` (plus the separately checked FP32 arithmetic error). A final-latent `2*BF16_epsilon*maxabs(final)` bound omits these coefficients and does **not** follow from a network-prediction 2-epsilon contract. The second network receives a changed x; a full trajectory bound additionally requires a justified state sensitivity bound. A BF16 quantized graph is discontinuous, so an unproven smooth Lipschitz constant cannot repair this argument.

## Source-only sensitivity and full-sampler negative controls

Twenty profiles were fixed independently of native output: shift each source BF16 network output 1 or 2 representable values via `nextafter`, in two opposing branch directions and eight SHA256 sign patterns; then re-evaluate unchanged source at each changed latent. This measures a **fixed empirical sensitivity envelope**, not a universal error guarantee or accepted tolerance.

| references | 1-ULP full final delta range | 2-ULP full final delta range | original final-only two-epsilon bound |
|---|---:|---:|---:|
|1|.08237505–.08706236|.16559362–.17075062|.06641600|
|4|.07946897–.08828092|.16250038–.17346859|.05908982|

Thus even these source-only one-ULP output perturbations exceed the final-only bound. This proves that aggregate bound does not compose from the primitive allowance; it does **not** establish that any particular native discrepancy is correct. The `.73` standalone prediction checks alone are insufficient: complete sampling evaluates t1 and t.024 at separate states.

| full sampler source graph | refs | wrong reference t0 | missing reference visibility | reversed reference order |
|---|---:|---:|---:|---:|
|BF16|1|.86025000|1.78162491|N/A|
|BF16|4|1.23768783|1.83131242|.16162503|
|FP32 represented BF16 weights|1|.81060743|1.79482973|N/A|
|FP32 represented BF16 weights|4|1.31328487|1.82670546|.10991740|

Wrong-time control prepares each step's reference cache using **that step's body time**, rather than a fixed `.73`. Missing-visibility and reversed-order controls alter only their named contract. Correct paths always execute unchanged source math. A blanket aggregate tolerance near `.17` could hide the four-reference order defect; it is explicitly **not recommended**. Likewise the source per-step same-input two-epsilon analytic sums `.26643946/.22337696` are not full-trajectory bounds and should not become aggregate tolerances.

## Recommended coherent composed gate

1. **Strict FP32 complete path:** run the exact represented-v2 base/adapter values in FP32 through production edit/reference-cache/sampler graph against `fp32_sampler.json`, 1 and4 references, using the existing tight `2e-5` primitive consistency threshold for per-step states/predictions and final. All source semantic mutants are well separated from that gate. This establishes model/control/sampling algorithm composition without BF16 network amplification obscuring defects. Synthetic FP32 injection remains test-only; production edit loading must still require BF16 adapters.
2. **BF16 networks at every fixed source evaluation input:** independently compare conditional and unconditional predictions at each `tiny_edit_v2` source step input (t1 and t.024), retaining unchanged existing prediction bounds, plus `.73` body/intermediate checks. Test masked positive/negative contexts, exact coordinates/cache shapes and cached/uncached equality. A passing `.73` test cannot substitute for these four network evaluations per reference case.
3. **Independent production CFG/Euler arithmetic:** retain native per-step x/c/u and validate actual production FP32 guided/output values against the source-defined `u+4*(c-u)` and `x+dt*guided`, at the agreed `1e-6` arithmetic threshold. Also replay production Euler on the fixed source c/u/x fixture to its source outputs; do not obtain expected values by calling the production helper under test. Assert FP32 dtype and exact schedule/time/branch use, not just a final error number. This checks integration algebra on the actual trajectory while fixed-input source tests check the model separately.
4. **Explicit semantic controls:** require wrong t0, missing visibility, reference order and cache-contract mutants to fail their associated fixed-input/body/coordinate or tight-FP32 tests with meaningful margin (e.g. >3 unchanged corresponding bounds where used). Do not rely solely on BF16 final-control separation. Existing fixed prediction order delta `.28515625`, exact ordinal coordinates and tight FP32 full-order delta `.10991740` remain discriminating even though an enlarged BF16 aggregate gate would not.
5. Report complete BF16 final drift against unchanged source final as a **diagnostic**, with its value and all primitive/composition results retained. Replace the mathematically invalid standalone aggregate assertion only with this explicit composed acceptance contract and review rationale; do not alter source goldens or silently widen a bound.

This proves required conditioning/cache/control algorithms and sampling arithmetic for the covered synthetic 1/4-reference geometries if all composed checks pass. It does not prove a uniform global BF16 trajectory-distance guarantee, real-weight image quality, complete published-size runtime behavior, peak-device use, or hardware performance. These remain separate requirements/acceptance work. It cannot excuse a failing fixed-input network, wrong FP32 operator, or semantic-control failure.
