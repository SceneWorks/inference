# sc-25328 corrected prototype readout

**Decision: NO-GO (completed corrected experiment).** The corrected experiment does not meet E11: neither the other-model generated clip nor live-action clip shows useful faithful improvement over bicubic, and both exhibit material preservation regressions. H3 offers at most one modest/borderline improvement. The required two of three useful cases and preservation floor are unmet.

This candidate uses published F32 VAE weights and the pinned decoder's already-blended full-width neighbour strips, with Ref2VA BF16 and the learned upscaler FP16. Ordinary generation remains unchanged. The earlier BF16 candidate and its NO-GO are superseded; its original reports, actual parity failures and hashes are retained in the history.

All three fixed fixtures contain 39 frames at 512x288 and 24 fps, enlarged to 1024x576. Original media provenance and normalization remain explicit in readout.json. Source/audio, bicubic and SeedVR2 baselines were reused only after hash verification; this does not claim arbitrary source timing support. All 39 output PTS and decoded source soundtracks pass.

Nine independently derived real source/guide captures and three composed learned-network comparisons pass the unchanged predeclared max-absolute 0.08 AND peak-relative 0.015 limits. Each expected VAE tensor comes directly from fixed RGB through the unmodified pinned Comfy VAE; no native latent is used to produce its source or guide. All nine real capture value mutations are rejected numerically with valid RGB/export hashes and fresh changed-tensor hashes. RGB/export hash changes also fail. Per-case errors, source/weight/reference/native hashes and exact exporter provenance are bound in readout.json. Full real RefDiT parity is outside this claim.

The isolated H3 decoder error was 0.220713 in RGB under original-neighbour seams, falling to 0.000078708 after the experimental stitch correction; guide latent error fell from 0.619808 to 0.002047. The tiny fixture executes actual pinned Comfy spatial methods and rejects the old original-tail stitch.

Measured on RTX PRO 6000 Blackwell Max-Q, 97887 MiB, driver 596.36; Candle/CUDA 12.9 and supported MSVC 14.44. Peaks are process RSS/device-used memory sampled every 200 ms. Cache/temperature state was not controlled into cold/warm distributions.

| Case | Run | Wall seconds | Device MiB | Host RSS bytes |
|---|---|---:|---:|---:|
| h3 | guided | 154.53 | 73255 | 5920718848 |
| h3 | unguided | 90.03 | 72935 | 5781385216 |
| h3 | latent-only | 38.66 | 18629 | 5363359744 |
| h3 | SeedVR2 (reused) | 82.02 | 58691 | 7131844608 |
| other-model | guided | 115.08 | 73255 | 5920907264 |
| other-model | unguided | 90.81 | 72935 | 5780979712 |
| other-model | latent-only | 38.73 | 18629 | 5482065920 |
| other-model | SeedVR2 (reused) | 83.73 | 58691 | 7131369472 |
| live-action | guided | 116.92 | 73243 | 6078570496 |
| live-action | unguided | 90.91 | 72919 | 5779787776 |
| live-action | latent-only | 39.75 | 18613 | 5309632512 |
| live-action | SeedVR2 (reused) | 83.50 | 58531 | 7131770880 |

Guided denoise 0.1 selects shifted sigma 0.5714286, one Euler step and LoRA strength 1 with 312 targets. Controls disable the guide or use denoise zero for learned-only enlargement. Internal audio is frozen clean-zero; delivered audio is the unchanged fixture soundtrack.

**h3: Modest/borderline sharpening of arch, ship silhouette and console edges.** Composition and silhouette broadly retained across frames 0/19/38; small console glyph/light patterns and ship surfaces are redrawn. Guided and unguided look broadly similar; learned-only enlargement explains much of the limited sharpening. This does not establish a second useful case. SeedVR2: Cleaner edges and better small-detail retention than H3 refinement; no claim of original high-resolution ground truth

**other-model: No useful faithful improvement.** Watermark becomes gibberish at beginning/middle/tail; floor, light beams and crowd develop woven texture and smear; robot face/surfaces lose source detail while pose remains broadly aligned. Latent-only already loses watermark/robot detail and smears textures; guided and unguided retain similar damage. This identifies a recipe limitation after validated source/guide stages, separate from the superseded VAE precision/stitch defect. SeedVR2: Retains more legible watermark and robot/scene outlines; some texture changes remain

**live-action: No useful faithful improvement.** Moving birds collapse into angular ghost trails/double outlines, water/masts/boats gain woven or line texture, and standing birds soften/redraw across frames 0/19/38. Unguided and latent-only controls show related outline and texture damage; guided refinement does not restore faithful motion/detail. SeedVR2: Preserves bird outlines and marina structure more clearly; texture changes remain

Implementation agent and coordinator inspected beginning/middle/tail frames 0/19/38, six-column controls and full-size pairs. Change metrics over all 39 frames describe deviations from bicubic, which is a preservation baseline rather than high-resolution truth. They are not automatic quality scores.

Affected checks pass: 376 H3 unit + 205 integration tests (20 hardware tests ignored); eight acceptance tooling tests; CUDA compile-only, all-target Clippy and rustdoc with warnings denied; affected-package formatting, workspace checks for 105 members, clock ratchet with 279 existing flags, and diff checks. Earlier broad first-push lane evidence remains recorded for unchanged surfaces, including explicit unrelated Windows/Linux platform limitations. No broad campaign or second adversarial review was repeated.

Viewable current evidence: evidence-f32-final/<case>/{source,bicubic,guided,unguided,latent-only,seedvr2}.mp4, review-begin-middle-tail.png and review-full-frame{0,19,38}.png. Actual VAE parity is in evidence-f32-final/vae-parity; independent fixed-RGB exports and exact executed exporter are in evidence-vae-reference-f32. The final release binary is retained with all nine matching receipt hashes.

S2-S9 remain unstarted pending plan revision. No product route/capability, generation-validator, cross-provider dependency, inference pin or terminal measurement gate changes. Existing notices and user-provided written LoRA authorization are retained; no model weights/source media are redistributed.
