# Pre-migration fact audit

Inference revision `2b09938e33370915d4f853389961709b8bbf8bf9` (I); SceneWorks
revision `5a28d413dd9f579cc9607d44f252a045d025bd8e` (S), which pins that engine.
The 26 rows follow the published architecture plan §10.2 in order.

To replay a row run `git grep -n -E -e 'PATTERN' REV -- crates apps config` in the
specified repository, then read the identified scope. Each witness below writes a
value, relation or ID set in production. A use of a constant, generated capability
dump, checked expectation, product menu or upgrader alias does not add an encoding.
The baseline names the compared routes per row. The listed locations establish
lower bounds; raw grep hit counts cannot establish single ownership across the
catalog. Platform-specific differences and product defaults have independent semantics.

Path prefixes below expand as follows:

| Prefix | Directory |
|---|---|
| M | I `crates/media/mlx-gen/mlx-gen-flux/src` |
| C | I `crates/media/candle-gen/candle-gen-flux/src` |
| G | I `crates/contracts/gen-core/src` |
| SD | I `crates/media/mlx-gen/mlx-gen-sdxl/src` |
| F2 | I `crates/media/mlx-gen/mlx-gen-flux2/src` |
| W | S `crates/sceneworks-worker/src` |
| P | S `crates/sceneworks-core/src` |

| Fact and measured scope | Source encoding(s) at baseline | Grep pattern |
|---|---|---|
| 1. Default steps, FLUX.1 dev/schnell | M/config.rs:51–52 and C/lib.rs:181–182 each encode 4/25. W/engines.rs:150,159 requests 4/28 at the worker boundary. | `default_steps\|Schnell => 4\|Dev => 25` |
| 2. Default guidance, FLUX.1 dev | M/config.rs:17, C/lib.rs:194 and W/engines.rs:160 each encode 3.5. | `DEFAULT_GUIDANCE\|default_guidance\|3\.5` |
| 3. Supported samplers, MLX base | M/config.rs:21,26 adds flow_match and hyper to the curated set. M/config.rs:102 scopes hyper to dev. The curated-list reader in C/lib.rs:384 is derived. | `HYPER_SAMPLER\|DEFAULT_SAMPLER\|curated_sampler_names` |
| 4. Supported schedulers, MLX base | M/config.rs:111 extends the curated set with linear. Candle reads the curated set at C/lib.rs:385. | `curated_scheduler_names\|push\("linear"\)` |
| 5. Supported modes, app MOSS-Image | P/moss_image.rs:231 explicitly permits text_to_image and edit_image. | `text_to_image.*edit_image` |
| 6. Img2img support, MLX SDXL | SD/model.rs:204 advertises Reference and the init match at :1742ff distinguishes true source-image initialization from IP conditioning. FLUX.1's M/config.rs:89 Reference is an IP input. | `ConditioningKind::Reference\|SdxlInitLatents::Img2Img` |
| 7. Inpaint support, MLX SDXL | SD/model.rs:205 advertises Mask; the matched inpaint initialization at :1696ff uses source pixels and the mask. | `ConditioningKind::Mask\|SdxlInitLatents::Inpaint` |
| 8. Control modes, MLX FLUX.1 strict | M/model_control.rs:395–400 encodes pose, canny and depth in the Fun union. | `fn accepted_control_kinds\|ControlKind::(Pose\|Canny\|Depth)` |
| 9. Reference/multi-reference, MLX FLUX.1 IP | M/config.rs:89 declares Reference; M/model.rs:1034–1053 admits exactly one Reference and refuses mixed/multi. | `ConditioningKind::Reference\|only a single Reference` |
| 10. Pose, MLX FLUX.1 strict | M/model_control.rs:397 includes Pose in the finite accepted set. Reused here from row 8. | `ControlKind::Pose` |
| 11. LoRA/LoKr compatibility, FLUX.1 base | M/config.rs:91–92 and C/lib.rs:378–379 both advertise both formats. | `supports_lo(ra\|kr): true` |
| 12. Quant tiers per component, MLX FLUX.1 | M/config.rs:90 declares Q4/Q8. M/convert.rs:73–96 sends transformer, text_encoder, text_encoder_2 and vae through their respective packing predicates at those tiers. | `supported_quants:\|quantize_component\(` |
| 13. Max steps, shared request floor | G/generator/capabilities.rs:479 encodes the 100,000 sanity ceiling. Individual routes may tighten it. | `const MAX_STEPS` |
| 14. Width/height bounds and multiples, FLUX.1 base | M/config.rs:114–115 and C/lib.rs:386–387 both declare 256–2048. M/model.rs:37 and C/lib.rs:147 both declare stride 16. | `min_size: 256\|max_size: 2048\|SIZE_MULTIPLE: u32 = 16` |
| 15. Frame and fps limits, H3 | G/h3_memory.rs:91–95 encodes 24 FPS and 124 through 345 frames. | `const (FPS\|MIN_FRAMES\|MAX_FRAMES):` |
| 16. Duration limit, MOSS-SFX | I crates/audio/candle-audio-moss-sfx/src/model.rs:79 caps duration at 30 seconds. | `const MAX_DURATION_SECS:` |
| 17. Hires.fix eligibility, app MOSS-Image | P/moss_image.rs:289–294 unconditionally rejects enabled hiresFix for this model. | `MOSS-Image does not support hiresFix` |
| 18. Prompt enhancement eligibility/limits, MLX FLUX.2 | F2/config.rs:326,439 scopes enhancement to dev/dev-edit. F2/caption_upsample.rs:60,65 encodes default 512 and cap 2048 new tokens. | `supports_prompt_enhancement:\|const (DEFAULT_MAX_NEW_TOKENS\|MAX_NEW_TOKENS_CAP)` |
| 19. Negative prompt, FLUX.1 | C/lib.rs:370 declares false. M/model.rs:1034–1059 allows the separate dev reference-CFG case. | `supports_negative_prompt:\|let ref_cfg_ok` |
| 20. Guidance kinds, shared methods | G/guidance.rs:57–60 and :68–71 each spell cfg, cfg_rescale, apg and cfg_pp, once in each wire direction. FLUX.1 is also gated by variant, M/config.rs:64 and C/lib.rs:189. | `"cfg_rescale"\|fn supports_guidance` |
| 21. Preview, FLUX.1 base | M/config.rs:123 and C/lib.rs:395 explicitly advertise preview on both native backends. | `supports_preview: true` |
| 22. Decoder options, Candle FLUX.1 | C/pipeline.rs:77 declares the flux PiD backbone ID; C/lib.rs:319–327 refuses PiD with bounded native-VAE optimizations. | `PID_BACKBONE:\|do not support PiD decode` |
| 23. Memory-rung support, Candle FLUX.1 | C/memory_strategy.rs:408–432 qualifies bounded transformer residency on a streamable source and declares distinct decode/attention/block parameters; :460–469 declares staged residency prerequisites. | `MemoryStrategy::(BoundedTransformerResidency\|StagedResidency)` |
| 24. Engine ID to model ID, FLUX.1 base | W/engines.rs:146–147,155–156 maps flux_schnell/dev to flux1_schnell/dev. Both engine crates define their registry IDs in M/config.rs:9–10 and C/lib.rs:136–138. | `"flux_schnell"\|"flux_dev"\|engine_id: "flux1_` |
| 25. Trainer support and techniques, MLX FLUX.3 | I crates/media/mlx-gen/mlx-gen-flux3/src/training/trainer.rs:58–63 declares LoRA, no LoKr/control/full finetune and TrainingTechniques::NONE. | `supports_full_finetune:\|techniques: TrainingTechniques::` |
| 26. Per-image seed relation, MLX FLUX.1 | M/model.rs:866 and M/model_control.rs:308 both encode base seed plus output index with u64 wrapping. | `base_seed\.wrapping_add\(i as u64\)` |

The vertical bars escaped for Markdown table rendering above are ERE alternation
operators when passed to `git grep -E`. The proof of a duplicate is in the scoped
source declarations. For example, changing the MLX or Candle dev default at only
one of the two locations in row 1 would change behavior when a native request
omits steps. The distinct SceneWorks default already overrides it for app jobs.
