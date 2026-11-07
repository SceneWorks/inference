//! Weight-free bounded conditional execution fixture for SenseNova interleave.
//!
//! This contract prices every possible rollout, image pass, and feedback before a provider
//! loads weights. MLX/Candle production interleave remains on its established path until its
//! separately tracked migration; the executor here is an intentionally tensor-free seam.

use crate::{CancelFlag, Error, Progress, Result};
use serde::{Deserialize, Serialize};

const CACHE_COUNT: usize = 3;
const IMAGE_CHANNELS: u64 = 3;
const F32_BYTES: u64 = 4;
const IMAGE_BUFFERS: u64 = 4;
const GUIDANCE_BRANCHES: u64 = 3;

fn overflow() -> Error {
    Error::Unsupported("conditional plan arithmetic overflow".into())
}

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(overflow)
}

fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(overflow)
}

fn refused(reason: &str) -> Error {
    Error::Unsupported(format!("conditional plan: {reason}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheKind {
    Condition,
    TextUncondition,
    ImageUncondition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Boundary {
    Prefill,
    Terminal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheLifetime {
    pub kind: CacheKind,
    pub starts: Boundary,
    pub ends: Boundary,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageTemplate {
    TextRollout {
        max_calls: u32,
        max_draws: u32,
    },
    ImageDenoise {
        max_passes: u32,
        steps_per_pass: u32,
        token_grid: u64,
    },
    GeneratedImageFeedback {
        max_calls: u32,
        token_grid: u64,
    },
    Terminal,
}

/// Precompiled from each actual source tensor before admission. `storage_bytes` is the
/// caller-owned tensor's live allocation bound, including a wider original dtype if present.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceImageGeometry {
    pub width: u32,
    pub height: u32,
    pub storage_bytes: u64,
}

/// Metadata read from the selected model configuration before weight loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillModelShape {
    pub hidden_width: u32,
    pub intermediate_width: u32,
    pub vision_hidden_width: u32,
    pub source_patch_size: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub layers: u32,
    pub vocab_size: u32,
}

/// All output-affecting request facts and conservative upper bounds are frozen pre-load.
/// `step_scratch_bytes` is an admission upper bound for all simultaneously live step tensors;
/// `cache_token_bytes` is the upper bound for each retained KV token across the three caches.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionalInput {
    pub provider_id: String,
    pub prompt_sha256: [u8; 32],
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub cell: u32,
    pub input_images: u32,
    pub source_images: Vec<SourceImageGeometry>,
    pub prefill_tokens: [u32; CACHE_COUNT],
    pub prefill_hidden_width: u32,
    pub prefill_intermediate_width: u32,
    pub prefill_vision_hidden_width: u32,
    pub source_patch_size: u32,
    pub prefill_attention_heads: u32,
    pub prefill_kv_heads: u32,
    pub prefill_head_dim: u32,
    pub prefill_layers: u32,
    pub prefill_vocab_size: u32,
    pub max_new_tokens: u32,
    pub max_images: u32,
    pub steps_per_image: u32,
    pub model_resident_bytes: u64,
    pub cache_token_bytes: u64,
    pub step_scratch_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionalCeilings {
    pub max_width: u32,
    pub max_height: u32,
    pub max_input_images: u32,
    pub max_source_width: u32,
    pub max_source_height: u32,
    pub max_source_token_grid: u64,
    pub max_source_live_bytes: u64,
    pub max_prefill_hidden_width: u32,
    pub max_prefill_intermediate_width: u32,
    pub max_prefill_vision_hidden_width: u32,
    pub max_prefill_attention_heads: u32,
    pub max_prefill_kv_heads: u32,
    pub max_prefill_head_dim: u32,
    pub max_prefill_layers: u32,
    pub max_prefill_vocab_size: u32,
    pub max_prefill_scratch_bytes: u64,
    pub max_token_grid: u64,
    pub max_new_tokens: u32,
    pub max_images: u32,
    pub max_steps_per_image: u32,
    pub max_cache_tokens_per_cache: u64,
    pub max_peak_bytes: u64,
    pub max_total_work: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionalBudget {
    pub image_tokens: u64,
    pub source_image_tokens: u64,
    pub source_patch_tokens: u64,
    pub source_live_bytes: u64,
    pub prefill_scratch_bytes: u64,
    pub cache_token_caps: [u64; CACHE_COUNT],
    pub max_text_rollouts: u32,
    pub max_outer_steps: u32,
    pub peak_live_bytes: u64,
    pub total_work_units: u64,
}

/// The only stage families a run may realize. Deserialization is followed by `validate` so a
/// forged plan cannot widen a stage or weaken a cap after the caller's admission decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionalPlan {
    input: ConditionalInput,
    ceilings: ConditionalCeilings,
    templates: [StageTemplate; 4],
    cache_lifetimes: [CacheLifetime; CACHE_COUNT],
    budget: ConditionalBudget,
}

impl ConditionalPlan {
    pub fn compile(input: ConditionalInput, ceilings: ConditionalCeilings) -> Result<Self> {
        if input.provider_id != "sensenova_u1_8b"
            || input.width == 0
            || input.height == 0
            || input.cell == 0
            || !input.width.is_multiple_of(input.cell)
            || !input.height.is_multiple_of(input.cell)
            || input.steps_per_image == 0
            || input.prefill_tokens.contains(&0)
            || input.prefill_hidden_width == 0
            || input.prefill_intermediate_width == 0
            || input.prefill_vision_hidden_width == 0
            || input.source_patch_size == 0
            || !input.cell.is_multiple_of(input.source_patch_size)
            || input.prefill_attention_heads == 0
            || input.prefill_kv_heads == 0
            || input.prefill_head_dim == 0
            || input.prefill_layers == 0
            || input.prefill_vocab_size == 0
            || !input
                .prefill_attention_heads
                .is_multiple_of(input.prefill_kv_heads)
            || input.cache_token_bytes == 0
            || input.step_scratch_bytes == 0
        {
            return Err(refused(
                "invalid SenseNova geometry, prefill, or per-step bound",
            ));
        }
        if input.width > ceilings.max_width
            || input.height > ceilings.max_height
            || input.input_images > ceilings.max_input_images
            || input.prefill_hidden_width > ceilings.max_prefill_hidden_width
            || input.prefill_intermediate_width > ceilings.max_prefill_intermediate_width
            || input.prefill_vision_hidden_width > ceilings.max_prefill_vision_hidden_width
            || input.prefill_attention_heads > ceilings.max_prefill_attention_heads
            || input.prefill_kv_heads > ceilings.max_prefill_kv_heads
            || input.prefill_head_dim > ceilings.max_prefill_head_dim
            || input.prefill_layers > ceilings.max_prefill_layers
            || input.prefill_vocab_size > ceilings.max_prefill_vocab_size
            || u32::try_from(input.source_images.len()).map_err(|_| overflow())?
                != input.input_images
            || input.max_new_tokens > ceilings.max_new_tokens
            || input.max_images > ceilings.max_images
            || input.steps_per_image > ceilings.max_steps_per_image
        {
            return Err(refused("request exceeds an admitted geometry or count cap"));
        }

        // Both condition and text-uncondition prefixes splice every source image as
        // `<img><IMG_CONTEXT>×grid</img>`. Price the precompiled source tensor shapes,
        // retained patch arrays, and concatenated pixel_values before admitting prefill.
        let mut source_image_tokens = 0u64;
        let mut source_patch_tokens = 0u64;
        let mut source_pixels = 0u64;
        let mut source_live_bytes = 0u64;
        for source in &input.source_images {
            if source.width == 0
                || source.height == 0
                || source.width > ceilings.max_source_width
                || source.height > ceilings.max_source_height
                || !source.width.is_multiple_of(input.cell)
                || !source.height.is_multiple_of(input.cell)
            {
                return Err(refused("source image geometry exceeds its compiled bounds"));
            }
            let tokens = mul(
                u64::from(source.width / input.cell),
                u64::from(source.height / input.cell),
            )?;
            if tokens > ceilings.max_source_token_grid {
                return Err(refused("source image token grid exceeds its cap"));
            }
            let pixels = mul(u64::from(source.width), u64::from(source.height))?;
            let f32_rgb_bytes = mul(pixels, mul(IMAGE_CHANNELS, F32_BYTES)?)?;
            if source.storage_bytes < mul(pixels, IMAGE_CHANNELS)? {
                return Err(refused(
                    "source tensor storage bound understates its geometry",
                ));
            }
            source_image_tokens = add(source_image_tokens, tokens)?;
            source_patch_tokens = add(
                source_patch_tokens,
                mul(
                    u64::from(source.width / input.source_patch_size),
                    u64::from(source.height / input.source_patch_size),
                )?,
            )?;
            source_pixels = add(source_pixels, pixels)?;
            // Upper bound simultaneous original tensors, conversion/normalization
            // intermediates, per-image patches, and concatenated pixel_values.
            source_live_bytes = add(
                source_live_bytes,
                add(source.storage_bytes, mul(f32_rgb_bytes, 5)?)?,
            )?;
        }
        if source_live_bytes > ceilings.max_source_live_bytes {
            return Err(refused("source image live bytes exceed their cap"));
        }
        let source_prefix_min = add(
            add(source_image_tokens, mul(u64::from(input.input_images), 2)?)?,
            1,
        )?;
        if u64::from(input.prefill_tokens[0]) < source_prefix_min
            || u64::from(input.prefill_tokens[1]) < source_prefix_min
        {
            return Err(refused(
                "source-image prefill tokens understate expanded prefixes",
            ));
        }

        // Candle's one-hot source scatter materializes both host `p` and device `p_arr`
        // at [S,n_ctx]. MLX can also retain both around its dtype conversion. Price the
        // largest of the sequential prefixes, vision features, dense host/device block
        // masks, full-head score/softmax buffers, Q/K/V and MLP intermediates, and one
        // widened projection weight at f32 even when the resident store is narrower.
        let prefill_sequence = u64::from(*input.prefill_tokens.iter().max().unwrap());
        let hidden = u64::from(input.prefill_hidden_width);
        let intermediate = u64::from(input.prefill_intermediate_width);
        let heads = u64::from(input.prefill_attention_heads);
        let kv_heads = u64::from(input.prefill_kv_heads);
        let head_dim = u64::from(input.prefill_head_dim);
        let vision_hidden = u64::from(input.prefill_vision_hidden_width);
        let dense_scatter = mul(mul(prefill_sequence, source_image_tokens)?, F32_BYTES)?;
        let embed = mul(mul(prefill_sequence, hidden)?, F32_BYTES)?;
        let vision = mul(mul(source_image_tokens, hidden)?, F32_BYTES)?;
        let mask = mul(mul(prefill_sequence, prefill_sequence)?, F32_BYTES)?;
        let vision_patch = mul(mul(source_patch_tokens, vision_hidden)?, F32_BYTES)?;
        let mlp = mul(mul(prefill_sequence, intermediate)?, F32_BYTES)?;
        let q = mul(mul(prefill_sequence, mul(heads, head_dim)?)?, F32_BYTES)?;
        let kv = mul(mul(prefill_sequence, mul(kv_heads, head_dim)?)?, F32_BYTES)?;
        let merge = u64::from(input.cell / input.source_patch_size);
        let projection_elements = [
            mul(hidden, intermediate)?,
            mul(hidden, u64::from(input.prefill_vocab_size))?,
            mul(hidden, hidden)?,
            mul(hidden, mul(heads, head_dim)?)?,
            mul(hidden, mul(kv_heads, head_dim)?)?,
            mul(mul(hidden, vision_hidden)?, mul(merge, merge)?)?,
            mul(
                vision_hidden,
                mul(
                    IMAGE_CHANNELS,
                    mul(
                        u64::from(input.source_patch_size),
                        u64::from(input.source_patch_size),
                    )?,
                )?,
            )?,
        ]
        .into_iter()
        .max()
        .unwrap();
        let widened_projection = mul(projection_elements, F32_BYTES)?;
        let prefill_scratch_bytes = add(
            add(
                add(mul(dense_scatter, 2)?, mul(embed, 12)?)?,
                add(mul(vision, 4)?, mul(vision_patch, 8)?)?,
            )?,
            add(
                add(mul(mask, add(2, mul(heads, 2)?)?)?, mul(mlp, 4)?)?,
                add(
                    add(mul(q, 8)?, mul(kv, 12)?)?,
                    add(mul(widened_projection, 2)?, mul(prefill_sequence, 32)?)?,
                )?,
            )?,
        )?;
        if prefill_scratch_bytes > ceilings.max_prefill_scratch_bytes {
            return Err(refused("dense source prefill scratch exceeds its cap"));
        }

        let image_tokens = mul(
            u64::from(input.width / input.cell),
            u64::from(input.height / input.cell),
        )?;
        if image_tokens > ceilings.max_token_grid {
            return Err(refused("image token grid exceeds its cap"));
        }
        let image_block = add(image_tokens, 1)?; // the generated <img> splice and image block
        let future_blocks = mul(u64::from(input.max_images), image_block)?;
        let cond_cap = add(
            add(
                u64::from(input.prefill_tokens[0]),
                u64::from(input.max_new_tokens),
            )?,
            future_blocks,
        )?;
        let tu_cap = add(u64::from(input.prefill_tokens[1]), future_blocks)?;
        let iu_cap = u64::from(input.prefill_tokens[2]); // fixed image-uncondition prefix
        let min_cache_token_bytes = mul(
            mul(
                mul(u64::from(input.prefill_layers), 2)?,
                mul(kv_heads, head_dim)?,
            )?,
            F32_BYTES,
        )?;
        if input.cache_token_bytes < min_cache_token_bytes {
            return Err(refused(
                "retained KV token byte bound understates model geometry",
            ));
        }
        let cache_token_caps = [cond_cap, tu_cap, iu_cap];
        if cache_token_caps
            .iter()
            .any(|cap| *cap > ceilings.max_cache_tokens_per_cache)
        {
            return Err(refused("a retained KV cache exceeds its cap"));
        }
        let max_text_rollouts = input.max_images.checked_add(1).ok_or_else(overflow)?;
        let max_outer_steps = input
            .max_images
            .checked_mul(input.steps_per_image)
            .ok_or_else(overflow)?;
        let retained_image_bytes = mul(
            mul(
                mul(u64::from(input.width), u64::from(input.height))?,
                mul(IMAGE_CHANNELS, F32_BYTES)?,
            )?,
            u64::from(input.max_images),
        )?;
        let cache_sum = add(add(cond_cap, tu_cap)?, iu_cap)?;
        // All three caches, every retained output, model residents, and simultaneous image
        // scratch can coexist. Four image buffers bound noise/current/decoded/preview storage.
        let image_buffer_bytes = mul(
            mul(
                mul(u64::from(input.width), u64::from(input.height))?,
                mul(IMAGE_CHANNELS, F32_BYTES)?,
            )?,
            IMAGE_BUFFERS,
        )?;
        let peak_live_bytes = add(
            add(
                add(
                    input.model_resident_bytes,
                    mul(mul(cache_sum, input.cache_token_bytes)?, 2)?,
                )?,
                add(retained_image_bytes, source_live_bytes)?,
            )?,
            add(
                add(input.step_scratch_bytes, image_buffer_bytes)?,
                prefill_scratch_bytes,
            )?,
        )?;
        if peak_live_bytes > ceilings.max_peak_bytes {
            return Err(refused(
                "conservative simultaneously-live bytes exceed admission",
            ));
        }

        // Upper work bound in token-attention units. Prefill uses a quadratic upper bound;
        // rollout prices each draw against a full condition cache; each image step prices all
        // three guidance branches against the three full caches; feedback updates two caches.
        let mut total_work_units = 0u64;
        for tokens in input.prefill_tokens {
            total_work_units = add(total_work_units, mul(u64::from(tokens), u64::from(tokens))?)?;
        }
        total_work_units = add(
            total_work_units,
            add(
                mul(source_pixels, IMAGE_CHANNELS)?,
                mul(mul(source_image_tokens, source_image_tokens)?, 2)?,
            )?,
        )?;
        total_work_units = add(
            total_work_units,
            add(
                mul(
                    mul(prefill_sequence, prefill_sequence)?,
                    u64::from(input.prefill_attention_heads),
                )?,
                mul(
                    source_patch_tokens,
                    u64::from(input.prefill_vision_hidden_width),
                )?,
            )?,
        )?;
        total_work_units = add(
            total_work_units,
            mul(u64::from(input.max_new_tokens), cond_cap)?,
        )?;
        total_work_units = add(
            total_work_units,
            mul(
                mul(
                    mul(
                        u64::from(input.max_images),
                        u64::from(input.steps_per_image),
                    )?,
                    mul(image_tokens, cache_sum)?,
                )?,
                GUIDANCE_BRANCHES,
            )?,
        )?;
        total_work_units = add(
            total_work_units,
            mul(
                mul(u64::from(input.max_images), image_tokens)?,
                add(cond_cap, tu_cap)?,
            )?,
        )?;
        if total_work_units > ceilings.max_total_work {
            return Err(refused("conservative total work exceeds admission"));
        }

        let templates = [
            StageTemplate::TextRollout {
                max_calls: max_text_rollouts,
                max_draws: input.max_new_tokens,
            },
            StageTemplate::ImageDenoise {
                max_passes: input.max_images,
                steps_per_pass: input.steps_per_image,
                token_grid: image_tokens,
            },
            StageTemplate::GeneratedImageFeedback {
                max_calls: input.max_images,
                token_grid: image_tokens,
            },
            StageTemplate::Terminal,
        ];
        let cache_lifetimes = [
            CacheLifetime {
                kind: CacheKind::Condition,
                starts: Boundary::Prefill,
                ends: Boundary::Terminal,
            },
            CacheLifetime {
                kind: CacheKind::TextUncondition,
                starts: Boundary::Prefill,
                ends: Boundary::Terminal,
            },
            CacheLifetime {
                kind: CacheKind::ImageUncondition,
                starts: Boundary::Prefill,
                ends: Boundary::Terminal,
            },
        ];
        Ok(Self {
            input,
            ceilings,
            templates,
            cache_lifetimes,
            budget: ConditionalBudget {
                image_tokens,
                source_image_tokens,
                source_patch_tokens,
                source_live_bytes,
                prefill_scratch_bytes,
                cache_token_caps,
                max_text_rollouts,
                max_outer_steps,
                peak_live_bytes,
                total_work_units,
            },
        })
    }

    pub fn validate(&self) -> Result<()> {
        if *self != Self::compile(self.input.clone(), self.ceilings.clone())? {
            return Err(refused("serialized plan changed after admission"));
        }
        Ok(())
    }

    pub fn check_input(&self, input: &ConditionalInput) -> Result<()> {
        self.validate()?;
        if &self.input != input {
            return Err(refused("compiled plan belongs to a different request"));
        }
        Ok(())
    }

    pub fn check_source_images(&self, actual: &[SourceImageGeometry]) -> Result<()> {
        self.validate()?;
        if self.input.source_images != actual {
            return Err(refused("source tensors differ from precompiled geometry"));
        }
        Ok(())
    }

    pub fn check_prefill_shape(&self, actual: PrefillModelShape) -> Result<()> {
        self.validate()?;
        if self.input.prefill_hidden_width != actual.hidden_width
            || self.input.prefill_intermediate_width != actual.intermediate_width
            || self.input.prefill_vision_hidden_width != actual.vision_hidden_width
            || self.input.source_patch_size != actual.source_patch_size
            || self.input.prefill_attention_heads != actual.attention_heads
            || self.input.prefill_kv_heads != actual.kv_heads
            || self.input.prefill_head_dim != actual.head_dim
            || self.input.prefill_layers != actual.layers
            || self.input.prefill_vocab_size != actual.vocab_size
        {
            return Err(refused("model shape differs from precompiled prefill"));
        }
        Ok(())
    }

    pub fn check_prefill_tokens(&self, actual: [u32; CACHE_COUNT]) -> Result<()> {
        self.validate()?;
        if self.input.prefill_tokens != actual {
            return Err(refused("tokenized prefix differs from precompiled prefill"));
        }
        Ok(())
    }

    pub fn input(&self) -> &ConditionalInput {
        &self.input
    }

    pub fn templates(&self) -> &[StageTemplate; 4] {
        &self.templates
    }

    pub fn cache_lifetimes(&self) -> &[CacheLifetime; CACHE_COUNT] {
        &self.cache_lifetimes
    }

    pub fn budget(&self) -> &ConditionalBudget {
        &self.budget
    }
}

/// One realized text segment. `emitted` counts decoded text tokens; an End or Image stop
/// consumes one further token draw, which must fit the granted rollout budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextRollout {
    pub text: String,
    pub emitted: u32,
    pub stop: RolloutStop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RolloutStop {
    End,
    Image,
    Exhausted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminalReason {
    EndToken,
    TokenLimit,
    ImageLimit,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RealizedStage {
    Prefill {
        caches: [CacheKind; CACHE_COUNT],
    },
    TextRollout {
        call: u32,
        granted: u32,
        draws: u32,
        stop: RolloutStop,
    },
    ImageDenoise {
        image: u32,
        step: u32,
    },
    GeneratedImageFeedback {
        image: u32,
    },
    Terminal {
        reason: TerminalReason,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionalReceipt {
    pub stages: Vec<RealizedStage>,
    pub text_draws: u32,
    pub images: u32,
    pub feedback_calls: u32,
    pub actual_outer_steps: u32,
    pub reported_final_steps: Option<u32>,
    pub terminal: TerminalReason,
}

pub struct ConditionalOutput<I> {
    pub text: String,
    pub images: Vec<I>,
    pub receipt: ConditionalReceipt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    BeforePrefill,
    Ready,
    Rollout { granted: u32 },
    Denoise { image: u32, next_step: u32 },
    Feedback { image: u32 },
    Finished,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Image(u32),
    Terminal(TerminalReason),
}

/// Owner-local meter. Every public transition validates the compiled stage and remaining
/// budget before a driver may invoke its corresponding provider callback.
pub struct ConditionalGuard<'p> {
    plan: &'p ConditionalPlan,
    state: State,
    text_draws: u32,
    rollout_calls: u32,
    images: u32,
    feedback_calls: u32,
    actual_outer_steps: u32,
    stages: Vec<RealizedStage>,
}

impl<'p> ConditionalGuard<'p> {
    pub fn new(plan: &'p ConditionalPlan) -> Result<Self> {
        plan.validate()?;
        Ok(Self {
            plan,
            state: State::BeforePrefill,
            text_draws: 0,
            rollout_calls: 0,
            images: 0,
            feedback_calls: 0,
            actual_outer_steps: 0,
            stages: Vec::new(),
        })
    }

    fn expect(&self, expected: State) -> Result<()> {
        if self.state != expected {
            return Err(refused("transition is absent or has already been spent"));
        }
        Ok(())
    }

    pub fn prefilled(&mut self) -> Result<()> {
        self.expect(State::BeforePrefill)?;
        self.stages.push(RealizedStage::Prefill {
            caches: [
                CacheKind::Condition,
                CacheKind::TextUncondition,
                CacheKind::ImageUncondition,
            ],
        });
        self.state = State::Ready;
        Ok(())
    }

    pub fn remaining_text_draws(&self) -> Result<u32> {
        self.expect(State::Ready)?;
        Ok(self.plan.input.max_new_tokens - self.text_draws)
    }

    pub fn begin_rollout(&mut self) -> Result<u32> {
        let remaining = self.remaining_text_draws()?;
        if remaining == 0 || self.rollout_calls >= self.plan.budget.max_text_rollouts {
            return Err(refused("text rollout budget is exhausted"));
        }
        self.state = State::Rollout { granted: remaining };
        Ok(remaining)
    }

    fn finish_rollout(&mut self, segment: &TextRollout) -> Result<Action> {
        let State::Rollout { granted } = self.state else {
            return Err(refused("text rollout was not admitted"));
        };
        let stop_draw = u32::from(segment.stop != RolloutStop::Exhausted);
        let draws = segment
            .emitted
            .checked_add(stop_draw)
            .ok_or_else(overflow)?;
        if draws == 0 || draws > granted {
            return Err(refused("text rollout exceeded its granted draw budget"));
        }
        self.text_draws = self.text_draws.checked_add(draws).ok_or_else(overflow)?;
        self.rollout_calls = self.rollout_calls.checked_add(1).ok_or_else(overflow)?;
        self.stages.push(RealizedStage::TextRollout {
            call: self.rollout_calls,
            granted,
            draws,
            stop: segment.stop,
        });
        self.state = State::Ready;
        match segment.stop {
            RolloutStop::End => Ok(Action::Terminal(TerminalReason::EndToken)),
            RolloutStop::Exhausted => Ok(Action::Terminal(TerminalReason::TokenLimit)),
            RolloutStop::Image if self.images >= self.plan.input.max_images => {
                Ok(Action::Terminal(TerminalReason::ImageLimit))
            }
            RolloutStop::Image => {
                self.state = State::Denoise {
                    image: self.images,
                    next_step: 0,
                };
                Ok(Action::Image(self.images))
            }
        }
    }

    fn check_step(&self, image: u32, step: u32) -> Result<()> {
        self.expect(State::Denoise {
            image,
            next_step: step,
        })?;
        if image >= self.plan.input.max_images || step >= self.plan.input.steps_per_image {
            return Err(refused("image pass or step exceeds its compiled template"));
        }
        Ok(())
    }

    fn completed_step(&mut self, image: u32, step: u32) -> Result<u32> {
        self.check_step(image, step)?;
        self.actual_outer_steps = self
            .actual_outer_steps
            .checked_add(1)
            .ok_or_else(overflow)?;
        if self.actual_outer_steps > self.plan.budget.max_outer_steps {
            return Err(refused("actual outer steps exceed the plan"));
        }
        self.stages.push(RealizedStage::ImageDenoise {
            image,
            step: step + 1,
        });
        self.state = State::Denoise {
            image,
            next_step: step + 1,
        };
        Ok(self.actual_outer_steps)
    }

    fn begin_feedback(&mut self, image: u32) -> Result<()> {
        self.expect(State::Denoise {
            image,
            next_step: self.plan.input.steps_per_image,
        })?;
        self.state = State::Feedback { image };
        Ok(())
    }

    fn completed_feedback(&mut self, image: u32) -> Result<()> {
        self.expect(State::Feedback { image })?;
        if image != self.images || self.feedback_calls >= self.plan.input.max_images {
            return Err(refused("feedback pass is absent or already spent"));
        }
        self.stages
            .push(RealizedStage::GeneratedImageFeedback { image });
        self.feedback_calls += 1;
        self.images += 1;
        self.state = State::Ready;
        Ok(())
    }

    fn finish(mut self, reason: TerminalReason) -> Result<ConditionalReceipt> {
        self.expect(State::Ready)?;
        self.state = State::Finished;
        self.stages.push(RealizedStage::Terminal { reason });
        let reported_final_steps = if self.images > 0 && self.images < self.plan.input.max_images {
            Some(self.plan.budget.max_outer_steps)
        } else {
            None
        };
        Ok(ConditionalReceipt {
            stages: self.stages,
            text_draws: self.text_draws,
            images: self.images,
            feedback_calls: self.feedback_calls,
            actual_outer_steps: self.actual_outer_steps,
            reported_final_steps,
            terminal: reason,
        })
    }
}

/// A provider supplies operations, while the driver fixes their order and budget. The cache array
/// is owned by one invocation and drops on every terminal/error/cancel path; each element has a
/// distinct compiled lifetime even though a native provider may use a common KV-cache type.
pub trait ConditionalExecutor {
    type Cache;
    type Context;
    type Image;

    /// Must inspect the actual caller-owned tensors before weights load or prefill begins.
    /// `storage_bytes` includes the original tensor allocation, even for a wider dtype.
    fn source_images(&self) -> Result<Vec<SourceImageGeometry>>;
    /// Read the selected model configuration before any prefill allocation.
    fn prefill_model_shape(&self) -> Result<PrefillModelShape>;
    /// Return actual tokenized lengths of the three expanded prefixes before prefill.
    fn prefill_token_counts(&self) -> Result<[u32; CACHE_COUNT]>;

    fn prefill(
        &mut self,
        plan: &ConditionalPlan,
    ) -> Result<(Self::Context, [Self::Cache; CACHE_COUNT])>;
    fn rollout(
        &mut self,
        context: &mut Self::Context,
        caches: &mut [Self::Cache; CACHE_COUNT],
        granted_draws: u32,
    ) -> Result<TextRollout>;
    fn denoise_step(
        &mut self,
        context: &mut Self::Context,
        caches: &mut [Self::Cache; CACHE_COUNT],
        image: u32,
        step: u32,
    ) -> Result<()>;
    fn finish_image(&mut self, context: &mut Self::Context, image: u32) -> Result<Self::Image>;
    fn feedback(
        &mut self,
        context: &mut Self::Context,
        caches: &mut [Self::Cache; CACHE_COUNT],
        image: u32,
        output: &Self::Image,
    ) -> Result<()>;
}

pub fn run_conditional<E: ConditionalExecutor>(
    plan: &ConditionalPlan,
    input: &ConditionalInput,
    cancel: &CancelFlag,
    executor: &mut E,
    progress: &mut dyn FnMut(Progress),
) -> Result<ConditionalOutput<E::Image>> {
    plan.check_input(input)?;
    plan.check_source_images(&executor.source_images()?)?;
    plan.check_prefill_shape(executor.prefill_model_shape()?)?;
    plan.check_prefill_tokens(executor.prefill_token_counts()?)?;
    let mut guard = ConditionalGuard::new(plan)?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    let (mut context, mut caches) = executor.prefill(plan)?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    guard.prefilled()?;
    let mut text = String::new();
    let mut images = Vec::new();
    let terminal = loop {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if guard.remaining_text_draws()? == 0 {
            break TerminalReason::TokenLimit;
        }
        let granted = guard.begin_rollout()?;
        progress(Progress::ConditioningComplete);
        let segment = executor.rollout(&mut context, &mut caches, granted)?;
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let action = guard.finish_rollout(&segment)?;
        text.push_str(&segment.text);
        match action {
            Action::Terminal(reason) => break reason,
            Action::Image(image) => {
                for step in 0..plan.input.steps_per_image {
                    if cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
                    guard.check_step(image, step)?;
                    executor.denoise_step(&mut context, &mut caches, image, step)?;
                    if cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
                    let current = guard.completed_step(image, step)?;
                    progress(Progress::Step {
                        current,
                        total: plan.budget.max_outer_steps,
                    });
                }
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                guard.begin_feedback(image)?;
                let output = executor.finish_image(&mut context, image)?;
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                executor.feedback(&mut context, &mut caches, image, &output)?;
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                guard.completed_feedback(image)?;
                images.push(output);
            }
        }
    };
    let receipt = guard.finish(terminal)?;
    if let Some(total) = receipt.reported_final_steps {
        progress(Progress::Step {
            current: total,
            total,
        });
    }
    if receipt.images > 0 {
        progress(Progress::Decoding);
    }
    Ok(ConditionalOutput {
        text,
        images,
        receipt,
    })
}

#[cfg(test)]
mod tests;
