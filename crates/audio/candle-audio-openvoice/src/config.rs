//! Architecture + front-end constants for the OpenVoice V2 tone-color converter (sc-13223).
//!
//! These reproduce `myshell-ai/OpenVoiceV2/converter/config.json` exactly (the pinned checkpoint's
//! `data` + `model` blocks) plus the `spectrogram_torch` front-end OpenVoice's `mel_processing.py`
//! feeds both the tone-color reference encoder and the posterior encoder. Getting any of these
//! wrong would silently mis-shape the port against its trained weights, so they live in one audited
//! place next to the model. Values are asserted against the shipped `config.json` at load
//! ([`crate::pipeline`]) — a drifted checkpoint is a typed error, never a silent mis-load.

/// Converter operating sample rate (Hz) — `data.sampling_rate`. Both reference clips and the source
/// clip are resampled to this before analysis, and the decoder emits at this rate.
pub const SAMPLE_RATE: u32 = 22_050;

/// STFT / FFT size (`data.filter_length`) — a power of two, so the host radix-2 DFT serves it.
pub const FILTER_LENGTH: usize = 1024;

/// STFT hop (`data.hop_length`) — also the decoder's total upsample factor (∏ upsample_rates), so
/// `n_output_samples ≈ n_frames · HOP`, preserving the source duration.
pub const HOP_LENGTH: usize = 256;

/// STFT window length (`data.win_length`) — equals [`FILTER_LENGTH`], so the Hann window needs no
/// centering pad.
pub const WIN_LENGTH: usize = 1024;

/// Linear-spectrogram channel count (`filter_length / 2 + 1`) — the posterior encoder's input width
/// and the reference encoder's frequency axis.
pub const SPEC_CHANNELS: usize = FILTER_LENGTH / 2 + 1; // 513

/// VITS latent width (`model.inter_channels`) — the flow operates on this channel count and the
/// decoder consumes it.
pub const INTER_CHANNELS: usize = 192;

/// WaveNet hidden width (`model.hidden_channels`) shared by the posterior encoder and every flow
/// coupling's residual stack.
pub const HIDDEN_CHANNELS: usize = 192;

/// Tone-color / speaker conditioning width (`model.gin_channels`) — the reference encoder's output
/// and the flow's conditioning input.
pub const GIN_CHANNELS: usize = 256;

/// WaveNet kernel size for the posterior encoder + flow couplings (`5`).
pub const WN_KERNEL_SIZE: usize = 5;

/// Posterior encoder (`enc_q`) WaveNet depth — `PosteriorEncoder(..., n_layers=16)`.
pub const ENC_Q_N_LAYERS: usize = 16;

/// Flow coupling WaveNet depth — `ResidualCouplingBlock(..., n_layers=4)`.
pub const FLOW_N_LAYERS: usize = 4;

/// Number of residual-coupling flows (`ResidualCouplingBlock(n_flows=4)`); the module list is
/// `[coupling, flip] × N_FLOWS` (checkpoint indices `flow.flows.{0,2,4,6}` = couplings).
pub const N_FLOWS: usize = 4;

/// HiFi-GAN decoder upsample strides (`model.upsample_rates`); ∏ = [`HOP_LENGTH`].
pub const UPSAMPLE_RATES: [usize; 4] = [8, 8, 2, 2];

/// HiFi-GAN decoder transposed-conv kernel sizes (`model.upsample_kernel_sizes`).
pub const UPSAMPLE_KERNEL_SIZES: [usize; 4] = [16, 16, 4, 4];

/// HiFi-GAN first-layer channel count (`model.upsample_initial_channel`).
pub const UPSAMPLE_INITIAL_CHANNEL: usize = 512;

/// HiFi-GAN residual-block kernel sizes (`model.resblock_kernel_sizes`).
pub const RESBLOCK_KERNEL_SIZES: [usize; 3] = [3, 7, 11];

/// HiFi-GAN residual-block dilation sets (`model.resblock_dilation_sizes`).
pub const RESBLOCK_DILATIONS: [[usize; 3]; 3] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

/// LeakyReLU slope inside the HiFi-GAN generator (`modules.LRELU_SLOPE = 0.1`).
pub const LRELU_SLOPE: f64 = 0.1;

/// Reference-encoder Conv2d channel progression (`ref_enc_filters`); each is a `(3,3)`, stride
/// `(2,2)`, padding `(1,1)` weight-normed conv followed by ReLU.
pub const REF_ENC_FILTERS: [usize; 6] = [32, 32, 64, 64, 128, 128];

/// Reference-encoder GRU hidden size (`256 // 2`); the projection reads `2·GRU_HIDDEN = 128`.
pub const REF_ENC_GRU_HIDDEN: usize = 128;

/// `zero_g` (`model.zero_g = true`): the posterior encoder and decoder receive a **zeroed** `g`
/// (their conditioning contributes only its bias); the entire timbre transfer therefore happens in
/// the flow — forward-conditioned on the source tone color, reverse-conditioned on the target.
pub const ZERO_G: bool = true;

/// Default posterior-sampling temperature (`convert(..., tau=0.3)`) — scales the Gaussian drawn
/// around the posterior mean. Overridable via `AudioTransformRequest::strength`.
pub const DEFAULT_TAU: f32 = 0.3;

/// Minimum source / reference clip length (samples at the source rate) the converter accepts. One
/// STFT frame is meaningless, and the reference encoder's six stride-2 convs collapse a short clip
/// to zero time steps — so a shorter clip is a typed error, not a degenerate run.
pub const MIN_SAMPLES: usize = FILTER_LENGTH;
