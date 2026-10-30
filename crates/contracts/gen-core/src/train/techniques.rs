//! Optional training techniques, their noise derivations, and enabled auxiliary losses.

use super::*;

/// Per-technique support flags for the optional training techniques of epic 2123 (weight noising
/// weight noising, gradient noise, resolution buckets and depth anchoring today; masked loss and
/// the perceptual identity/body/latent losses join here as their stories land). Each flag gates one
/// technique's [`TrainingConfig`] knob(s) through `gen_core_train::validate_training_techniques`.
///
/// Non-supporting descriptors spell [`TrainingTechniques::NONE`], so a new flag defaults to
/// *unsupported* everywhere without touching them; a supporting descriptor names the flags it
/// implements (and, once more than one flag exists, completes the rest with `..NONE`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrainingTechniques {
    /// Honors [`TrainingConfig::weight_noise_sigma`] (relative-mode adapter weight noising).
    pub weight_noise: bool,
    /// Honors [`TrainingConfig::depth_anchoring`] (decoded-x0 Depth-Anything-V2 anchoring loss).
    pub depth_anchoring: bool,
    /// Honors [`TrainingConfig::gradient_noise_eta`] / [`TrainingConfig::gradient_noise_gamma`]
    /// (annealed adapter gradient noise, sc-24827).
    pub gradient_noise: bool,
    /// Honors [`TrainingConfig::resolution_buckets`] (multi-resolution buckets with per-bucket
    /// repeat counts, sc-2127).
    pub resolution_buckets: bool,
    /// Honors [`TrainingConfig::subject_mask_loss`] (subject-masked loss weighting, sc-24828).
    pub subject_mask_loss: bool,
    /// Honors [`TrainingConfig::identity_loss`] (decoded-x0 ArcFace identity loss, sc-24831).
    pub identity_loss: bool,
    /// Honors [`TrainingConfig::face_landmark_loss`] (decoded-x0 FaceMesh landmark loss, sc-24831).
    pub face_landmark_loss: bool,
    /// Honors [`BodyLossesConfig::proportion`] (ViTPose+ bone-length-ratio loss, sc-24832).
    pub body_proportion_loss: bool,
    /// Honors [`BodyLossesConfig::shape`] (HybrIK SMPL-beta loss, sc-24832).
    pub body_shape_loss: bool,
    /// Honors [`BodyLossesConfig::normal`] (Sapiens surface-normal loss, sc-24832).
    pub normal_loss: bool,
    /// Honors [`TrainingConfig::vae_anchor`] (decoded-x0 FLUX.2-VAE-encoder anchor loss, sc-24833).
    pub vae_anchor_loss: bool,
    /// Honors [`TrainingConfig::latent_lpips`] (E-LatentLPIPS on the x0 latent, sc-24833) — only a
    /// trainer whose latent family has published weights ([`LatentLpipsFamily`]) declares it.
    pub latent_lpips_loss: bool,
    /// Honors [`TrainingConfig::training_adapter`] (a frozen LoRA applied to the base for the
    /// training forward/backward only — the Z-Image-Turbo de-distill adapter, sc-25213).
    pub training_adapter: bool,
    /// Not a technique knob: the trainer's aux-loss builder supplies its **own x0 decoder** (e.g.
    /// Mage decodes through its full VAE), so the decoded-x0 losses do not need
    /// [`TrainingConfig::perceptual_decoder_dir`] and the shared floor does not require it.
    pub builtin_x0_decoder: bool,
}

impl TrainingTechniques {
    /// No optional technique supported — every technique knob must stay at its off value.
    pub const NONE: Self = Self {
        weight_noise: false,
        subject_mask_loss: false,
        depth_anchoring: false,
        gradient_noise: false,
        resolution_buckets: false,
        identity_loss: false,
        face_landmark_loss: false,
        body_proportion_loss: false,
        body_shape_loss: false,
        normal_loss: false,
        vae_anchor_loss: false,
        latent_lpips_loss: false,
        training_adapter: false,
        builtin_x0_decoder: false,
    };

    /// The adapter-noise pair every LoRA/LoKr trainer implements at its optimizer step (epic 2123
    /// S2, sc-24827): weight noising after the update and gradient noise between clip and step.
    pub const ADAPTER_NOISE: Self = Self {
        weight_noise: true,
        gradient_noise: true,
        resolution_buckets: false,
        depth_anchoring: false,
        subject_mask_loss: false,
        identity_loss: false,
        face_landmark_loss: false,
        body_proportion_loss: false,
        body_shape_loss: false,
        normal_loss: false,
        vae_anchor_loss: false,
        latent_lpips_loss: false,
        training_adapter: false,
        builtin_x0_decoder: false,
    };
}

/// Upstream / Neelakantan et al. (2015) default annealing exponent for gradient noise.
pub const DEFAULT_GRADIENT_NOISE_GAMMA: f32 = 0.55;

/// The gradient-noise standard deviation at optimizer update `update_idx` (0-based):
/// `σ_t = eta / (1 + t)^gamma` — upstream ai-toolkit-perceptual's `neelakantan` mode, the one
/// formula both backends' gradient-noise kernels share. `eta == 0` ⇒ `0` (off).
pub fn gradient_noise_std(eta: f32, gamma: f32, update_idx: u32) -> f32 {
    if eta == 0.0 {
        return 0.0;
    }
    (eta as f64 / (1.0 + update_idx as f64).powf(gamma as f64)) as f32
}

/// Domain-separation salt for the **weight-noise** RNG stream (epic 2123, sc-24826).
pub const WEIGHT_NOISE_SALT: u64 = 0x5745_4947_4854_4E5A; // "WEIGHTNZ"
/// Domain-separation salt for the **gradient-noise** RNG stream (sc-24827) — independent of the
/// weight-noise stream so turning one technique on never shifts the other's draws.
pub const GRADIENT_NOISE_SALT: u64 = 0x4752_4144_4E4F_4953; // "GRADNOIS"

/// SplitMix64 finalizer — a bijective 64-bit mix, so distinct inputs map to well-separated keys.
pub fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The RNG seed for one adapter tensor's technique-noise draw (epic 2123 E4): derived only from
/// the job `seed`, the technique's stream `salt` ([`WEIGHT_NOISE_SALT`] / [`GRADIENT_NOISE_SALT`]),
/// the 0-based optimizer `update_idx`, and the tensor's index in **sorted key order** — so a
/// seeded run (or a resumed one) reproduces the same noise on either backend's kernel, and
/// `HashMap` iteration order can never leak in.
pub fn technique_noise_key(seed: u64, salt: u64, update_idx: u32, tensor_idx: usize) -> u64 {
    let stream = splitmix64(seed ^ salt).wrapping_add(update_idx as u64);
    splitmix64(splitmix64(stream) ^ tensor_idx as u64)
}

/// The enabled **auxiliary (perceptual) losses** of `cfg`, by name, in loss-index order — the
/// same arms both backends' shared aux-loss builders drive (depth, identity, face-landmark, the
/// three body losses, the VAE anchor, E-LatentLPIPS). Empty when none is on.
pub fn enabled_aux_losses(cfg: &TrainingConfig) -> Vec<&'static str> {
    [
        ("depth", cfg.depth_anchoring.schedule.is_enabled()),
        ("identity", cfg.identity_loss.schedule.is_enabled()),
        (
            "face-landmark",
            cfg.face_landmark_loss.schedule.is_enabled(),
        ),
        ("body-proportion", cfg.body_losses.proportion.is_enabled()),
        ("body-shape", cfg.body_losses.shape.is_enabled()),
        ("normal", cfg.body_losses.normal.is_enabled()),
        ("vae_anchor", cfg.vae_anchor.schedule.is_enabled()),
        ("latent_lpips", cfg.latent_lpips.schedule.is_enabled()),
    ]
    .into_iter()
    .filter_map(|(name, on)| on.then_some(name))
    .collect()
}

/// Whether a decoded-x0 loss lacks its x0 decoder: [`TrainingConfig::perceptual_decoder_dir`] is
/// unset and the trainer does not build its own decoder
/// ([`TrainingTechniques::builtin_x0_decoder`]).
#[doc(hidden)]
pub fn lacks_x0_decoder(desc: &TrainerDescriptor, cfg: &TrainingConfig) -> bool {
    cfg.perceptual_decoder_dir.is_none() && !desc.techniques.builtin_x0_decoder
}
