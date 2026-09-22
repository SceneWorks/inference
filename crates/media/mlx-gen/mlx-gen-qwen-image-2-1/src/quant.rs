//! Packed (pre-quantized) tier vocabulary for Qwen-Image 2.1 — the consume side of
//! [`crate::convert`] (sc-24112).
//!
//! # What a tier is
//!
//! A tier is a **complete, standalone snapshot directory** in the released
//! `Qwen/Qwen-Image-2.1` layout whose weight-bearing components have already been affine-quantized
//! offline. Nothing is quantized on the fly: [`crate::load`] points at the tier it wants and every
//! Linear auto-detects its packed triple `{base}.weight` (u32 codes) + `{base}.scales` +
//! `{base}.biases` through the shared [`mlx_gen::quant::lin`] loader. The 2512 crate
//! (`mlx_gen_qwen_image::quant`) is the template; the deliberate differences are documented below
//! and are **not** inherited from it.
//!
//! # Per-component tiering — and why it is NOT the 2512 table
//!
//! | component | bf16 tier | q8 tier | q4 tier | why |
//! |---|---|---|---|---|
//! | `transformer/` (DiT, 7.12 B) | dense bf16 | **packed Q8** g64 | **packed Q4** g64 | every leaf is a `Linear`; all released input widths (64 / 256 / 4096 / 12288) are multiples of 64 |
//! | `text_encoder/` Qwen3 language tower (7.57 B loaded) | dense bf16 | **packed Q8** g64 | **packed Q8** g64 ([`TEXT_ENCODER_Q4_FLOOR`]) | see below |
//! | `text_encoder/` token embedding (622 M) | dense | dense | dense | [`crate::text_encoder`]'s `quantize` packs decoder layers only; keeping it dense preserves pre-quantize ≡ quantize-at-load |
//! | `vae/` (338 M) | dense f32 | dense f32 | dense f32 | all-conv; zero group-quantizable 2-D leaves, and the released file ships f32 |
//!
//! **The text encoder is packed here, and it is dense in the 2512 crate.** That is a deliberate
//! divergence, not an oversight. 2512 pairs a ~20 B DiT with a ~7 B Qwen2.5-VL tower, so a dense
//! tower is a minority of the footprint and upstream's `skip_quantization` note costs little. Qwen-
//! Image 2.1 pairs a **7.12 B** DiT with a **7.57 B** Qwen3 tower: with a staged (`Sequential`) text
//! encoder the resident floor is `max(text encoder, DiT + VAE)` (never the sum), so a dense tower
//! would pin that floor at ~15.1 GB at *every* tier and a Q4 tier would buy essentially nothing.
//! Reusing 2512's table here would have produced a tier that cannot do its job. Numbers:
//! [`crate::memory_strategy`].
//!
//! [`TEXT_ENCODER_Q4_FLOOR`] keeps the tower at Q8 on the Q4 tier. That floor is a **prior**, not a
//! measurement of this model: `mlx_gen_mage::quant`'s measured Qwen-LM-tower sweep found a Q4 SwiGLU
//! MLP collapses generation quality while Q8 attention + MLP holds, and the Qwen3 tower here is the
//! same block shape. It is declared through `Capabilities::component_precision_floors` so a caller's
//! effective-tier label and memory evidence carry the substitution rather than having to rediscover
//! it. Confirming or retiring it against real weights belongs to the epic's terminal measurement
//! story; nothing here claims to have measured it.
//!
//! # Group size 64, uniformly
//!
//! The converter packs at [`GROUP_SIZE`] **only**. A packed component's `config.json` carries one
//! `quantization.group_size`, and [`mlx_gen::quant::packed_bits`] derives the bit-width from the
//! packed shapes *at the group size the loader passes* — so a tier that mixed group sizes would be
//! silently mis-read (a group-32 Q8 pack read at group 64 decodes as Q4). A `Linear` whose input
//! width is not a multiple of 64 therefore stays **dense** in a packed tier. The released model has
//! none ([`crate::convert::transformer_widths_are_group_aligned`]); only the miniature
//! parity snapshot does, which is why its converted tiers are partially dense.
//!
//! Note this is a narrower rule than [`crate::transformer::QwenImage21Transformer::quantize`]'s
//! load-time fallback (64 → 32 → dense). On the released geometry the two agree exactly, and
//! [`crate::convert`]'s byte-identity test pins that. On the tiny fixture they differ, deliberately.

use std::path::Path;

use mlx_gen::gen_core::{ComponentPrecisionFloor, PrecisionFloorComponent};
use mlx_gen::{Error, Quant, Result};

/// Group size every packed Qwen-Image 2.1 component is written and read at.
///
/// Spelled as a literal rather than as [`mlx_gen::quant::DEFAULT_GROUP_SIZE`] so the cross-backend
/// parity gate can compare it against the Candle twin's own constant textually;
/// `group_size_is_the_codebase_default` pins the two to the same number.
pub const GROUP_SIZE: i32 = 64;

/// The tier the Qwen3 language tower is resident at when **Q4** is selected. See the module docs:
/// a prior carried from `mlx_gen_mage`'s measured Qwen-LM-tower sweep, declared rather than hidden.
pub const TEXT_ENCODER_Q4_FLOOR: Quant = Quant::Q8;

/// The snapshot subdirectories a tier owns, in conversion order.
pub const PACKED_COMPONENTS: &[&str] = &["transformer", "text_encoder"];

/// Snapshot subdirectories a tier copies through **dense**: the RGBA autoencoder (all-conv, no
/// group-quantizable 2-D leaf) plus the tokenizer/processor/scheduler trees.
pub const DENSE_COMPONENTS: &[&str] = &["vae", "processor", "tokenizer", "scheduler"];

/// One installable numeric tier of `qwen_image_2_1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// The released dense snapshot itself — no conversion, just a mirror.
    Bf16,
    /// Packed Q8 DiT + Q8 language tower.
    Q8,
    /// Packed Q4 DiT + **Q8** language tower ([`TEXT_ENCODER_Q4_FLOOR`]).
    Q4,
}

impl Tier {
    /// Every installable tier, densest first.
    pub const ALL: [Self; 3] = [Self::Bf16, Self::Q8, Self::Q4];

    /// The per-tier snapshot directory name (`<tiers_root>/<dir_name>/`).
    pub const fn dir_name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }

    /// The selected [`Quant`] a caller passes through `LoadSpec::quantize`, or `None` for dense.
    pub const fn selected_quant(self) -> Option<Quant> {
        match self {
            Self::Bf16 => None,
            Self::Q8 => Some(Quant::Q8),
            Self::Q4 => Some(Quant::Q4),
        }
    }

    /// The tier a selected [`Quant`] names.
    pub const fn from_selected(quant: Option<Quant>) -> Option<Self> {
        match quant {
            None => Some(Self::Bf16),
            Some(Quant::Q8) => Some(Self::Q8),
            Some(Quant::Q4) => Some(Self::Q4),
            Some(_) => None,
        }
    }

    /// Bits the `transformer/` is packed at in this tier (`None` = dense).
    pub const fn transformer_bits(self) -> Option<i32> {
        match self {
            Self::Bf16 => None,
            Self::Q8 => Some(8),
            Self::Q4 => Some(4),
        }
    }

    /// Bits the `text_encoder/` decoder layers are packed at in this tier (`None` = dense). Q4
    /// resolves to 8 — the declared [`TEXT_ENCODER_Q4_FLOOR`].
    pub const fn text_encoder_bits(self) -> Option<i32> {
        match self {
            Self::Bf16 => None,
            Self::Q8 | Self::Q4 => Some(8),
        }
    }
}

/// The `component_precision_floors` this provider advertises — the single Q4 text-encoder floor.
/// Published at every tier (the descriptor is tier-independent); it *applies* only at Q4, which
/// [`mlx_gen::gen_core::ComponentPrecisionFloor::applies_to`] enforces.
pub const COMPONENT_PRECISION_FLOORS: &[ComponentPrecisionFloor] = &[ComponentPrecisionFloor {
    component: PrecisionFloorComponent::TextEncoder,
    selected_tier: Quant::Q4,
    resident_tier: TEXT_ENCODER_Q4_FLOOR,
}];

/// The tier a snapshot on disk **is**, read from the `quantization` markers its packed components
/// carry ([`mlx_gen::quant::packed_quant_bits_at`]).
///
/// A snapshot whose `transformer/` and `text_encoder/` disagree in a way no tier produces is a hard
/// error: silently picking one of them would serve a mislabelled tier, which is exactly the failure
/// [`mlx_gen::quant::write_quantized_config`] documents. A dense snapshot reads as [`Tier::Bf16`].
pub fn installed_tier(root: &Path) -> Result<Tier> {
    let dit = mlx_gen::quant::packed_quant_bits_at(&root.join("transformer"))?;
    let te = mlx_gen::quant::packed_quant_bits_at(&root.join("text_encoder"))?;
    for (label, dir) in [
        ("transformer", "transformer"),
        ("text_encoder", "text_encoder"),
    ] {
        let component = root.join(dir);
        if mlx_gen::quant::packed_quant_bits_at(&component)?.is_some() {
            let group = mlx_gen::quant::packed_quant_group_size_at(&component)?;
            if group != Some(GROUP_SIZE) {
                return Err(Error::Msg(format!(
                    "qwen_image_2_1: {label}/ declares quantization.group_size {group:?}, but every \
                     packed tier of this model is written and read at group {GROUP_SIZE}; a \
                     mismatched group size decodes the packed codes at the wrong bit-width"
                )));
            }
        }
    }
    Tier::ALL
        .into_iter()
        .find(|tier| tier.transformer_bits() == dit && tier.text_encoder_bits() == te)
        .ok_or_else(|| {
            Error::Msg(format!(
                "qwen_image_2_1: no installable tier has transformer bits {dit:?} with text-encoder \
                 bits {te:?}; the installable tiers are bf16 (dense/dense), q8 (8/8) and q4 (4/8)"
            ))
        })
}

/// Validate a caller's `LoadSpec::quantize` against the tier actually on disk.
///
/// Qwen-Image 2.1 ships **pre-quantized** tiers; `spec.quantize` selects one, it does not request a
/// load-time quantization pass. So a request that matches the installed tier is accepted (and the
/// load is a plain packed-detect load), and a mismatch is refused with the path to take instead —
/// never silently served at the installed tier, which is what a no-op `quantize()` over packed
/// weights would otherwise do.
///
/// A **dense** snapshot with a Q4/Q8 request keeps the historical behaviour: it is quantized at
/// load ([`crate::transformer::QwenImage21Transformer::quantize`]), and this returns `true`.
pub fn needs_load_time_quant(root: &Path, requested: Option<Quant>) -> Result<bool> {
    let installed = installed_tier(root)?;
    let Some(requested) = requested else {
        // A dense request against a packed tier would serve the packed weights unannounced.
        if installed != Tier::Bf16 {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: {} is a pre-quantized {} tier but no quantization was requested; \
                 point at the bf16 snapshot, or pass the matching quantize tier",
                root.display(),
                installed.dir_name()
            )));
        }
        return Ok(false);
    };
    let Some(wanted) = Tier::from_selected(Some(requested)) else {
        return Err(Error::Unsupported(format!(
            "qwen_image_2_1: {requested:?} is not an MLX affine tier (Q4/Q8)"
        )));
    };
    match installed {
        Tier::Bf16 => Ok(true),
        installed if installed == wanted => Ok(false),
        installed => Err(Error::Msg(format!(
            "qwen_image_2_1: {} is a pre-quantized {} tier but {} was requested; quantize is a \
             no-op on packed weights, so the request would silently serve {}. Point at the {} \
             snapshot (or a dense one).",
            root.display(),
            installed.dir_name(),
            wanted.dir_name(),
            installed.dir_name(),
            wanted.dir_name()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_table_is_the_documented_per_component_assignment() {
        assert_eq!(Tier::Bf16.transformer_bits(), None);
        assert_eq!(Tier::Bf16.text_encoder_bits(), None);
        assert_eq!(Tier::Q8.transformer_bits(), Some(8));
        assert_eq!(Tier::Q8.text_encoder_bits(), Some(8));
        assert_eq!(Tier::Q4.transformer_bits(), Some(4));
        assert_eq!(
            Tier::Q4.text_encoder_bits(),
            Some(8),
            "the Q4 tier keeps the Qwen3 tower at the declared Q8 floor"
        );
        // Round-trip through the caller-visible selector.
        for tier in Tier::ALL {
            assert_eq!(Tier::from_selected(tier.selected_quant()), Some(tier));
        }
        assert_eq!(Tier::from_selected(Some(Quant::Nvfp4)), None);
    }

    /// [`GROUP_SIZE`] is the codebase-wide default, spelled as a literal only so the
    /// cross-backend parity gate can compare it with the Candle twin's.
    #[test]
    fn group_size_is_the_codebase_default() {
        assert_eq!(GROUP_SIZE, mlx_gen::quant::DEFAULT_GROUP_SIZE);
    }

    #[test]
    fn the_declared_floor_matches_the_tier_table() {
        let floor = COMPONENT_PRECISION_FLOORS
            .iter()
            .find(|floor| floor.component == PrecisionFloorComponent::TextEncoder)
            .expect("the text-encoder floor is declared");
        assert_eq!(floor.selected_tier, Quant::Q4);
        assert_eq!(floor.resident_tier, Quant::Q8);
        assert!(floor.applies_to(Quant::Q4));
        assert!(!floor.applies_to(Quant::Q8));
        // The descriptor declaration and the converter's own table must not drift apart.
        assert_eq!(
            Tier::Q4.text_encoder_bits(),
            Some(floor.resident_tier.bits())
        );
        assert_eq!(
            Tier::Q8.text_encoder_bits(),
            Some(Quant::Q8.bits()),
            "Q8 applies no promotion"
        );
    }

    #[test]
    fn a_dense_snapshot_reads_as_bf16_and_still_needs_load_time_quant() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::create_dir_all(root.join("text_encoder")).unwrap();
        std::fs::write(root.join("transformer/config.json"), "{}").unwrap();
        std::fs::write(root.join("text_encoder/config.json"), "{}").unwrap();

        assert_eq!(installed_tier(root).unwrap(), Tier::Bf16);
        assert!(!needs_load_time_quant(root, None).unwrap());
        assert!(needs_load_time_quant(root, Some(Quant::Q4)).unwrap());
        assert!(needs_load_time_quant(root, Some(Quant::Q8)).unwrap());
    }

    fn write_marker(dir: &Path, bits: i32, group_size: i32) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "quantization": { "bits": bits, "group_size": group_size }
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn a_packed_tier_is_selected_exactly_and_a_mismatch_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_marker(&root.join("transformer"), 4, GROUP_SIZE);
        write_marker(&root.join("text_encoder"), 8, GROUP_SIZE);

        assert_eq!(installed_tier(root).unwrap(), Tier::Q4);
        assert!(
            !needs_load_time_quant(root, Some(mlx_gen::Quant::Q4)).unwrap(),
            "a matching request loads packed with no quantize pass"
        );
        let err = needs_load_time_quant(root, Some(mlx_gen::Quant::Q8))
            .unwrap_err()
            .to_string();
        assert!(err.contains("silently serve q4"), "{err}");
        let err = needs_load_time_quant(root, None).unwrap_err().to_string();
        assert!(err.contains("no quantization was requested"), "{err}");
    }

    #[test]
    fn a_tier_no_converter_produces_and_a_foreign_group_size_are_hard_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Q4 DiT with a Q4 tower: below the declared floor, so no installable tier names it.
        write_marker(&root.join("transformer"), 4, GROUP_SIZE);
        write_marker(&root.join("text_encoder"), 4, GROUP_SIZE);
        let err = installed_tier(root).unwrap_err().to_string();
        assert!(err.contains("no installable tier"), "{err}");

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_marker(&root.join("transformer"), 8, 32);
        write_marker(&root.join("text_encoder"), 8, GROUP_SIZE);
        let err = installed_tier(root).unwrap_err().to_string();
        assert!(err.contains("group_size"), "{err}");
    }
}
