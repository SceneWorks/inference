//! Curated bundle-level consumer facades (`SW-6982`, `REG-5`, `REG-6`).
//!
//! Consumers (such as SceneWorks `sceneworks-worker` and `sceneworks-memory-adapter`) should
//! route all generative inference through [`crate::catalog()`] / `ProviderRegistry`, and consume
//! remaining non-generator utilities, weight converters, memory helpers, and calibration surfaces
//! through these organized bundle facades rather than importing `runtime_cuda::providers::*` or
//! adding direct `candle-gen-*` Cargo dependencies.

/// Non-generator computer-vision, face-analysis, segmentation, PiD decode, and geometry utilities.
pub mod utilities {
    pub use candle_gen_catalog::providers::depth;
    pub use candle_gen_catalog::providers::face;
    pub use candle_gen_catalog::providers::iris::depth as iris_depth;
    pub use candle_gen_catalog::providers::pid;
    pub use candle_gen_catalog::providers::sam3;

    /// Pure geometry, keypoint, OpenPose skeleton, and face-restore mask primitives shared with InstantID.
    pub mod instantid_geometry {
        pub use candle_gen_catalog::providers::instantid::{
            draw_bodypose, draw_kps, face_box_from_keypoints, feather_mask, letterbox,
            normalize_keypoints, paste_alpha, square_fit, view_angle_kps, BodyPoint,
            ANGLE_SET_ORDER, FACE_RESTORE_PROMPT, NUM_BODY_KEYPOINTS, STICKWIDTH, VIEW_ANGLE_KPS,
        };
    }

    /// Krea tokenizer and hires encoder weight helpers outside the `Generator` trait.
    pub mod krea {
        pub use candle_gen_catalog::providers::krea::{hires_encoder_weights, KreaTokenizer};
    }

    /// SenseNova understanding / multimodal model helpers outside the `Generator` trait.
    pub mod sensenova {
        pub use candle_gen_catalog::providers::sensenova::T2iModel;
    }
}

/// Offline/load-time weight converters, quantizers, and single-file import loaders.
pub mod converters {
    pub mod anima {
        pub use candle_gen_catalog::providers::anima::loader;
    }
    pub mod bernini {
        pub use candle_gen_catalog::providers::bernini::convert;
    }
    pub mod flux2 {
        pub use candle_gen_catalog::providers::flux2::{
            convert, convert_and_assemble, load_from_comfyui_dit,
        };
    }
    pub mod flux3 {
        pub use candle_gen_catalog::providers::flux3::convert;
    }
    pub mod krea {
        pub use candle_gen_catalog::providers::krea::{
            actual_quant_tier, control, convert, hires_encoder_weights, phase_adapter_memory,
            Krea2Config, KreaTokenizer, PhaseAdapterMemory, DEFAULT_CONTROL_SCALE,
            KREA_HIRES_ATTENTION_SCORE_BUDGET,
        };
    }
    pub mod ltx {
        pub use candle_gen_catalog::providers::ltx::training;
    }
    pub mod minimax_h3 {
        pub use candle_gen_catalog::providers::minimax_h3::convert;
    }
    pub mod qwen_image {
        pub use candle_gen_catalog::providers::qwen_image::load_from_comfyui_dit;
    }
    pub mod scail2 {
        pub use candle_gen_catalog::providers::scail2::{
            has_diff_patch_keys, snapshot_layout, SHARED_TIER_FILES,
        };
    }
    pub mod seedvr2 {
        pub use candle_gen_catalog::providers::seedvr2::{convert, video};
    }
    pub mod wan {
        pub use candle_gen_catalog::providers::wan::{vae16, wan14b};
    }
    pub mod z_image {
        pub use candle_gen_catalog::providers::z_image::{
            load_from_comfyui_checkpoint, load_from_comfyui_components,
        };
    }
}

/// Family memory-strategy, decode-tiling, and residency admission helpers.
pub mod memory_helpers {
    pub use candle_gen_catalog::BESPOKE_UTILITY_CRATES;

    pub mod anima {
        pub use candle_gen_catalog::providers::anima::memory_strategy;
    }
    pub mod bernini {
        pub use candle_gen_catalog::providers::bernini::{memory_strategy, vae_tiling};
    }
    pub mod boogu {
        pub use candle_gen_catalog::providers::boogu::memory_strategy;
    }
    pub mod flux {
        pub use candle_gen_catalog::providers::flux::memory_strategy;
    }
    pub mod flux2 {
        pub use candle_gen_catalog::providers::flux2::memory_strategy;
    }
    pub mod flux3 {
        pub use candle_gen_catalog::providers::flux3::memory_strategy;
    }
    pub mod instantid {
        pub use candle_gen_catalog::providers::instantid::memory_strategy;
    }
    pub mod iris {
        pub use candle_gen_catalog::providers::iris::memory_strategy;
    }
    pub mod kolors {
        pub use candle_gen_catalog::providers::kolors::memory_strategy;
    }
    pub mod ltx {
        pub use candle_gen_catalog::providers::ltx::{
            memory_strategy, memory_strategy_2_5, vae_tiling,
        };
    }
    pub mod mage {
        pub use candle_gen_catalog::providers::mage::memory_strategy;
    }
    pub mod minimax_h3 {
        pub use candle_gen_catalog::providers::minimax_h3::memory_strategy;
    }
    pub mod moss_image {
        pub use candle_gen_catalog::providers::moss_image::{memory_strategy, training};
    }
    pub mod pulid {
        pub use candle_gen_catalog::providers::pulid::memory_strategy;
    }
    pub mod qwen_image {
        pub use candle_gen_catalog::providers::qwen_image::MEMORY_CALIBRATION_FINGERPRINT;
    }
    pub mod qwen_image_2_1 {
        pub use candle_gen_catalog::providers::qwen_image_2_1::memory_strategy;
    }
    pub mod scail2 {
        pub use candle_gen_catalog::providers::scail2::{memory_strategy, vae_tiling};
    }
    pub mod sd3 {
        pub use candle_gen_catalog::providers::sd3::memory_strategy;
    }
    pub mod sdxl {
        pub use candle_gen_catalog::providers::sdxl::{
            flash_attn_enabled, provider_contract_for_spec, set_flash_attn, vae_tiling_enabled,
            SDXL_ROUTES,
        };
    }
    pub mod sensenova {
        pub use candle_gen_catalog::providers::sensenova::memory_strategy;
    }
    pub mod supra2 {
        pub use candle_gen_catalog::providers::supra2::memory_strategy;
    }
    pub mod wan {
        pub use candle_gen_catalog::providers::wan::{memory_strategy, vae_tiling};
    }
    pub mod z_image {
        pub use candle_gen_catalog::providers::z_image::{accel_attn_enabled, set_accel_attn};
    }
}

/// Calibration and memory-adapter harness helpers (`sceneworks-memory-adapter`).
pub mod calibration {
    pub mod instantid {
        pub use candle_gen_catalog::providers::instantid::{
            memory_strategy, InstantId, InstantIdPaths, InstantIdRequest, SdxlComponents,
            COMPONENT_TOKENIZER_CLIP_BIGG, COMPONENT_TOKENIZER_CLIP_L, COMPONENT_VAE_FP16_FIX,
            DEFAULT_CONTROLNET_SCALE, DEFAULT_IP_SCALE, DEFAULT_OPENPOSE_SCALE,
            REQUIRED_COMPONENTS,
        };
    }
    pub mod pulid {
        pub use candle_gen_catalog::providers::pulid::{
            compile_pulid_plan, memory_strategy, pulid_flux, pulid_load_spec, PulidFlux,
            PulidFluxPaths, PulidFluxRequest, DEFAULT_ID_WEIGHT,
        };
    }
    pub mod kolors {
        pub use candle_gen_catalog::providers::kolors::{
            memory_strategy, IpAdapterKolors, IpAdapterKolorsPaths, IpAdapterKolorsRequest,
            KolorsControl, KolorsControlPaths, KolorsControlRequest,
        };
    }
    pub mod flux {
        pub use candle_gen_catalog::providers::flux::{
            memory_strategy, Flux1ControlPaths, Flux1ControlRequest, Flux1DevControl,
            IpAdapterFlux, IpAdapterFluxPaths, IpAdapterFluxRequest,
        };
    }
    pub mod flux2 {
        pub use candle_gen_catalog::providers::flux2::{
            memory_strategy, Flux2Control, Flux2ControlPaths, Flux2ControlRequest, Flux2Edit,
            Flux2EditPaths, Flux2EditRequest,
        };
    }
    pub mod qwen_image {
        pub use candle_gen_catalog::providers::qwen_image::{
            QwenEdit, QwenEditPaths, QwenEditRequest, QwenFunControl, QwenFunControlPaths,
            QwenFunControlRequest, MEMORY_CALIBRATION_FINGERPRINT,
        };
    }
    pub mod sdxl {
        pub use candle_gen_catalog::providers::sdxl::{
            flash_attn_enabled, provider_contract_for_spec, set_flash_attn, IpAdapterSdxl,
            IpAdapterSdxlPaths, IpAdapterSdxlRequest, SdxlDetail, SdxlDetailPaths,
            SdxlDetailRequest, SdxlEdit, SdxlEditPaths, SdxlEditRequest, SDXL_ROUTES,
        };
    }
    pub mod z_image {
        pub use candle_gen_catalog::providers::z_image::{
            ZImageControl, ZImageControlPaths, ZImageControlRequest, ZImageEdit, ZImageEditPaths,
            ZImageEditRequest,
        };
    }
    pub use crate::memory_helpers::{
        anima, bernini, boogu, flux3, iris, ltx, mage, minimax_h3, moss_image, qwen_image_2_1,
        scail2, sd3, sensenova, supra2, wan,
    };
}
