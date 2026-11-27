//! Curated bundle-level consumer facades (`SW-6982`, `REG-5`, `REG-6`).
//!
//! Consumers (such as SceneWorks `sceneworks-worker` and `sceneworks-memory-adapter`) should
//! route all generative inference through [`crate::catalog()`] / `ProviderRegistry`, and consume
//! remaining non-generator utilities, weight converters, memory helpers, and calibration surfaces
//! through these organized bundle facades rather than importing `runtime_macos::providers::*` or
//! adding direct `mlx-gen-*` Cargo dependencies.

/// Non-generator computer-vision, face-analysis, segmentation, PiD decode, and geometry utilities.
pub mod utilities {
    pub use mlx_gen_catalog::providers::depth;
    pub use mlx_gen_catalog::providers::face;
    pub use mlx_gen_catalog::providers::iris::depth as iris_depth;
    pub use mlx_gen_catalog::providers::pid;
    pub use mlx_gen_catalog::providers::sam2;
    pub use mlx_gen_catalog::providers::sam3;

    /// Pure geometry, keypoint, OpenPose skeleton, and face-restore mask primitives shared with InstantID.
    pub mod instantid_geometry {
        pub use mlx_gen_catalog::providers::instantid::{
            draw_bodypose, draw_kps, face_box_from_keypoints, feather_mask, letterbox,
            normalize_keypoints, paste_alpha, square_fit, view_angle_kps, BodyPoint,
            ANGLE_SET_ORDER, FACE_RESTORE_PROMPT, NUM_BODY_KEYPOINTS, STICKWIDTH, VIEW_ANGLE_KPS,
        };
    }

    /// SenseNova understanding / chat-template / tokenizer helpers outside the `Generator` trait.
    pub mod sensenova {
        pub use mlx_gen_catalog::providers::sensenova::{
            load_raw, load_tokenizer, NeoChatConfig, T2iModel,
        };
    }
}

/// Offline/load-time weight converters, quantizers, and single-file import loaders.
pub mod converters {
    pub mod anima {
        pub use mlx_gen_catalog::providers::anima::{
            convert, is_dit_quant_target, quantize_anima_dit,
        };
    }
    pub mod bernini {
        pub use mlx_gen_catalog::providers::bernini::convert;
    }
    pub mod boogu {
        pub use mlx_gen_catalog::providers::boogu::convert;
    }
    pub mod chroma {
        pub use mlx_gen_catalog::providers::chroma::convert;
    }
    pub mod flux {
        pub use mlx_gen_catalog::providers::flux::convert;
    }
    pub mod flux2 {
        pub use mlx_gen_catalog::providers::flux2::{
            config, convert, convert_and_assemble, product_load, quantize_flux2_dit,
            quantize_flux2_text_encoder_dir,
        };
    }
    pub mod ideogram {
        pub use mlx_gen_catalog::providers::ideogram::convert;
    }
    pub mod kolors {
        pub use mlx_gen_catalog::providers::kolors::convert;
    }
    pub mod krea {
        pub use mlx_gen_catalog::providers::krea::{
            block_stream_diagnostics, convert, descriptor, load_control_from_native_dit_file,
            load_from_native_dit_file, multiphase, reset_block_stream_diagnostics,
            BlockStreamDiagnostics, Krea2Config, KreaTokenizer,
        };
    }
    pub mod krea_realtime {
        pub use mlx_gen_catalog::providers::krea_realtime::convert;
    }
    pub mod lens {
        pub use mlx_gen_catalog::providers::lens::convert;
    }
    pub mod ltx {
        pub use mlx_gen_catalog::providers::ltx::{
            convert, convert_and_assemble, convert_vae_components, pipeline, training,
            LtxConvertOpts,
        };
    }
    pub mod mage {
        pub use mlx_gen_catalog::providers::mage::{
            convert, descriptor_for, load_finetuned, memory, model, training, MageVariant,
            COMPONENT_TEXT_ENCODER,
        };
    }
    pub mod minimax_h3 {
        pub use mlx_gen_catalog::providers::minimax_h3::convert;
    }
    pub mod mochi {
        pub use mlx_gen_catalog::providers::mochi::convert;
    }
    pub mod sana {
        pub use mlx_gen_catalog::providers::sana::convert;
    }
    pub mod scail2 {
        pub use mlx_gen_catalog::providers::scail2::{convert, has_diff_patch_keys};
    }
    pub mod sd3 {
        pub use mlx_gen_catalog::providers::sd3::{convert, quantize_sd3_dir, Sd3Arch, Sd3Variant};
    }
    pub mod sdxl {
        pub use mlx_gen_catalog::providers::sdxl::convert;
    }
    pub mod seedvr2 {
        pub use mlx_gen_catalog::providers::seedvr2::{convert, video};
    }
    pub mod sensenova {
        pub use mlx_gen_catalog::providers::sensenova::convert;
    }
    pub mod wan {
        pub use mlx_gen_catalog::providers::wan::{convert, pipeline, product_load, WanVae};
    }
    pub mod z_image {
        pub use mlx_gen_catalog::providers::z_image::convert;
    }
}

/// Family memory-strategy, decode-tiling, and residency admission helpers.
pub mod memory_helpers {
    pub use mlx_gen_catalog::BESPOKE_UTILITY_CRATES;

    pub mod anima {
        pub use mlx_gen_catalog::providers::anima::memory_strategy;
    }
    pub mod bernini {
        pub use mlx_gen_catalog::providers::bernini::{memory_strategy, vae_tiling};
    }
    pub mod chroma {
        pub use mlx_gen_catalog::providers::chroma::memory_strategy;
    }
    pub mod flux {
        pub use mlx_gen_catalog::providers::flux::memory_strategy;
    }
    pub mod flux2 {
        pub use mlx_gen_catalog::providers::flux2::memory_strategy;
    }
    pub mod flux3 {
        pub use mlx_gen_catalog::providers::flux3::memory_strategy;
    }
    pub mod ideogram {
        pub use mlx_gen_catalog::providers::ideogram::memory_strategy;
    }
    pub mod instantid {
        pub use mlx_gen_catalog::providers::instantid::memory_strategy;
    }
    pub mod iris {
        pub use mlx_gen_catalog::providers::iris::memory_strategy;
    }
    pub mod kolors {
        pub use mlx_gen_catalog::providers::kolors::memory_strategy;
    }
    pub mod krea {
        pub use mlx_gen_catalog::providers::krea::memory_strategy;
    }
    pub mod krea_realtime {
        pub use mlx_gen_catalog::providers::krea_realtime::{memory_strategy, vae_tiling};
    }
    pub mod lens {
        pub use mlx_gen_catalog::providers::lens::memory_strategy;
    }
    pub mod ltx {
        pub use mlx_gen_catalog::providers::ltx::{
            memory_strategy, memory_strategy_2_5, vae_tiling,
        };
    }
    pub mod mage {
        pub use mlx_gen_catalog::providers::mage::memory;
    }
    pub mod minimax_h3 {
        pub use mlx_gen_catalog::providers::minimax_h3::{
            denoise, h3_image, memory_strategy, pipeline, text_encoder,
        };
    }
    pub mod moss_image {
        pub use mlx_gen_catalog::providers::moss_image::{memory_strategy, training};
    }
    pub mod pulid {
        pub use mlx_gen_catalog::providers::pulid::memory_strategy;
    }
    pub mod qwen_image {
        pub use mlx_gen_catalog::providers::qwen_image::memory_strategy;
    }
    pub mod qwen_image_2_1 {
        pub use mlx_gen_catalog::providers::qwen_image_2_1::memory_strategy;
    }
    pub mod sana {
        pub use mlx_gen_catalog::providers::sana::memory_strategy;
    }
    pub mod scail2 {
        pub use mlx_gen_catalog::providers::scail2::{memory_strategy, vae_tiling};
    }
    pub mod sd3 {
        pub use mlx_gen_catalog::providers::sd3::memory_strategy;
    }
    pub mod sdxl {
        pub use mlx_gen_catalog::providers::sdxl::memory_strategy;
    }
    pub mod sensenova {
        pub use mlx_gen_catalog::providers::sensenova::memory_strategy;
    }
    pub mod supra2 {
        pub use mlx_gen_catalog::providers::supra2::memory_strategy;
    }
    pub mod svd {
        pub use mlx_gen_catalog::providers::svd::memory_strategy;
    }
    pub mod wan {
        pub use mlx_gen_catalog::providers::wan::{
            i2v_memory_strategy, memory_strategy, vae_tiling,
        };
    }
    pub mod z_image {
        pub use mlx_gen_catalog::providers::z_image::memory_strategy;
    }
}

/// Calibration and memory-adapter harness helpers (`sceneworks-memory-adapter`).
pub mod calibration {
    pub mod instantid {
        pub use mlx_gen_catalog::providers::instantid::{
            memory_strategy, InstantId, InstantIdPaths, InstantIdRequest, DEFAULT_CONTROLNET_SCALE,
            DEFAULT_IP_SCALE, DEFAULT_OPENPOSE_SCALE,
        };
    }
    pub mod pulid {
        pub use mlx_gen_catalog::providers::pulid::{memory_strategy, pulid_flux};
    }
    pub mod sensenova {
        pub use mlx_gen_catalog::providers::sensenova::{
            load_raw, load_tokenizer, memory_strategy, NeoChatConfig, T2iModel,
        };
    }
    pub use crate::memory_helpers::{
        anima, bernini, chroma, flux, flux2, flux3, ideogram, iris, kolors, krea, krea_realtime,
        lens, ltx, mage, minimax_h3, moss_image, qwen_image, qwen_image_2_1, sana, scail2, sd3,
        sdxl, supra2, svd, wan, z_image,
    };
}
