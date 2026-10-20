//! Frozen MOSS file identities from the approved sc-26321 acquisition manifests.
//!
//! These path-free values authenticate supplied bytes; a repository directory name is never
//! evidence of its contents. Providers retain and recheck the same source pins while loading.

use crate::{Error, PinnedWeightsFile, Result};

/// One exact file of an approved MOSS snapshot. Sizes are physical file bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MossImageAssetIdentity {
    pub repository: &'static str,
    pub revision: &'static str,
    pub relative_path: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

/// Complete approved 35-file ordinary/mixed bundle plus six full-FP32 files.
pub const MOSS_IMAGE_ASSETS: &[MossImageAssetIdentity] = &[
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "1ceede80badddd6a7812069619edaa5633587192",
        relative_path: "LICENSE",
        bytes: 11358,
        sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "1ceede80badddd6a7812069619edaa5633587192",
        relative_path: "README.md",
        bytes: 8480,
        sha256: "5225ccaf3e4c8135353e5cf2c585290024c923079e688ea0b82dca5990b4d5b2",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "1ceede80badddd6a7812069619edaa5633587192",
        relative_path: "transformer/config.json",
        bytes: 240,
        sha256: "3291c3a1c7adc22b8182a0de25ca8ec97474d72715ce84adbab09cc0be24f666",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "1ceede80badddd6a7812069619edaa5633587192",
        relative_path: "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        bytes: 4991978016,
        sha256: "ae085fba123b6df67daf90aa9698fadee60c73e45bcaf3f3043d15f34e7a1804",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "1ceede80badddd6a7812069619edaa5633587192",
        relative_path: "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        bytes: 4983137016,
        sha256: "806e876d80c7fba560663a3a95580e6c7033da3f9ea3c52765245ef0e1818f75",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "1ceede80badddd6a7812069619edaa5633587192",
        relative_path: "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        bytes: 189549184,
        sha256: "5f319b72c40324294007c270d78112425627b537fcd8769b3c00f9f070e506a9",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "1ceede80badddd6a7812069619edaa5633587192",
        relative_path: "transformer/diffusion_pytorch_model.safetensors.index.json",
        bytes: 40298,
        sha256: "b609327fc75dbe5080e7e095d8b727d74d728b4f98e8b047133b0082cfa2fb03",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-Base",
        revision: "1ceede80badddd6a7812069619edaa5633587192",
        relative_path: "transformer/pipeline_moss_image.py",
        bytes: 25060,
        sha256: "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "LICENSE",
        bytes: 11358,
        sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "README.md",
        bytes: 8703,
        sha256: "bd254144a8163c88b5dcbd93746c99cb67bfabbd1c95daf584cdd623186b14f3",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "model_index.json",
        bytes: 412,
        sha256: "ca80b1f562a506e0eb24afb1acb464fb583163e3c240c04e4bc69fea090db173",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "pipeline_moss_image.py",
        bytes: 25060,
        sha256: "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "scheduler/pipeline_moss_image.py",
        bytes: 25060,
        sha256: "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "scheduler/scheduler_config.json",
        bytes: 494,
        sha256: "707322832cc507dfb8c51c0e0cd04d44ee2d30a9b036b6f3c47d6c17aa71adf0",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "text_encoder/config.json",
        bytes: 2813,
        sha256: "b31ee18a242bcfcccbe740f825f6b1d3ae231b0655eeecb5de3e98d504b7d68c",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "text_encoder/model-00001-of-00004.safetensors",
        bytes: 4973910744,
        sha256: "abfaa3d3a8612508c85ee8f6da9c734b862f10298b903765c4ad384177c32b0b",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "text_encoder/model-00002-of-00004.safetensors",
        bytes: 4988364288,
        sha256: "9112637fa6bf4d07bafd6ae268852b18d3d0c1242f4ee901c5579006e5bd4894",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "text_encoder/model-00003-of-00004.safetensors",
        bytes: 4954202112,
        sha256: "956f1ef51d0cfe610b0d22987c0b7821c36e3ce228abceccef523518bcf079dc",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "text_encoder/model-00004-of-00004.safetensors",
        bytes: 1869001848,
        sha256: "9e21d20e8adde88c3e0b65d4a0d251f143d98ee9d52db36b4348361091e4e208",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "text_encoder/model.safetensors.index.json",
        bytes: 64641,
        sha256: "0af6fa3fa36ba8dcbb6b7ad0325a36c5b1b69862a9a1f1486b1b80385bd8f5ee",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "tokenizer/chat_template.jinja",
        bytes: 7756,
        sha256: "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "tokenizer/tokenizer.json",
        bytes: 19989325,
        sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "tokenizer/tokenizer_config.json",
        bytes: 1123,
        sha256: "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "transformer/config.json",
        bytes: 240,
        sha256: "3291c3a1c7adc22b8182a0de25ca8ec97474d72715ce84adbab09cc0be24f666",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        bytes: 4991978016,
        sha256: "e40d3444940a0b0fd7cd45ba7959a5bd3d5efd7ddf87aae5ce4d87ce8aa03d01",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        bytes: 4983137016,
        sha256: "d106da790ec2924ffd8af04b4ffa8f8de281a974b58e55649c3ec596dc8ef033",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        bytes: 189549184,
        sha256: "11f637ab17c7093344a6110a1e87ffda8e56558dc55183d8e790222cfcdcf52d",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "transformer/diffusion_pytorch_model.safetensors.index.json",
        bytes: 40298,
        sha256: "b609327fc75dbe5080e7e095d8b727d74d728b4f98e8b047133b0082cfa2fb03",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "transformer/pipeline_moss_image.py",
        bytes: 25060,
        sha256: "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "vae/config.json",
        bytes: 838,
        sha256: "cc77f3f0ce0357039c71018dc19edaa4fcceb0c2f7488fdc2b9c0580d6baafa1",
    },
    MossImageAssetIdentity {
        repository: "OpenMOSS-Team/MOSS-Image-SFT",
        revision: "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        relative_path: "vae/diffusion_pytorch_model.safetensors",
        bytes: 249521372,
        sha256: "c3bdf94dc67ec69b30ea9fe1fa8340085679cae64d019f7c7e4a9921f1a7f17a",
    },
    MossImageAssetIdentity {
        repository: "lodestones/MOSS-Image-SFT-8-steps-dmd",
        revision: "5ff314b4c1470876b114b810146085db8a181a6e",
        relative_path: "README.md",
        bytes: 2338,
        sha256: "714b1072d9bd9667f6a6615164271d0eba78aad2930d3ccefa3b5741e09bb957",
    },
    MossImageAssetIdentity {
        repository: "lodestones/MOSS-Image-SFT-8-steps-dmd",
        revision: "5ff314b4c1470876b114b810146085db8a181a6e",
        relative_path: "dataspace.safetensors",
        bytes: 65728,
        sha256: "816cf216448000c9f525539adcc005a6b929d1bdb2e4e30e2fe82e094152dcb5",
    },
    MossImageAssetIdentity {
        repository: "lodestones/MOSS-Image-SFT-8-steps-dmd",
        revision: "5ff314b4c1470876b114b810146085db8a181a6e",
        relative_path: "lora.safetensors",
        bytes: 1189176800,
        sha256: "4abb14e39c1a59018907581e609f43616ec6810126144ec2b1d2e6f8969628fe",
    },
    MossImageAssetIdentity {
        repository: "lodestones/MOSS-Image-SFT-8-steps-dmd",
        revision: "5ff314b4c1470876b114b810146085db8a181a6e",
        relative_path: "time_embedding.safetensors",
        bytes: 41017896,
        sha256: "54f081f33f48f4abda3362cdc41abec947ddabd11894ce6158b15a9ff6340a3d",
    },
    MossImageAssetIdentity {
        repository: "lodestones/MOSS-Image-SFT-8-steps-dmd",
        revision: "5ff314b4c1470876b114b810146085db8a181a6e",
        relative_path: "transformer/config.json",
        bytes: 240,
        sha256: "3291c3a1c7adc22b8182a0de25ca8ec97474d72715ce84adbab09cc0be24f666",
    },
    MossImageAssetIdentity {
        repository: "lodestones/MOSS-Image-SFT-8-steps-dmd",
        revision: "5ff314b4c1470876b114b810146085db8a181a6e",
        relative_path: "transformer/diffusion_pytorch_model-00001-of-00004.safetensors",
        bytes: 5362224248,
        sha256: "4ca6f4fb468498d56322d71c517495a438fb5022782b44c04b43109773d86e67",
    },
    MossImageAssetIdentity {
        repository: "lodestones/MOSS-Image-SFT-8-steps-dmd",
        revision: "5ff314b4c1470876b114b810146085db8a181a6e",
        relative_path: "transformer/diffusion_pytorch_model-00002-of-00004.safetensors",
        bytes: 5360639512,
        sha256: "f910d5754e26938be4e645cb1cb4eebf6a0fd5c847c0a085d54b879caaa87254",
    },
    MossImageAssetIdentity {
        repository: "lodestones/MOSS-Image-SFT-8-steps-dmd",
        revision: "5ff314b4c1470876b114b810146085db8a181a6e",
        relative_path: "transformer/diffusion_pytorch_model-00003-of-00004.safetensors",
        bytes: 5234810496,
        sha256: "de9b35419cbd02169409d19daa8089dc2510bbd30d10828011db0882c6918fab",
    },
    MossImageAssetIdentity {
        repository: "lodestones/MOSS-Image-SFT-8-steps-dmd",
        revision: "5ff314b4c1470876b114b810146085db8a181a6e",
        relative_path: "transformer/diffusion_pytorch_model-00004-of-00004.safetensors",
        bytes: 4371606768,
        sha256: "b868a4e566f455bc1ba1fca92e4d3ee16809338e7de575c9e1512ef786c9fd8d",
    },
    MossImageAssetIdentity {
        repository: "lodestones/MOSS-Image-SFT-8-steps-dmd",
        revision: "5ff314b4c1470876b114b810146085db8a181a6e",
        relative_path: "transformer/diffusion_pytorch_model.safetensors.index.json",
        bytes: 40297,
        sha256: "a8f42088bfeb12511db9354ca4d5cc21c1c4600c3df955f18df87f85d1f910bb",
    },
];

/// Look up an exact identity; unknown repositories, revisions and paths never fall back.
pub fn moss_image_asset(
    repository: &str,
    revision: &str,
    relative_path: &str,
) -> Option<&'static MossImageAssetIdentity> {
    MOSS_IMAGE_ASSETS.iter().find(|asset| {
        asset.repository == repository
            && asset.revision == revision
            && asset.relative_path == relative_path
    })
}

/// Authenticate the actual retained source pin before tensor allocation/materialization.
/// A token proves stable bytes; this additional check proves those bytes belong to the frozen model.
pub fn verify_moss_image_asset(
    pin: &PinnedWeightsFile,
    repository: &str,
    revision: &str,
    relative_path: &str,
) -> Result<()> {
    let expected = moss_image_asset(repository, revision, relative_path).ok_or_else(|| {
        Error::Unsupported(format!(
            "unknown frozen MOSS asset {repository}@{revision}/{relative_path}"
        ))
    })?;
    pin.verify_unchanged()?;
    let digest: String = pin
        .content_sha256()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if pin.target_fingerprint().size != expected.bytes || digest != expected.sha256 {
        return Err(Error::Unsupported(format!("MOSS asset content mismatch for {repository}@{revision}/{relative_path}; provision the exact frozen file")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moss_image::{
        BASE_REPO, BASE_REVISION, DMD_REPO, DMD_REVISION, SFT_REPO, SFT_REVISION,
    };
    use std::collections::BTreeSet;

    #[test]
    fn complete_unique_manifest_has_only_exact_frozen_sources() {
        assert_eq!(MOSS_IMAGE_ASSETS.len(), 41);
        let mut unique = BTreeSet::new();
        for a in MOSS_IMAGE_ASSETS {
            assert!(unique.insert((a.repository, a.revision, a.relative_path)));
            assert!(a.bytes > 0);
            assert_eq!(a.sha256.len(), 64);
            assert!(a.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
            assert!(!a.relative_path.starts_with('/'));
            assert!(!a.relative_path.split('/').any(|p| p == ".."));
            assert!([
                (BASE_REPO, BASE_REVISION),
                (SFT_REPO, SFT_REVISION),
                (DMD_REPO, DMD_REVISION)
            ]
            .contains(&(a.repository, a.revision)));
        }
        assert!(moss_image_asset(SFT_REPO, "main", "transformer/config.json").is_none());
        assert!(moss_image_asset(BASE_REPO, SFT_REVISION, "transformer/config.json").is_none());
        assert!(moss_image_asset(SFT_REPO, SFT_REVISION, "../config.json").is_none());
    }

    #[test]
    fn authenticates_small_file_and_refuses_mutation_foreign_bytes_and_crossed_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("config.json");
        std::fs::write(
            &file,
            include_bytes!("../tests/fixtures/moss_image/transformer_config.json"),
        )
        .unwrap();
        let pin = PinnedWeightsFile::pin(&file).unwrap();
        verify_moss_image_asset(&pin, SFT_REPO, SFT_REVISION, "transformer/config.json").unwrap();
        assert!(
            verify_moss_image_asset(&pin, DMD_REPO, DMD_REVISION, "dataspace.safetensors").is_err()
        );
        assert!(
            verify_moss_image_asset(&pin, SFT_REPO, "main", "transformer/config.json").is_err()
        );
        std::fs::write(&file, b"foreign bytes").unwrap();
        assert!(
            verify_moss_image_asset(&pin, SFT_REPO, SFT_REVISION, "transformer/config.json")
                .is_err()
        );
        let foreign = PinnedWeightsFile::pin(&file).unwrap();
        assert!(verify_moss_image_asset(
            &foreign,
            SFT_REPO,
            SFT_REVISION,
            "transformer/config.json"
        )
        .is_err());
    }
    #[test]
    fn independent_author_fixture_provenance_and_precision_discrimination_are_preserved() {
        use sha2::{Digest, Sha256};
        let base: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/moss_image/tiny_author/fixture.json"
        ))
        .unwrap();
        let dmd: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/moss_image/tiny_author/dmd_fixture.json"
        ))
        .unwrap();
        let source =
            include_bytes!("../tests/fixtures/moss_image/tiny_author/frozen_author_pipeline.py");
        assert_eq!(
            format!("{:x}", Sha256::digest(source)),
            crate::moss_image::PIPELINE_SHA256
        );
        assert_eq!(
            "6f202ab34de607fb7a1272cf50fc08e89f9a0ea21529b712c86f214dc467741a",
            dmd["reference_sha256"].as_str().unwrap()
        );
        assert_eq!(base["device"], "cpu");
        assert_eq!(base["cases"].as_array().unwrap().len(), 4);
        assert_eq!(dmd["synthetic_hyperparameters"], true);
        assert_eq!(
            dmd["runs"]["correct_widened"]["steps"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(dmd["projection_probe"]["correct"]["dtype"], "float32");
        assert_eq!(
            dmd["projection_probe"]["incorrect_cast_down"]["dtype"],
            "bfloat16"
        );
        for name in ["incorrect_cast_down", "incorrect_merged_bf16"] {
            assert!(
                dmd["incorrect_path_differences"][name]["final_max_abs"]
                    .as_f64()
                    .unwrap()
                    > 0.0
            );
        }
    }
}
