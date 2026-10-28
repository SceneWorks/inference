//! Guarded counterparts of the retained encoder receipt's EXACT metadata predicates.
use super::*;
use crate::source_inventory::{
    bounded_clone::{scalar, sum, CloneLaw},
    bounded_fs as fs,
};

type Check<'a> = &'a mut dyn FnMut(u64) -> Result<()>;
fn changed<T>(next: Check<'_>) -> Result<T> {
    crate::SourceInventory::metadata_gap(
        next,
        "prepared text-encoder receipt changed after validation",
    )
}
impl CloneLaw for PreparedEncoderLoadReceipt {
    fn clone_bytes(&self) -> Result<u64> {
        let Self {
            requested_source,
            pinned,
            tokenizer,
        } = self;
        sum([
            requested_source.clone_bytes(),
            pinned.clone_bytes(),
            tokenizer.clone_bytes(),
        ])
    }
}
impl CloneLaw for PinnedEncoderSource {
    fn clone_bytes(&self) -> Result<u64> {
        let Self {
            weights,
            shard_paths,
            shard_pins,
            config_candidate,
            config_pin,
        } = self;
        sum([
            weights.clone_bytes(),
            shard_paths.clone_bytes(),
            shard_pins.clone_bytes(),
            config_candidate.clone_bytes(),
            config_pin.clone_bytes(),
        ])
    }
}
impl CloneLaw for PinnedTokenizerArtifact {
    fn clone_bytes(&self) -> Result<u64> {
        let Self {
            pin,
            semantic_sha256,
        } = self;
        sum([pin.clone_bytes(), scalar(semantic_sha256)])
    }
}
impl CloneLaw for ValidatedTokenizerSource {
    fn clone_bytes(&self) -> Result<u64> {
        let Self {
            base,
            base_candidates,
            selected,
            selected_candidates,
            disposition,
        } = self;
        sum([
            base.clone_bytes(),
            base_candidates.clone_bytes(),
            selected.clone_bytes(),
            selected_candidates.clone_bytes(),
            scalar(disposition),
        ])
    }
}

impl PreparedEncoderLoadReceipt {
    pub(crate) fn ensure_unchanged_for_with_capacity(
        &self,
        selected_source: Option<&WeightsSource>,
        next: Check<'_>,
    ) -> Result<()> {
        if selected_source != Some(&self.requested_source) {
            return changed(next);
        }
        // Same resolve_source priority: a nested text_encoder/config.json selects
        // the nested directory; otherwise the original source shape is retained.
        match &self.requested_source {
            WeightsSource::File(path) => {
                if self.pinned.weights != self.requested_source {
                    return changed(next);
                }
                let _ = path;
            }
            WeightsSource::Dir(path) => {
                let nested = fs::join(path, std::ffi::OsStr::new("text_encoder"), next)?;
                let config = fs::join(&nested, std::ffi::OsStr::new("config.json"), next)?;
                if fs::is_file_at(&config, next)? {
                    if self.pinned.weights != WeightsSource::Dir(nested) {
                        return changed(next);
                    }
                } else if self.pinned.weights != self.requested_source {
                    return changed(next);
                }
            }
        }
        self.pinned.ensure_unchanged_with_capacity(next)?;
        if let Some(tokenizer) = &self.tokenizer {
            tokenizer.ensure_unchanged_with_capacity(next)?;
        }
        Ok(())
    }
}
impl PinnedEncoderSource {
    fn ensure_unchanged_with_capacity(&self, next: Check<'_>) -> Result<()> {
        match &self.weights {
            WeightsSource::File(path) => {
                let absolute = fs::absolute(path, next)?;
                if self.shard_paths.as_slice() != std::slice::from_ref(&absolute) {
                    return changed(next);
                }
            }
            WeightsSource::Dir(path) => {
                let absolute = fs::absolute(path, next)?;
                let mut seen = fs::vec::<bool>(self.shard_paths.len(), next)?;
                seen.resize(self.shard_paths.len(), false);
                let mut directory = fs::Directory::open(&absolute, next)?;
                while let Some(name) = directory.next(next)? {
                    let leaf = Path::new(name);
                    if leaf.extension().and_then(|extension| extension.to_str())
                        != Some("safetensors")
                        || crate::weightsmeta::is_hidden_file(leaf)
                    {
                        continue;
                    }
                    let Some(index) = self.shard_paths.iter().position(|expected| {
                        expected.parent() == Some(absolute.as_path())
                            && expected.file_name() == Some(name)
                    }) else {
                        return changed(next);
                    };
                    if !fs::is_file_at(&self.shard_paths[index], next)? {
                        return changed(next);
                    }
                    seen[index] = true;
                }
                if self.shard_paths.is_empty() || seen.iter().any(|seen| !seen) {
                    return changed(next);
                }
            }
        }
        let present = match &self.config_candidate {
            Some(path) => fs::is_file_at(path, next)?,
            None => false,
        };
        if present != self.config_pin.is_some() {
            return changed(next);
        }
        if let Some(pin) = &self.config_pin {
            pin.verify_unchanged_with_capacity(next)?;
        }
        for pin in &self.shard_pins {
            pin.verify_unchanged_with_capacity(next)?;
        }
        Ok(())
    }
}
impl ValidatedTokenizerSource {
    fn ensure_unchanged_with_capacity(&self, next: Check<'_>) -> Result<()> {
        fn selected(candidates: &[PathBuf], expected: &Path, next: Check<'_>) -> Result<()> {
            for path in candidates {
                if fs::is_file_at(path, next)? {
                    return if path == expected {
                        Ok(())
                    } else {
                        changed(next)
                    };
                }
            }
            changed(next)
        }
        selected(&self.base_candidates, self.base.pin.loader_path(), next)?;
        self.base.pin.verify_unchanged_with_capacity(next)?;
        if let Some(artifact) = &self.selected {
            let Some(candidates) = &self.selected_candidates else {
                return changed(next);
            };
            selected(candidates, artifact.pin.loader_path(), next)?;
            artifact.pin.verify_unchanged_with_capacity(next)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_inventory::guarded_allocation_tests::checked;

    fn prepared(root: &Path, directory: bool) -> crate::LoadSpec {
        let primary = root.join("primary");
        std::fs::write(&primary, b"primary").unwrap();
        let encoder = root.join("selected");
        std::fs::create_dir(&encoder).unwrap();
        let shard = encoder.join("weights.safetensors");
        std::fs::write(&shard, b"synthetic shard").unwrap();
        let config = encoder.join("config.json");
        std::fs::write(&config, b"{}").unwrap();
        let base = root.join("base-tokenizer.json");
        let selected = encoder.join("tokenizer.json");
        std::fs::write(&base, b"{}").unwrap();
        std::fs::write(&selected, b"{}").unwrap();
        let source = if directory {
            WeightsSource::Dir(encoder)
        } else {
            WeightsSource::File(shard)
        };
        let pinned = PinnedEncoderSource::pin(&source, Some(config.clone())).unwrap();
        let base_artifact = pin_tokenizer_artifact(&base, &[]).unwrap();
        let selected_artifact = pin_tokenizer_artifact(&selected, &[]).unwrap();
        let tokenizer = ValidatedTokenizerSource {
            base: base_artifact,
            base_candidates: vec![root.join("higher-priority.json"), base.clone()],
            selected: Some(selected_artifact),
            selected_candidates: Some(vec![selected.clone()]),
            disposition: EncoderTokenizerDisposition::MatchedSelectedTokenizer,
        };
        let receipt = PreparedEncoderLoadReceipt {
            requested_source: source.clone(),
            pinned,
            tokenizer: Some(tokenizer),
        };
        receipt.ensure_unchanged_for(Some(&source)).unwrap();
        let mut spec = crate::LoadSpec::new(WeightsSource::File(primary.clone()));
        spec.text_encoder = Some(source);
        let pins = [
            primary,
            config.clone(),
            base.clone(),
            selected.clone(),
            root.join("selected/weights.safetensors"),
        ]
        .into_iter()
        .map(PinnedWeightsFile::pin)
        .collect::<Result<Vec<_>>>()
        .unwrap();
        spec.prepare_with_validated_receipt_pins(pins, [config, base, selected], receipt)
            .unwrap();
        spec
    }

    #[test]
    fn guarded_prepared_encoder_full_clone_preserves_every_nested_owner_and_identity() {
        for directory in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut spec = prepared(root.path(), directory);
            let copy = checked(None, |next| spec.clone_with_capacity(next))
                .0
                .unwrap();
            assert_eq!(copy.prepared_file_pins(), spec.prepared_file_pins());
            checked(None, |next| {
                copy.validate_prepared_file_pins_with_capacity(next)
            })
            .0
            .unwrap();
            let inventory = checked(None, |next| {
                crate::SourceInventory::capture_with_capacity(&spec, next)
            })
            .0
            .unwrap();
            checked(None, |next| {
                spec.bind_source_inventory_with_capacity(inventory, next)
            })
            .0
            .unwrap();
            let mut ran = false;
            let read: Result<()> = checked(None, |next| {
                spec.read_prepared_files_unchanged_with_capacity(next, |_| {
                    ran = true;
                    Ok(())
                })
            })
            .0;
            read.unwrap();
            assert!(ran);
            // A higher-priority sibling tokenizer is outside the configured File
            // roots, but remains part of the prepared identity predicate.
            let read: Result<()> =
                spec.read_prepared_files_unchanged_with_capacity(&mut |_| Ok(()), |_| {
                    std::fs::write(root.path().join("higher-priority.json"), b"{}")?;
                    Err(Error::Canceled)
                });
            assert!(matches!(read, Err(Error::Unsupported(_))));
        }
    }

    #[test]
    fn guarded_prepared_encoder_rejects_direct_shard_and_sibling_config_changes() {
        for mutation in [
            "extra-shard",
            "missing-shard",
            "config-replacement",
            "higher-priority",
        ] {
            let root = tempfile::tempdir().unwrap();
            let spec = prepared(root.path(), true);
            match mutation {
                "extra-shard" => {
                    std::fs::write(root.path().join("selected/new.safetensors"), b"new").unwrap()
                }
                "missing-shard" => {
                    std::fs::remove_file(root.path().join("selected/weights.safetensors")).unwrap()
                }
                "config-replacement" => {
                    let replacement = root.path().join("replacement");
                    std::fs::write(&replacement, b"{}").unwrap();
                    std::fs::rename(replacement, root.path().join("selected/config.json")).unwrap();
                }
                "higher-priority" => {
                    std::fs::write(root.path().join("higher-priority.json"), b"{}").unwrap()
                }
                _ => unreachable!(),
            }
            assert!(
                spec.validate_prepared_file_pins().is_err(),
                "native {mutation}"
            );
            assert!(
                checked(None, |next| spec
                    .validate_prepared_file_pins_with_capacity(next))
                .0
                .is_err(),
                "guarded {mutation}"
            );
        }
    }
}
