use super::*;
use std::collections::HashSet;
use std::io::Read;
use std::sync::{Barrier, Mutex};

#[test]
fn quant_tier_labels_name_the_dense_tier_bf16_and_only_affine_tiers_partially() {
    assert_eq!(Quant::tier_label(None), "bf16");
    for quant in [Quant::Q4, Quant::Q8, Quant::Nvfp4, Quant::Fp8Rowwise] {
        assert_eq!(Quant::tier_label(Some(quant)), quant.key());
    }
    assert_eq!(Quant::affine_tier_label(None), Some("bf16"));
    assert_eq!(Quant::affine_tier_label(Some(Quant::Q4)), Some("q4"));
    assert_eq!(Quant::affine_tier_label(Some(Quant::Q8)), Some("q8"));
    assert_eq!(Quant::affine_tier_label(Some(Quant::Nvfp4)), None);
    assert_eq!(Quant::affine_tier_label(Some(Quant::Fp8Rowwise)), None);
}

#[test]
fn artifact_seal_progress_reports_actual_cold_memo_and_partial_cancel_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let size = (HASH_READ_BYTES * 3 + 17) as u64;
    let path = dir.path().join("observed-cold.synthetic");
    File::create(&path).unwrap().set_len(size).unwrap();
    let mut cold = Vec::new();
    let seal = ArtifactSeal::pin_with_progress(&path, || Ok(()), |e| cold.push(e.clone())).unwrap();
    assert_eq!(
        cold[0].acquisition,
        ArtifactSealAcquisition::ColdContentRead
    );
    assert_eq!(cold[0].state, ArtifactSealProgressState::Reading);
    assert_eq!(cold[0].hashed_bytes, HASH_READ_BYTES as u64);
    assert!(cold
        .iter()
        .all(|e| e.loader_path == path && e.source_bytes == Some(size)));
    assert_eq!(
        cold.last().unwrap().state,
        ArtifactSealProgressState::Acquired
    );
    assert_eq!(cold.last().unwrap().hashed_bytes, size);
    assert!(cold
        .iter()
        .any(|e| e.state == ArtifactSealProgressState::ReadComplete && e.hashed_bytes == size));
    assert!(cold
        .windows(2)
        .all(|w| w[0].hashed_bytes <= w[1].hashed_bytes));
    let work = test_full_hash_work_count();
    let mut memo = Vec::new();
    let again =
        ArtifactSeal::pin_with_progress(&path, || Ok(()), |e| memo.push(e.clone())).unwrap();
    assert_eq!(again.content_sha256(), seal.content_sha256());
    assert_eq!(test_full_hash_work_count(), work);
    assert_eq!(
        memo,
        vec![seal_progress(
            &path,
            ArtifactSealAcquisition::MemoryMemo,
            ArtifactSealProgressState::Acquired,
            0,
            Some(size)
        )]
    );
    for chunks in [1, 2] {
        let path = dir
            .path()
            .join(format!("observed-cancel-{chunks}.synthetic"));
        File::create(&path).unwrap().set_len(size).unwrap();
        let start_bytes = THREAD_HASH_BYTES_READ.with(Cell::get);
        let cancel = Cell::new(false);
        let mut events = Vec::new();
        let result = ArtifactSeal::pin_with_progress(
            &path,
            || {
                if cancel.get()
                    || (chunks == 2
                        && THREAD_HASH_BYTES_READ.with(Cell::get) - start_bytes
                            >= 2 * HASH_READ_BYTES)
                {
                    Err(crate::Error::Canceled)
                } else {
                    Ok(())
                }
            },
            |e| {
                if chunks == 1
                    && e.state == ArtifactSealProgressState::Reading
                    && e.hashed_bytes > 0
                {
                    cancel.set(true); // A consumer can cancel on actual progress, not callback counts.
                }
                events.push(e.clone());
            },
        );
        assert!(matches!(result, Err(crate::Error::Canceled)));
        assert_eq!(
            THREAD_HASH_BYTES_READ.with(Cell::get) - start_bytes,
            chunks * HASH_READ_BYTES
        );
        let last = events.last().unwrap();
        assert_eq!(last.state, ArtifactSealProgressState::Canceled);
        assert_eq!(last.acquisition, ArtifactSealAcquisition::ColdContentRead);
        assert_eq!(
            last.hashed_bytes,
            (chunks * HASH_READ_BYTES) as u64,
            "cancel bytes must include unreported chunks, not repeat last periodic progress"
        );
        assert_eq!(last.source_bytes, Some(size));
        assert!(!events
            .iter()
            .any(|e| e.state == ArtifactSealProgressState::Acquired));
    }
}

#[test]
fn artifact_seal_progress_names_real_persisted_hit_or_actual_cold_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("digests");
    let path = dir.path().join("persisted.synthetic");
    std::fs::write(&path, b"synthetic source bytes").unwrap();
    let first = ArtifactSeal::pin_with_store(&path, Some(&store)).unwrap();
    forget_proven_digests_for(first.loader_path());
    let before = test_full_hash_work_count();
    let mut events = Vec::new();
    let second =
        ArtifactSeal::pin_with_store_and_check(&path, Some(&store), &mut || Ok(()), &mut |e| {
            events.push(e.clone())
        })
        .unwrap();
    assert_eq!(first.content_sha256(), second.content_sha256());
    let persisted = persists_on_this_volume(dir.path());
    let last = events.last().unwrap();
    assert_eq!(last.state, ArtifactSealProgressState::Acquired);
    assert_eq!(
        last.acquisition,
        if persisted {
            ArtifactSealAcquisition::PersistedMemo
        } else {
            ArtifactSealAcquisition::ColdContentRead
        }
    );
    assert_eq!(last.hashed_bytes, if persisted { 0 } else { 22 });
    if persisted {
        assert_eq!(events.len(), 1);
        assert_eq!(test_full_hash_work_count(), before);
    } else {
        assert!(test_full_hash_work_count() > before);
        assert!(events
            .iter()
            .any(|e| e.state == ArtifactSealProgressState::ReadComplete));
    }
}

#[test]
fn artifact_seal_progress_read_completion_and_drift_audit_never_authorize_changed_source() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("observed-drift.synthetic");
    let size = (HASH_READ_BYTES * 3 + 17) as u64;
    File::create(&path).unwrap().set_len(size).unwrap();
    let mut events = Vec::new();
    let mut changed = false;
    let result = ArtifactSeal::pin_with_progress(
        &path,
        || Ok(()),
        |e| {
            if !changed
                && e.acquisition == ArtifactSealAcquisition::ColdContentRead
                && e.state == ArtifactSealProgressState::Reading
            {
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(&path)
                    .unwrap()
                    .write_all(&[0xa5])
                    .unwrap();
                changed = true;
            }
            events.push(e.clone());
        },
    );
    assert!(matches!(result, Err(crate::Error::Unsupported(_))));
    assert!(
        events.iter().any(
            |e| e.acquisition == ArtifactSealAcquisition::ColdContentRead
                && e.state == ArtifactSealProgressState::ReadComplete
                && e.hashed_bytes == size + 1
                && e.source_bytes == Some(size)
        ),
        "actual hashed bytes must not be replaced by the declared source size"
    );
    assert!(events
        .iter()
        .any(|e| e.acquisition == ArtifactSealAcquisition::DriftAudit && e.hashed_bytes > 0));
    assert_eq!(
        events.last().unwrap().state,
        ArtifactSealProgressState::Failed
    );
    assert!(!events
        .iter()
        .any(|e| e.state == ArtifactSealProgressState::Acquired));
}

#[test]
fn artifact_seal_progress_metadata_cancel_does_not_claim_a_cold_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("never-opened.synthetic");
    let before = test_full_hash_work_count();
    let bytes = THREAD_HASH_BYTES_READ.with(Cell::get);
    let mut events = Vec::new();
    let result = ArtifactSeal::pin_with_progress(
        &path,
        || Err(crate::Error::Canceled),
        |e| events.push(e.clone()),
    );
    assert!(matches!(result, Err(crate::Error::Canceled)));
    assert_eq!(
        events,
        vec![seal_progress(
            &path,
            ArtifactSealAcquisition::Metadata,
            ArtifactSealProgressState::Canceled,
            0,
            None
        )]
    );
    assert_eq!(test_full_hash_work_count(), before);
    assert_eq!(THREAD_HASH_BYTES_READ.with(Cell::get), bytes);
}

#[test]
fn checked_artifact_pin_canceled_chunk_with_drift_never_runs_second_audit() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cancel-drift.synthetic");
    File::create(&path)
        .unwrap()
        .set_len((HASH_READ_BYTES * 3 + 17) as u64)
        .unwrap();
    let original_identity = SealIdentity {
        loader_path: std::path::absolute(&path).unwrap(),
        canonical_target_path: std::fs::canonicalize(&path).unwrap(),
        path_component_fingerprints: path_component_fingerprints(&path).unwrap(),
        entry_fingerprint: stat_fingerprint(&path, false).unwrap(),
        target_fingerprint: stat_fingerprint(&path, true).unwrap(),
    };
    let start_bytes = THREAD_HASH_BYTES_READ.with(Cell::get);
    let start_reads = test_full_hash_work_count();
    let mut canceled = false;
    let result = ArtifactSeal::pin_with_check(&path, || {
        if !canceled && THREAD_HASH_BYTES_READ.with(Cell::get) > start_bytes {
            assert_eq!(
                THREAD_HASH_BYTES_READ.with(Cell::get) - start_bytes,
                HASH_READ_BYTES
            );
            // Change content AND a fingerprint field after the first real checked read.
            std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap()
                .write_all(&[0xa5])
                .unwrap();
            canceled = true;
            return Err(crate::Error::Canceled);
        }
        // Deliberately one-shot: a later callback must not mask an ignored cancellation.
        Ok(())
    });
    assert!(canceled);
    assert!(matches!(result, Err(crate::Error::Canceled)));
    assert_eq!(
        test_full_hash_work_count() - start_reads,
        1,
        "a canceled read must never start a second drift audit"
    );
    assert_eq!(
        THREAD_HASH_BYTES_READ.with(Cell::get) - start_bytes,
        HASH_READ_BYTES,
        "neither the rest of the acquisition nor an unchecked audit may read after cancel"
    );
    assert!(
        proven_digest(&original_identity).is_none(),
        "no partial/canceled seal may reach the memo"
    );
}

#[test]
fn checked_seal_pre_and_post_drift_audits_check_chunks_and_never_authorize() {
    use std::io::Write;
    for pre_drift in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit-drift.synthetic");
        File::create(&path)
            .unwrap()
            .set_len((HASH_READ_BYTES * 3 + 17) as u64)
            .unwrap();
        let seal = ArtifactSeal::pin(&path).unwrap();
        let drift = || {
            std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap()
                .write_all(&[0xa5])
                .unwrap()
        };
        if pre_drift {
            drift();
        }
        let start_bytes = THREAD_HASH_BYTES_READ.with(Cell::get);
        let start_audits = seal.full_hash_work_count();
        let read_called = Cell::new(false);
        let mut canceled = false;
        let result = seal.read_unchanged_with_check(
            &mut || {
                if !canceled && THREAD_HASH_BYTES_READ.with(Cell::get) > start_bytes {
                    canceled = true;
                    return Err(crate::Error::Canceled);
                }
                Ok(())
            },
            &mut |_| {},
            |_, _, _| {
                read_called.set(true);
                drift();
                Ok(42)
            },
        );
        assert!(
            matches!(result, Err(crate::Error::Canceled)),
            "drift audit must propagate cancellation, not authorize source bytes"
        );
        assert_eq!(
            read_called.get(),
            !pre_drift,
            "pre-verification drift cannot open the requested source read"
        );
        assert!(canceled);
        assert_eq!(seal.full_hash_work_count() - start_audits, 1);
        assert_eq!(
            THREAD_HASH_BYTES_READ.with(Cell::get) - start_bytes,
            HASH_READ_BYTES,
            "the actual pre/post drift audit must stop at its first checked chunk"
        );
    }
}

#[test]
fn checked_artifact_pin_cancels_inside_bounded_hash_and_preserves_digest_memo() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.synthetic");
    let file = File::create(&path).unwrap();
    file.set_len((HASH_READ_BYTES * 2 + 17) as u64).unwrap();
    let mut checks = 0;
    let aborted = hash_file_sha256_with_check(&path, &mut || {
        checks += 1;
        if checks == 4 {
            Err(crate::Error::Canceled)
        } else {
            Ok(())
        }
    });
    assert!(
        matches!(aborted, Err(crate::Error::Canceled)),
        "cancel after first bounded chunk, not after whole file"
    );
    assert_eq!(checks, 4);
    let mut interrupted_checks = 0;
    let interrupted = ArtifactSeal::pin_with_check(&path, || {
        interrupted_checks += 1;
        if interrupted_checks == 7 {
            Err(crate::Error::Canceled)
        } else {
            Ok(())
        }
    });
    assert!(matches!(interrupted, Err(crate::Error::Canceled)));
    let after_interrupted = test_full_hash_work_count();
    let mut checks = 0;
    let first = ArtifactSeal::pin_with_check(&path, || {
        checks += 1;
        Ok(())
    })
    .unwrap();
    assert!(
        test_full_hash_work_count() > after_interrupted,
        "an interrupted pin must not publish a partial digest to the memo"
    );
    assert!(
        checks >= 11,
        "metadata and every bounded read must check cancellation"
    );
    assert_eq!(first.content_sha256(), &hash_file_sha256(&path).unwrap());
    let before = test_full_hash_work_count();
    let mut warm_checks = 0;
    let warm = ArtifactSeal::pin_with_check(&path, || {
        warm_checks += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(warm.content_sha256(), first.content_sha256());
    assert_eq!(
        test_full_hash_work_count(),
        before,
        "checked warm pin must retain once-at-acquisition digest reuse"
    );
    assert!(warm_checks >= 3, "warm memo still checks cancellation");
    assert!(matches!(
        ArtifactSeal::pin_with_check(&path, || Err(crate::Error::Canceled)),
        Err(crate::Error::Canceled)
    ));
    let mut warm_checks = 0;
    let canceled_warm = ArtifactSeal::pin_with_check(&path, || {
        warm_checks += 1;
        if warm_checks == 3 {
            Err(crate::Error::Canceled)
        } else {
            Ok(())
        }
    });
    assert!(matches!(canceled_warm, Err(crate::Error::Canceled)));
}

#[test]
fn load_shape_declaration_result_defaults_clones_hashes_and_validates() {
    let spec = LoadSpec::new(WeightsSource::Dir("snapshot".into()));
    assert_eq!(
        spec.load_shape_declaration_result,
        LoadShapeDeclarationResult::NotEvaluated
    );
    assert_eq!(spec.load_shape, LoadShape::EagerMaterialization);

    spec.validate_load_shape_declaration()
        .expect("default declaration state is valid");

    let eligible = spec.clone().with_eligible_load_shape_declaration();
    assert_eq!(
        eligible.load_shape_declaration_result,
        LoadShapeDeclarationResult::Eligible
    );
    assert_eq!(eligible.load_shape, LoadShape::EagerMaterialization);
    eligible
        .validate_load_shape_declaration()
        .expect("eligible declaration is valid while eager");

    let applied = spec.clone().with_applied_load_shape_declaration();
    assert_eq!(
        applied.load_shape_declaration_result,
        LoadShapeDeclarationResult::Applied
    );
    assert_eq!(applied.load_shape, LoadShape::DeferredMaterialization);
    applied
        .validate_load_shape_declaration()
        .expect("applied declaration is valid");
    assert_eq!(
        applied.clone().load_shape_declaration_result,
        LoadShapeDeclarationResult::Applied,
        "clone must carry declaration authority across a cold-load boundary"
    );

    let refused = spec.with_refused_load_shape_declaration();
    assert_eq!(
        refused.load_shape_declaration_result,
        LoadShapeDeclarationResult::Refused
    );
    assert_eq!(
        refused.load_shape,
        LoadShape::EagerMaterialization,
        "an authoritative refusal keeps the compatibility shape"
    );
    refused
        .validate_load_shape_declaration()
        .expect("refused declaration is valid");

    let cache_axes: HashSet<_> = [
        LoadShapeDeclarationResult::NotEvaluated,
        LoadShapeDeclarationResult::Eligible,
        LoadShapeDeclarationResult::Applied,
        LoadShapeDeclarationResult::Refused,
    ]
    .into_iter()
    .collect();
    assert_eq!(cache_axes.len(), 4);

    let mut contradictory_eligible = eligible;
    contradictory_eligible.load_shape = LoadShape::DeferredMaterialization;
    assert!(contradictory_eligible
        .validate_load_shape_declaration()
        .is_err());

    let mut contradictory_applied = applied;
    contradictory_applied.load_shape = LoadShape::EagerMaterialization;
    assert!(contradictory_applied
        .validate_load_shape_declaration()
        .is_err());

    let mut contradictory_refused = refused;
    contradictory_refused.load_shape = LoadShape::DeferredMaterialization;
    assert!(contradictory_refused
        .validate_load_shape_declaration()
        .is_err());
}

/// Re-pinning an unmutated file reuses the proven digest instead of re-reading it.
///
/// Nothing upstream pins once: a single image job acquires the same text encoder from at least
/// two independent pre-load queries before the loader runs. On an 8.3 GB bf16 encoder that
/// second read was ~20 s of SHA-256 against an idle disk — re-proving a page-cached file the
/// previous query had just proven.
///
/// Stated as a hash-work count rather than a duration: the fixture here is a few bytes, so only
/// the count can express the claim.
#[test]
fn a_second_pin_of_an_unchanged_file_reuses_the_proven_digest() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("encoder.safetensors");
    std::fs::write(&file, b"encoder weights").expect("write fixture");

    let before = test_full_hash_work_count();
    let first = ArtifactSeal::pin(&file).expect("pin regular file");
    let after_first = test_full_hash_work_count();
    assert_eq!(
        after_first - before,
        1,
        "the first pin of a file has nothing to reuse and must read it"
    );

    let second = ArtifactSeal::pin(&file).expect("re-pin the same unchanged file");
    assert_eq!(
        test_full_hash_work_count(),
        after_first,
        "a file whose whole fingerprint set is unchanged must not be re-read"
    );
    assert_eq!(
        first.content_sha256(),
        second.content_sha256(),
        "the reused digest must be the one the first pin proved"
    );
}

/// Concurrent first pins of one file read it once; the other callers reuse the memo.
#[test]
fn concurrent_first_pins_of_one_file_read_it_once() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("transformer.safetensors");
    std::fs::write(&file, vec![7_u8; 4 * 1024 * 1024]).expect("write fixture");
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let file = file.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let before = test_full_hash_work_count();
                let seal = ArtifactSeal::pin(&file).expect("pin regular file");
                (test_full_hash_work_count() - before, *seal.content_sha256())
            })
        })
        .collect();
    let results: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().expect("pin thread"))
        .collect();
    assert_eq!(results.iter().map(|(reads, _)| reads).sum::<usize>(), 1);
    assert!(results.iter().all(|(_, digest)| *digest == results[0].1));
}

/// A waiter that parked on another caller's read re-proves its captured identity before it
/// trusts the memo: a file rewritten while it waited is re-read, never sealed with the old
/// digest. MUTATION: drop the post-claim `ensure_unchanged` and the waiter returns `stale`.
#[test]
fn a_waiter_does_not_seal_a_file_rewritten_while_it_waited() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("enhancer.safetensors");
    std::fs::write(&file, b"original weights").expect("write fixture");
    let loader_path = std::path::absolute(&file).expect("absolute path");
    let identity = ArtifactSeal::capture_identity(&file)
        .expect("capture identity")
        .seal_identity();
    // Stand in for the first caller: hold the identity's flight while "hashing".
    let held = seal_flights()
        .claim(identity.clone(), || Ok(()))
        .expect("first claim");
    let attempts_before = seal_claim_attempts()
        .lock()
        .unwrap()
        .iter()
        .filter(|path| **path == loader_path)
        .count();
    let waiter = {
        let file = file.clone();
        std::thread::spawn(move || ArtifactSeal::pin(&file).map(|seal| *seal.content_sha256()))
    };
    while seal_claim_attempts()
        .lock()
        .unwrap()
        .iter()
        .filter(|path| **path == loader_path)
        .count()
        == attempts_before
    {
        std::thread::yield_now();
    }
    // The first caller "finishes" with the old identity's digest, then the file changes before
    // the parked waiter resumes.
    let stale = [0xAA_u8; 32];
    remember_proven_digest(identity, stale);
    std::fs::write(&file, b"REWRITTEN WEIGHT").expect("rewrite fixture");
    drop(held);
    let digest = waiter.join().expect("waiter thread").expect("re-pin");
    assert_ne!(
        digest, stale,
        "a waiter must not seal the pre-rewrite digest"
    );
    let expected: [u8; 32] = Sha256::digest(b"REWRITTEN WEIGHT").into();
    assert_eq!(digest, expected);
}

/// A holder that panics releases its key, so later claimants are never stranded.
/// MUTATION: make `SingleFlightGuard::drop` a no-op and the second claim never returns.
#[test]
fn a_panicking_single_flight_holder_releases_its_key() {
    let flight = Arc::new(SingleFlight::<u32>::default());
    let holder = {
        let flight = Arc::clone(&flight);
        std::thread::spawn(move || {
            let _guard = flight.claim(7, || Ok(())).expect("first claim");
            panic!("injected panic while holding the key");
        })
    };
    assert!(holder.join().is_err());
    let (sender, receiver) = std::sync::mpsc::channel();
    let waiter = Arc::clone(&flight);
    std::thread::spawn(move || {
        let _ = sender.send(waiter.claim(7, || Ok(())).map(|_| ()));
    });
    receiver
        .recv_timeout(Duration::from_secs(60))
        .expect("a later claimant must not wait forever on an abandoned key")
        .expect("claim after the holder unwound");
}

/// A waiter stops waiting as soon as its cancellation check fails.
#[test]
fn a_cancelled_single_flight_waiter_returns_its_error() {
    let flight = SingleFlight::<u32>::default();
    let _held = flight.claim(1, || Ok(())).expect("first claim");
    let result = flight.claim(1, || Err(crate::Error::Canceled));
    assert!(matches!(result, Err(crate::Error::Canceled)));
}

/// The memo is keyed on the seal's own unchanged-predicate, so a mutation cannot hit it.
///
/// Same length, different bytes: only the fingerprints can tell these apart, which is exactly
/// the property that makes reuse safe. A rewrite always moves `ctime`, so this is a contract
/// rather than a timestamp-resolution race.
#[test]
fn a_mutated_file_misses_the_memo_and_is_read_again() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("encoder.safetensors");
    std::fs::write(&file, b"encoder weights").expect("write fixture");
    let first = ArtifactSeal::pin(&file).expect("pin regular file");

    std::fs::write(&file, b"ENCODER WEIGHTS").expect("mutate fixture");
    let before = test_full_hash_work_count();
    let second = ArtifactSeal::pin(&file).expect("re-pin the mutated file");
    assert!(
        test_full_hash_work_count() > before,
        "changed bytes must send the pin back through the full read"
    );
    assert_ne!(
        first.content_sha256(),
        second.content_sha256(),
        "the re-read must report the digest of the bytes that are there now"
    );
}

/// Persistence needs a volume identity. Developer and CI hosts must supply one for their temp
/// volume; only a synthetic filesystem with no id (e.g. a zero-fsid tmpfs) may decline.
fn persists_on_this_volume(dir: &Path) -> bool {
    let supplied = volume_identity(dir).is_some();
    if cfg!(any(target_os = "macos", windows)) {
        assert!(supplied, "no volume identity for {}", dir.display());
    } else if !supplied {
        eprintln!("skipping: {} has no filesystem id", dir.display());
    }
    supplied
}

/// The same file identity on a different volume is a different persisted entry.
#[test]
fn a_persisted_entry_is_bound_to_its_volume() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("transformer.safetensors");
    std::fs::write(&file, b"transformer weights").expect("write fixture");
    let pinned = ArtifactSeal::pin(&file).expect("pin regular file");
    let identity = SealIdentity {
        loader_path: pinned.loader_path.clone(),
        canonical_target_path: pinned.canonical_target_path.clone(),
        path_component_fingerprints: pinned.path_component_fingerprints.clone(),
        entry_fingerprint: pinned.entry_fingerprint.clone(),
        target_fingerprint: pinned.target_fingerprint.clone(),
    };
    let store = dir.path().join("seal-digests");
    assert_ne!(
        proven_digest_store_entry(&store, &identity, &[1; 16]),
        proven_digest_store_entry(&store, &identity, &[2; 16]),
        "a drive swapped in at the same path must not reach the first drive's entry"
    );
}

/// A restarted worker reuses the digest the previous process proved (sc-26610).
///
/// Forgetting this process's in-memory entries stands in for the restart; only the store can
/// then supply the digest.
#[test]
fn a_persisted_digest_survives_a_restart_without_a_re_read() {
    let dir = tempfile::tempdir().expect("temp dir");
    if !persists_on_this_volume(dir.path()) {
        return;
    }
    let store = dir.path().join("seal-digests");
    let file = dir.path().join("transformer.safetensors");
    std::fs::write(&file, b"transformer weights").expect("write fixture");

    let first = ArtifactSeal::pin_with_store(&file, Some(&store)).expect("pin regular file");
    forget_proven_digests_for(first.loader_path());
    let before = test_full_hash_work_count();
    let second = ArtifactSeal::pin_with_store(&file, Some(&store)).expect("re-pin after restart");
    assert_eq!(
        test_full_hash_work_count(),
        before,
        "an unchanged file proven by an earlier process must not be re-read"
    );
    assert_eq!(first.content_sha256(), second.content_sha256());
}

/// The persisted key is the whole seal identity, so a rewritten file misses the store.
#[test]
fn a_persisted_digest_never_authorizes_a_mutated_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    if !persists_on_this_volume(dir.path()) {
        return;
    }
    let store = dir.path().join("seal-digests");
    let file = dir.path().join("transformer.safetensors");
    std::fs::write(&file, b"transformer weights").expect("write fixture");
    let first = ArtifactSeal::pin_with_store(&file, Some(&store)).expect("pin regular file");

    // A different length, so the identity changes even within one coarse filesystem clock tick.
    std::fs::write(&file, b"transformer weights, rewritten").expect("mutate fixture");
    forget_proven_digests_for(first.loader_path());
    let before = test_full_hash_work_count();
    let second =
        ArtifactSeal::pin_with_store(&file, Some(&store)).expect("re-pin the mutated file");
    assert!(
        test_full_hash_work_count() > before,
        "changed bytes must send the pin back through the full read"
    );
    assert_ne!(first.content_sha256(), second.content_sha256());
}

/// A garbage store entry is a miss, never a digest.
#[test]
fn a_corrupt_persisted_entry_is_ignored() {
    let dir = tempfile::tempdir().expect("temp dir");
    if !persists_on_this_volume(dir.path()) {
        return;
    }
    let store = dir.path().join("seal-digests");
    let file = dir.path().join("transformer.safetensors");
    std::fs::write(&file, b"transformer weights").expect("write fixture");
    let first = ArtifactSeal::pin_with_store(&file, Some(&store)).expect("pin regular file");
    for entry in std::fs::read_dir(&store).expect("store exists") {
        std::fs::write(entry.expect("entry").path(), "not a digest").expect("corrupt");
    }

    forget_proven_digests_for(first.loader_path());
    let before = test_full_hash_work_count();
    let second =
        ArtifactSeal::pin_with_store(&file, Some(&store)).expect("re-pin with corrupt store");
    assert!(test_full_hash_work_count() > before);
    assert_eq!(first.content_sha256(), second.content_sha256());
}

fn sha256_of(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// sc-26691: a digest proven by the hashing writer seals the file without a content read, and
/// seeds the memo so later plain pins of the same identity read nothing either.
#[test]
fn an_attested_digest_seals_without_reading_while_the_identity_matches() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("derivative.safetensors");
    let bytes = vec![7_u8; 3 * 1024 * 1024];
    std::fs::write(&file, &bytes).expect("write fixture");
    let attestation = ArtifactDigestAttestation::capture(&file, sha256_of(&bytes)).expect("attest");

    let before = test_full_hash_work_count();
    let mut acquisitions = Vec::new();
    let sealed = ArtifactSeal::pin_attested_with_progress(
        &file,
        &attestation,
        || Ok(()),
        |progress| acquisitions.push((progress.acquisition, progress.hashed_bytes)),
    )
    .expect("attested pin");
    let again = ArtifactSeal::pin(&file).expect("plain pin after attestation");
    assert_eq!(
        test_full_hash_work_count(),
        before,
        "neither the attested pin nor the memo-seeded plain pin may read content"
    );
    assert_eq!(sealed.content_sha256(), &sha256_of(&bytes));
    assert_eq!(again.content_sha256(), sealed.content_sha256());
    assert_eq!(acquisitions, vec![(ArtifactSealAcquisition::Attested, 0)]);
    sealed
        .ensure_unchanged()
        .expect("the attested seal is a live seal");
}

/// The resolved cache publishes a produced bundle by renaming its staging DIRECTORY; the files
/// inside keep their identity, so the attestation captured in staging still redeems.
#[test]
fn an_attestation_survives_a_rename_of_the_parent_directory() {
    let dir = tempfile::tempdir().expect("temp dir");
    let staging = dir.path().join("staging");
    std::fs::create_dir(&staging).expect("staging");
    std::fs::write(staging.join("weights.safetensors"), b"produced bytes").expect("write");
    let attestation = ArtifactDigestAttestation::capture(
        staging.join("weights.safetensors"),
        sha256_of(b"produced bytes"),
    )
    .expect("attest");
    let bundle = dir.path().join("bundle");
    std::fs::rename(&staging, &bundle).expect("publish rename");

    let before = test_full_hash_work_count();
    let sealed =
        ArtifactSeal::pin_attested(bundle.join("weights.safetensors"), &attestation).unwrap();
    assert_eq!(test_full_hash_work_count(), before);
    assert_eq!(sealed.content_sha256(), &sha256_of(b"produced bytes"));
}

/// Tamper semantics are unchanged: a rewrite after the attestation moves the target identity,
/// so the pin reads the file and seals what is really there, never the attested digest.
#[test]
fn an_attestation_never_authorizes_a_rewritten_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("derivative.safetensors");
    std::fs::write(&file, b"original bytes").expect("write fixture");
    let attestation =
        ArtifactDigestAttestation::capture(&file, sha256_of(b"original bytes")).unwrap();
    // Same length; the identity still moves (ChangeTime/ctime, and mtime set explicitly so a
    // coarse filesystem clock cannot make this a race).
    std::fs::write(&file, b"tampered bytes").expect("mutate fixture");
    std::fs::File::options()
        .write(true)
        .open(&file)
        .unwrap()
        .set_modified(SystemTime::now() + std::time::Duration::from_secs(60))
        .unwrap();

    let before = test_full_hash_work_count();
    let sealed = ArtifactSeal::pin_attested(&file, &attestation).expect("falls back to a pin");
    assert!(
        test_full_hash_work_count() > before,
        "a moved identity must be re-read"
    );
    assert_eq!(sealed.content_sha256(), &sha256_of(b"tampered bytes"));
    assert_ne!(sealed.content_sha256(), attestation.content_sha256());
}

/// Overwrite `file` IN PLACE (same file object, same length) with `bytes`, then restore its
/// original mtime — the tamper only the change time (Windows ChangeTime / Unix ctime) can see.
/// The sleep first steps past the filesystem clock tick (NTFS / Linux timestamps come from a
/// coarse kernel clock), so the rewrite's change time is a later tick than the captured one.
fn rewrite_in_place_restoring_mtime(file: &Path, bytes: &[u8]) {
    use std::io::Write;
    let before = std::fs::metadata(file).unwrap();
    assert_eq!(before.len(), bytes.len() as u64, "a same-size rewrite");
    let mtime = before.modified().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));
    let mut handle = std::fs::File::options().write(true).open(file).unwrap();
    handle.write_all(bytes).unwrap();
    handle.sync_all().unwrap();
    handle.set_modified(mtime).unwrap();
    drop(handle);
    let after = std::fs::metadata(file).unwrap();
    assert_eq!(after.modified().unwrap(), mtime, "mtime restored exactly");
    assert_eq!(after.len(), before.len());
    assert_eq!(std::fs::read(file).unwrap(), bytes);
}

/// A same-size in-place rewrite that restores the original mtime keeps size, mtime and file
/// id — only the change time moves. It must still not redeem the attestation: the pin re-reads
/// and seals the real bytes. (Mutation-checked: with the change time not captured, this test
/// fails because the attested digest is returned without a read.)
#[test]
fn an_attestation_never_authorizes_an_mtime_restoring_same_size_rewrite() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("derivative.safetensors");
    std::fs::write(&file, b"original bytes").expect("write fixture");
    let attestation =
        ArtifactDigestAttestation::capture(&file, sha256_of(b"original bytes")).unwrap();
    rewrite_in_place_restoring_mtime(&file, b"tampered bytes");

    let before = test_full_hash_work_count();
    let mut acquisitions = Vec::new();
    let sealed = ArtifactSeal::pin_attested_with_progress(
        &file,
        &attestation,
        || Ok(()),
        |progress| acquisitions.push(progress.acquisition),
    )
    .expect("falls back to a pin");
    assert!(
        !acquisitions.contains(&ArtifactSealAcquisition::Attested),
        "a change-time-only tamper must not redeem the attestation: {acquisitions:?}"
    );
    assert!(
        test_full_hash_work_count() > before,
        "the tampered file must be re-read"
    );
    assert_eq!(sealed.content_sha256(), &sha256_of(b"tampered bytes"));
}

/// The hard-link path's twin: the same change-time-only tamper between the pin and the link
/// refuses the relink, so the old digest never labels the rewritten bytes. (Mutation-checked
/// like the attestation test above.)
#[test]
fn a_seal_refuses_to_hard_link_after_an_mtime_restoring_same_size_rewrite() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("derivative.safetensors");
    std::fs::write(&file, b"derivative bytes").expect("write fixture");
    let sealed = ArtifactSeal::pin(&file).expect("pin");
    rewrite_in_place_restoring_mtime(&file, b"tampered!! bytes");

    let link = dir.path().join("staged.safetensors");
    let error = sealed
        .hard_link_with_check(&link, || Ok(()))
        .expect_err("a change-time-only tamper must never be relinked under the old digest");
    assert!(error.to_string().contains("changed"), "{error}");
    assert!(
        !link.exists(),
        "the refusal happens before the link is created"
    );
}

/// sc-26691: a hard link the seal creates itself carries the digest across the ChangeTime/ctime
/// move it causes — no cold re-read — and the carried seal is live at the new identity.
#[test]
fn a_seal_carries_its_digest_across_a_hard_link_it_creates() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("derivative.safetensors");
    std::fs::write(&file, b"derivative bytes").expect("write fixture");
    let sealed = ArtifactSeal::pin(&file).expect("pin");
    let link = dir.path().join("staged.safetensors");

    let before = test_full_hash_work_count();
    let relinked = sealed
        .hard_link_with_check(&link, || Ok(()))
        .expect("hard link from the seal");
    let after_link = ArtifactSeal::pin(&file).expect("plain re-pin");
    assert_eq!(
        test_full_hash_work_count(),
        before,
        "neither the relink nor a later pin of the relinked identity may read content"
    );
    relinked
        .ensure_unchanged()
        .expect("live at the new identity");
    assert_eq!(relinked.content_sha256(), sealed.content_sha256());
    assert_eq!(after_link.content_sha256(), sealed.content_sha256());
    assert_eq!(std::fs::read(&link).unwrap(), b"derivative bytes");
}

/// A file rewritten before the link is not carried: the seal refuses rather than relabeling
/// changed bytes with its old digest.
#[test]
fn a_seal_refuses_to_hard_link_a_file_that_changed_since_it_was_pinned() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("derivative.safetensors");
    std::fs::write(&file, b"derivative bytes").expect("write fixture");
    let sealed = ArtifactSeal::pin(&file).expect("pin");
    std::fs::write(&file, b"rewritten, longer bytes").expect("mutate fixture");

    let error = sealed
        .hard_link_with_check(dir.path().join("staged.safetensors"), || Ok(()))
        .expect_err("a changed file must never be relinked under the old digest");
    assert!(error.to_string().contains("changed"), "{error}");
}

#[test]
fn pinned_weights_file_detects_target_mutation() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("model.safetensors");
    std::fs::write(&file, b"original").expect("write fixture");
    let pinned = PinnedWeightsFile::pin(&file).expect("pin regular file");

    assert_eq!(pinned.loader_path(), file.as_path());
    pinned.ensure_unchanged().expect("unchanged file");

    std::fs::write(&file, b"replacement bytes").expect("mutate fixture");
    let error = pinned
        .ensure_unchanged()
        .expect_err("mutation must invalidate the pin")
        .to_string();
    assert!(error.contains("changed after load"), "got: {error}");
}

#[test]
fn artifact_seal_skips_stable_full_hashes_and_audits_same_size_replacement() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("model.safetensors");
    std::fs::write(&file, b"same-size-a").expect("write fixture");
    let seal = ArtifactSeal::pin(&file).expect("seal regular file");

    let after_acquisition = seal.full_hash_work_count();
    seal.verify_unchanged().expect("first stable verification");
    seal.verify_unchanged().expect("second stable verification");
    assert_eq!(
        seal.full_hash_work_count(),
        after_acquisition,
        "stable request checks must consume only identity pins, not full content hashes"
    );

    std::fs::write(&file, b"same-size-b").expect("replace fixture with same-size bytes");
    let forced_stamp = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_123);
    // A read-only handle cannot carry a timestamp write on Windows (`PermissionDenied`), so the
    // replacement stamp goes through a writable handle, as the sibling fixture below does.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&file)
        .expect("open fixture")
        .set_modified(forced_stamp)
        .expect("advance replacement stamp");
    let error = seal
        .verify_unchanged()
        .expect_err("same-size replacement must fail closed");
    assert_artifact_seal_mismatch(error);
    assert_eq!(
        seal.full_hash_work_count(),
        after_acquisition + 1,
        "identity mismatch must trigger exactly one full-content audit before rejection"
    );
}

#[test]
fn pinned_weights_file_rejects_a_barrier_controlled_mid_read_replacement() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("model.safetensors");
    let original = vec![0x5a; 128 * 1024];
    std::fs::write(&file, &original).expect("write fixture");
    let pinned = PinnedWeightsFile::pin(&file).expect("pin regular file");
    let consumed_first_chunk = Arc::new(Barrier::new(2));
    let replacement_done = Arc::new(Barrier::new(2));

    let writer_file = file.clone();
    let writer_entered = Arc::clone(&consumed_first_chunk);
    let writer_done = Arc::clone(&replacement_done);
    let writer = std::thread::spawn(move || {
        writer_entered.wait();
        let replacement = writer_file.with_extension("replacement");
        // A replacing rename on every platform, so the reader below keeps consuming the original
        // object everywhere: an in-place overwrite would feed it the replacement's bytes off
        // Unix, and could not run at all against a source the caller had mapped.
        let swapped = std::fs::write(&replacement, vec![0xa5; 128 * 1024])
            .and_then(|()| std::fs::rename(replacement, writer_file));
        // Release the reader whatever the swap did: returning ahead of this barrier strands it
        // there and hangs the test binary. The outcome is asserted on `join` instead.
        writer_done.wait();
        swapped
    });

    let outcome = pinned.read_unchanged::<_, crate::Error>(|path| {
        // Open and consume part of the original payload before allowing replacement. The
        // remainder is then read from the already-open original file object — which the
        // replacing rename unlinks but does not disturb — proving the post-read check, not a
        // pre-open race, is what rejects the mixed-provenance operation.
        let mut source = std::fs::File::open(path)?;
        let mut bytes = vec![0; 4096];
        source.read_exact(&mut bytes)?;
        assert!(bytes.iter().all(|byte| *byte == 0x5a));
        consumed_first_chunk.wait();
        replacement_done.wait();
        source.read_to_end(&mut bytes)?;
        assert_eq!(bytes.len(), original.len());
        assert!(bytes.iter().all(|byte| *byte == 0x5a));
        Ok(bytes)
    });
    writer
        .join()
        .expect("writer thread")
        .expect("replace the pinned source mid-read");

    let error = outcome.expect_err("a replacement between the two checks must fail");
    assert_artifact_seal_mismatch(error);
}

#[cfg(unix)]
#[test]
fn pinned_weights_file_preserves_and_pins_the_symlink_entry() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().expect("temp dir");
    let first = dir.path().join("content-addressed-blob-a");
    let second = dir.path().join("content-addressed-blob-b");
    let link = dir.path().join("selected-model.safetensors");
    std::fs::write(&first, b"same-size-a").expect("write first target");
    std::fs::write(&second, b"same-size-b").expect("write second target");
    symlink(&first, &link).expect("create extension-bearing symlink");

    let pinned = PinnedWeightsFile::pin(&link).expect("pin symlinked file");
    assert_eq!(
        pinned.loader_path(),
        link.as_path(),
        "the loader path must stay lexical rather than canonicalizing to the blob"
    );
    pinned.ensure_unchanged().expect("unchanged symlink");

    std::fs::remove_file(&link).expect("remove old link");
    symlink(&second, &link).expect("retarget symlink");
    let error = pinned
        .ensure_unchanged()
        .expect_err("retargeting the entry must invalidate the pin")
        .to_string();
    assert!(error.contains("entry changed"), "got: {error}");
}

#[cfg(unix)]
#[test]
fn pinned_weights_file_detects_a_recreated_intermediate_symlink() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().expect("temp dir");
    let first = dir.path().join("snapshot-a");
    let second = dir.path().join("snapshot-b");
    std::fs::create_dir(&first).expect("create A snapshot");
    std::fs::create_dir(&second).expect("create B snapshot");
    std::fs::write(first.join("model.safetensors"), b"same-size-a").expect("write A");
    std::fs::write(second.join("model.safetensors"), b"same-size-b").expect("write B");
    let selected = dir.path().join("selected");
    let staged_b = dir.path().join("staged-b");
    let recreated_a = dir.path().join("recreated-a");
    symlink(&first, &selected).expect("select A directory");
    symlink(&second, &staged_b).expect("stage B directory link");
    symlink(&first, &recreated_a).expect("stage recreated A directory link");
    let lexical_file = selected.join("model.safetensors");
    let pinned = PinnedWeightsFile::pin(&lexical_file).expect("pin through intermediate link");
    assert_eq!(
        pinned.canonical_target_path(),
        std::fs::canonicalize(first.join("model.safetensors")).unwrap()
    );

    std::fs::rename(staged_b, &selected).expect("select B directory");
    std::fs::rename(recreated_a, &selected).expect("select recreated A directory link");
    assert_eq!(std::fs::read(&lexical_file).unwrap(), b"same-size-a");
    let error = pinned
        .ensure_unchanged()
        .expect_err("recreating an intermediate symlink must change component identity")
        .to_string();
    assert!(error.contains("path component changed"), "got: {error}");
}

#[cfg(windows)]
#[test]
fn pinned_weights_file_detects_a_same_size_windows_file_replacement() {
    use std::os::windows::fs::FileTimesExt;

    let dir = tempfile::tempdir().expect("temp dir");
    let selected = dir.path().join("model.safetensors");
    let replacement = dir.path().join("replacement.safetensors");
    std::fs::write(&selected, b"same-size-a").expect("write A");
    std::fs::write(&replacement, b"same-size-b").expect("write B");
    let timestamp = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
    let times = std::fs::FileTimes::new()
        .set_modified(timestamp)
        .set_created(timestamp);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&selected)
        .unwrap()
        .set_times(times)
        .expect("set A times");
    std::fs::OpenOptions::new()
        .write(true)
        .open(&replacement)
        .unwrap()
        .set_times(times)
        .expect("set B times");
    let selected_metadata = std::fs::metadata(&selected).unwrap();
    let replacement_metadata = std::fs::metadata(&replacement).unwrap();
    assert_eq!(
        (
            selected_metadata.len(),
            selected_metadata.modified().ok(),
            selected_metadata.created().ok(),
        ),
        (
            replacement_metadata.len(),
            replacement_metadata.modified().ok(),
            replacement_metadata.created().ok(),
        ),
        "the legacy size/mtime/created tuple must collide"
    );
    let pinned = PinnedWeightsFile::pin(&selected).expect("pin A");

    std::fs::remove_file(&selected).expect("remove A");
    std::fs::rename(&replacement, &selected).expect("install same-size B");
    let error = pinned
        .ensure_unchanged()
        .expect_err("Windows file ID/change time must reject same-size replacement")
        .to_string();
    assert!(
        error.contains("entry changed") || error.contains("target changed"),
        "got: {error}"
    );
}

#[cfg(windows)]
#[test]
fn pinned_weights_file_ignores_unrelated_windows_sibling_changes() {
    let dir = tempfile::tempdir().expect("temp dir");
    let selected = dir.path().join("model.safetensors");
    let sibling = dir.path().join("unrelated.tmp");
    std::fs::write(&selected, b"weights").expect("write model");
    let pinned = PinnedWeightsFile::pin(&selected).expect("pin model");

    std::fs::write(&sibling, b"unrelated").expect("create sibling");
    std::fs::remove_file(&sibling).expect("remove sibling");
    pinned
        .ensure_unchanged()
        .expect("parent identity must ignore ordinary child-entry timestamp changes");
}

#[cfg(windows)]
#[test]
fn pinned_weights_file_distinguishes_windows_symlink_entry_and_target() {
    use std::os::windows::fs::symlink_file;

    let dir = tempfile::tempdir().expect("temp dir");
    let first = dir.path().join("blob-a");
    let second = dir.path().join("blob-b");
    let selected = dir.path().join("selected.safetensors");
    std::fs::write(&first, b"same-size-a").expect("write A");
    std::fs::write(&second, b"same-size-b").expect("write B");
    if let Err(error) = symlink_file(&first, &selected) {
        if error.kind() == std::io::ErrorKind::PermissionDenied {
            eprintln!(
                    "skipping Windows symlink identity fixture because this host lacks symlink privilege: {error}"
                );
            return;
        }
        panic!("create Windows file symlink: {error}");
    }

    let pinned = PinnedWeightsFile::pin(&selected).expect("pin selected A link");
    assert_eq!(
        pinned.loader_path(),
        std::path::absolute(&selected).unwrap()
    );
    assert_eq!(
        pinned.canonical_target_path(),
        std::fs::canonicalize(&first).unwrap()
    );
    assert_ne!(
        pinned.entry_fingerprint().file_id,
        pinned.target_fingerprint().file_id,
        "OPEN_REPARSE_POINT must fingerprint the link entry, not its target"
    );

    std::fs::remove_file(&selected).expect("remove A link");
    symlink_file(&second, &selected).expect("retarget selected link to B");
    let error = pinned
        .ensure_unchanged()
        .expect_err("retargeting the Windows link must invalidate its entry token")
        .to_string();
    assert!(
        error.contains("entry changed") || error.contains("resolution changed"),
        "got: {error}"
    );
}

#[test]
fn load_spec_rejects_a_prepared_token_for_another_lexical_path() {
    let dir = tempfile::tempdir().expect("temp dir");
    let expected = dir.path().join("expected.safetensors");
    let other = dir.path().join("other.safetensors");
    std::fs::write(&expected, b"expected").expect("write expected file");
    std::fs::write(&other, b"other").expect("write other file");

    let token = PinnedWeightsFile::pin(&other).expect("pin other file");
    let error = LoadSpec::new(WeightsSource::File(expected.clone()))
        .with_prepared_file_pin(&expected, token)
        .expect_err("a token for another file must fail closed")
        .to_string();

    assert!(error.contains("path mismatch"), "got: {error}");
}

#[test]
fn load_spec_reuses_the_exact_prepared_primary_file_token() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("model.safetensors");
    std::fs::write(&file, b"weights").expect("write weights");
    let prepared = PinnedWeightsFile::pin(&file).expect("prepare file");
    let mut spec = LoadSpec::new(WeightsSource::File(file.clone()))
        .with_prepared_file_pin(&file, prepared.clone())
        .expect("attach matching token");
    spec.finish_file_source_preparation()
        .expect("finalize matching token set");

    let provider_pin = spec
        .weights_file_pin()
        .expect("resolve provider pin")
        .expect("file pin");
    assert_eq!(provider_pin, prepared);
    assert_eq!(provider_pin.loader_path(), file.as_path());
    assert_eq!(
        spec.prepared_file_pin_for(&file)
            .expect("validate prepared token")
            .expect("prepared token"),
        &prepared
    );
}

#[test]
fn unprepared_file_callers_still_receive_a_primary_file_pin() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("model.safetensors");
    std::fs::write(&file, b"weights").expect("write weights");
    let spec = LoadSpec::new(WeightsSource::File(file.clone()));

    assert!(spec.prepared_file_pins.is_empty());
    let provider_pin = spec
        .weights_file_pin()
        .expect("pin ordinary file caller")
        .expect("file pin");
    assert_eq!(provider_pin.loader_path(), file.as_path());
}

#[test]
fn unprepared_callback_guard_preserves_raw_callback_behavior() {
    let spec = LoadSpec::new(WeightsSource::File(
        "/a/compatibility/caller/need/not/open/this.safetensors".into(),
    ));

    let value = spec
        .read_prepared_files_unchanged(|| Ok::<_, crate::Error>(17))
        .expect("an unprepared callback must run without eager filesystem access");
    assert_eq!(value, 17);
    assert!(!spec.prepared_file_pins.is_prepared());
}

#[test]
fn prepared_only_file_guard_preserves_compatibility_and_rejects_a_stale_token() {
    let missing = PathBuf::from("/a/compatibility/caller/need/not/open/this-file.safetensors");
    let ordinary = LoadSpec::new(WeightsSource::File(missing.clone()));
    let value = ordinary
        .read_file_unchanged_if_prepared(&missing, |path| {
            assert_eq!(path, missing);
            Ok::<_, crate::Error>(23)
        })
        .expect("an unprepared file read must not eagerly touch the filesystem");
    assert_eq!(value, 23);

    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("prepared.safetensors");
    std::fs::write(&file, b"original").expect("write original file");
    let mut prepared = LoadSpec::new(WeightsSource::File(file.clone()));
    prepared.prepare_file_sources().expect("prepare file token");
    std::fs::write(&file, b"replacement-with-another-size").expect("replace prepared file");

    let error = prepared
        .read_file_unchanged_if_prepared(&file, |_| Ok::<_, crate::Error>(()))
        .expect_err("a prepared read must validate the caller-installed token");
    assert_artifact_seal_mismatch(error);
}

#[test]
fn prepared_dir_only_mode_stays_sticky_when_a_file_slot_is_added() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("late-control.safetensors");
    std::fs::write(&file, b"control").expect("write control");
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().join("snapshot")));

    spec.prepare_file_sources().expect("prepare Dir-only spec");
    assert!(spec.prepared_file_pins.is_prepared());
    assert!(spec.prepared_file_pins.is_empty());

    spec.control = Some(WeightsSource::File(file.clone()));
    let error = spec
        .file_pin_for(&file)
        .expect_err("a later File slot must not downgrade to compatibility repinning")
        .to_string();
    assert!(error.contains("missing configured source"), "got: {error}");
    assert!(
        spec.prepare_file_sources().is_err(),
        "re-preparing a sticky spec must validate rather than fill a newly added slot"
    );
    assert!(spec.prepared_file_pins.is_prepared());
    assert!(spec.prepared_file_pins.is_empty());
}

#[test]
fn atomic_prepared_token_install_uses_exact_tokens_and_supports_zero_files() {
    let dir = tempfile::tempdir().expect("temp dir");
    let primary = dir.path().join("primary.safetensors");
    let control = dir.path().join("control.safetensors");
    std::fs::write(&primary, b"primary").expect("write primary");
    std::fs::write(&control, b"control").expect("write control");
    let primary_pin = PinnedWeightsFile::pin(&primary).expect("pin primary");
    let control_pin = PinnedWeightsFile::pin(&control).expect("pin control");
    let mut spec = LoadSpec::new(WeightsSource::File(primary.clone()))
        .with_control(WeightsSource::File(control.clone()));

    spec.prepare_with_file_pins([primary_pin.clone(), control_pin.clone()])
        .expect("atomically install exact set");
    assert!(spec.prepared_file_pins().is_prepared());
    assert!(spec.prepared_file_pins().is_finalized());
    assert_eq!(spec.file_pin_for(&primary).unwrap(), primary_pin);
    assert_eq!(spec.file_pin_for(&control).unwrap(), control_pin);

    let mut dir_only = LoadSpec::new(WeightsSource::Dir(dir.path().join("snapshot")));
    dir_only
        .prepare_with_file_pins(std::iter::empty())
        .expect("zero-token set explicitly prepares a Dir-only spec");
    assert!(dir_only.prepared_file_pins().is_prepared());
    assert!(dir_only.prepared_file_pins().is_finalized());
    assert!(dir_only.prepared_file_pins().is_empty());
}

#[test]
fn prepared_directory_members_are_guarded_and_orphans_fail_closed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let transformer = dir.path().join("transformer");
    std::fs::create_dir(&transformer).expect("create transformer dir");
    let config = transformer.join("config.json");
    let weights = transformer.join("diffusion_pytorch_model.safetensors");
    std::fs::write(&config, b"{\"kind\":\"mage\"}").expect("write config");
    std::fs::write(&weights, b"weights-v1").expect("write weights");

    let mut spec = LoadSpec::new(WeightsSource::Dir(transformer));
    spec.prepare_with_file_pins([
        PinnedWeightsFile::pin(&config).expect("pin config"),
        PinnedWeightsFile::pin(&weights).expect("pin weights"),
    ])
    .expect("directory members are valid prepared identities");
    assert_eq!(spec.prepared_file_pins().len(), 2);
    spec.read_prepared_files_unchanged(|| Ok::<_, crate::Error>(()))
        .expect("unchanged directory members pass the callback guard");

    std::fs::write(&weights, b"weights-v2").expect("replace weights in place");
    spec.read_prepared_files_unchanged(|| Ok::<_, crate::Error>(()))
        .expect_err("a changed directory member must fail before provider reuse");

    let outside = dir.path().join("outside.safetensors");
    std::fs::write(&outside, b"outside").expect("write outside file");
    let mut orphaned = LoadSpec::new(WeightsSource::Dir(dir.path().join("other")));
    orphaned
        .prepare_with_file_pins([PinnedWeightsFile::pin(outside).expect("pin outside")])
        .expect_err("a token outside every configured directory must fail closed");
}

#[test]
fn prepared_mode_rejects_an_orphan_after_a_file_slot_is_removed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = dir.path().join("primary.safetensors");
    std::fs::write(&file, b"primary").expect("write primary");
    let mut spec = LoadSpec::new(WeightsSource::File(file));
    spec.prepare_file_sources().expect("prepare File spec");

    spec.weights = WeightsSource::Dir(dir.path().join("snapshot"));
    let error = spec
        .validate_prepared_file_pins()
        .expect_err("removing a prepared File slot must leave an invalid orphan")
        .to_string();
    assert!(error.contains("does not match"), "got: {error}");
    assert!(spec.prepared_file_pins.is_prepared());
    assert_eq!(spec.prepared_file_pins.len(), 1);
}

#[test]
fn file_source_paths_covers_the_complete_load_spec_slot_matrix() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = |name: &str| dir.path().join(format!("{name}.safetensors"));
    let primary = path("primary");
    let control = path("control");
    let extra_control_0 = path("extra-control-0");
    let extra_control_1 = path("extra-control-1");
    let ip_adapter = path("ip-adapter");
    let adapter = path("adapter");
    let pid_checkpoint = path("pid-checkpoint");
    let pid_gemma = path("pid-gemma");
    let identity_encoder = path("identity-encoder");
    let identity_eva = path("identity-eva");
    let identity_face = path("identity-face");
    let text_encoder = path("text-encoder");
    let component_alpha = path("component-alpha");
    let component_zeta = path("component-zeta");
    let expected = vec![
        primary.clone(),
        control.clone(),
        extra_control_0.clone(),
        extra_control_1.clone(),
        ip_adapter.clone(),
        adapter.clone(),
        pid_checkpoint.clone(),
        pid_gemma.clone(),
        identity_encoder.clone(),
        identity_eva.clone(),
        identity_face.clone(),
        text_encoder.clone(),
        component_alpha.clone(),
        component_zeta.clone(),
    ];
    for file in &expected {
        std::fs::write(file, b"fixture").expect("write slot fixture");
    }

    let mut spec = LoadSpec::new(WeightsSource::File(primary));
    spec.control = Some(WeightsSource::File(control));
    spec.extra_controls = vec![
        WeightsSource::File(extra_control_0),
        WeightsSource::File(extra_control_1),
    ];
    spec.ip_adapter = Some(WeightsSource::File(ip_adapter));
    spec.adapters = vec![AdapterSpec::new(adapter, 1.0, AdapterKind::Lora)];
    spec.pid = Some(PidWeights {
        checkpoint: WeightsSource::File(pid_checkpoint),
        gemma: WeightsSource::File(pid_gemma),
    });
    spec.identity = Some(IdentityWeights {
        encoder: Some(WeightsSource::File(identity_encoder)),
        eva: Some(WeightsSource::File(identity_eva)),
        face_dir: Some(WeightsSource::File(identity_face)),
    });
    spec.text_encoder = Some(WeightsSource::File(text_encoder));
    // BTreeMap value iteration is key-sorted, independently of insertion order.
    spec.components
        .insert("zeta".into(), WeightsSource::File(component_zeta));
    spec.components
        .insert("alpha".into(), WeightsSource::File(component_alpha));

    let actual: Vec<PathBuf> = spec
        .file_source_paths()
        .into_iter()
        .map(Path::to_path_buf)
        .collect();
    assert_eq!(actual, expected);

    spec.prepare_file_sources()
        .expect("prepare every File slot");
    spec.validate_prepared_file_pins()
        .expect("prepared map exactly covers the slot matrix");
    assert_eq!(spec.prepared_file_pins.len(), expected.len());
    for file in expected {
        let token = spec
            .file_pin_for(&file)
            .expect("resolve prepared slot token");
        assert_eq!(token.loader_path(), file.as_path());
    }
}

#[cfg(unix)]
#[test]
fn prepared_load_spec_rejects_a_to_b_to_recreated_a_rebinding() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().expect("temp dir");
    let first = dir.path().join("blob-a");
    let second = dir.path().join("blob-b");
    let selected = dir.path().join("selected.safetensors");
    let staged_b = dir.path().join("staged-b.safetensors");
    let staged_a = dir.path().join("staged-a.safetensors");
    std::fs::write(&first, b"same-size-a").expect("write A");
    std::fs::write(&second, b"same-size-b").expect("write B");
    symlink(&first, &selected).expect("select A");
    // Create both replacement entries before the race so their inodes are provably distinct
    // from the original selected entry, even after the final path once again resolves to A.
    symlink(&second, &staged_b).expect("stage B link");
    symlink(&first, &staged_a).expect("stage replacement A link");

    let mut spec = LoadSpec::new(WeightsSource::File(selected.clone()));
    spec.prepare_file_sources().expect("prepare A identity");
    let key_pin = spec
        .prepared_file_pin_for(&selected)
        .expect("validate cache-key token")
        .expect("prepared token")
        .clone();
    let start_mutation = Arc::new(Barrier::new(2));
    let mutation_done = Arc::new(Barrier::new(2));
    let writer_start = Arc::clone(&start_mutation);
    let writer_done = Arc::clone(&mutation_done);
    let writer_selected = selected.clone();
    let writer = std::thread::spawn(move || {
        writer_start.wait();
        let rebound = std::fs::rename(staged_b, &writer_selected)
            .and_then(|()| std::fs::rename(staged_a, &writer_selected));
        // Release the main thread whatever the rebinding did. It is parked on `mutation_done`
        // below, so a writer that panics — or returns — ahead of this barrier strands it there
        // and hangs the whole test binary; the outcome is asserted on `join` instead, so a
        // failed rebinding reads as a red test naming the error.
        writer_done.wait();
        rebound
    });

    start_mutation.wait();
    mutation_done.wait();
    writer
        .join()
        .expect("writer thread")
        .expect("rebind the selected path through B back to a recreated A");
    assert_eq!(
        std::fs::read(&selected).expect("read final selected target"),
        b"same-size-a",
        "the lexical path must resolve to A again before the provider consumes the token"
    );
    let error = spec
        .weights_file_pin()
        .expect_err("the provider must reject the stale cache-key token");
    assert_artifact_seal_mismatch(error);
    assert_eq!(
        key_pin.loader_path(),
        selected.as_path(),
        "cache identity must retain the extension-bearing lexical path"
    );
}

#[test]
fn prepared_mode_rejects_a_missing_token_instead_of_repinning() {
    let dir = tempfile::tempdir().expect("temp dir");
    let primary = dir.path().join("primary.safetensors");
    let adapter = dir.path().join("adapter.safetensors");
    std::fs::write(&primary, b"primary").expect("write primary");
    std::fs::write(&adapter, b"adapter").expect("write adapter");
    let mut spec =
        LoadSpec::new(WeightsSource::File(primary.clone())).with_adapters(vec![AdapterSpec::new(
            adapter.clone(),
            1.0,
            AdapterKind::Lora,
        )]);
    spec.set_prepared_file_pin(
        &primary,
        PinnedWeightsFile::pin(&primary).expect("pin primary"),
    )
    .expect("attach primary token");

    let error = spec
        .finish_file_source_preparation()
        .expect_err("a partial caller-installed set must not finalize")
        .to_string();
    assert!(error.contains("does not match"), "got: {error}");
    let use_error = spec
        .file_pin_for(&adapter)
        .expect_err("an unfinished prepared set must not silently re-pin")
        .to_string();
    assert!(use_error.contains("not been finalized"), "got: {use_error}");
    assert!(spec.validate_prepared_file_pins().is_err());
}

#[cfg(unix)]
#[test]
fn prepared_adapter_component_and_control_reject_a_to_b_to_recreated_a_rebinding() {
    use std::os::unix::fs::symlink;

    for role in ["adapter", "component", "control"] {
        let dir = tempfile::tempdir().expect("temp dir");
        let first = dir.path().join("blob-a");
        let second = dir.path().join("blob-b");
        let selected = dir.path().join(format!("{role}.safetensors"));
        let staged_b = dir.path().join(format!("{role}-staged-b.safetensors"));
        let staged_a = dir.path().join(format!("{role}-staged-a.safetensors"));
        std::fs::write(&first, b"same-size-a").expect("write A");
        std::fs::write(&second, b"same-size-b").expect("write B");
        symlink(&first, &selected).expect("select A");
        symlink(&second, &staged_b).expect("stage B link");
        symlink(&first, &staged_a).expect("stage replacement A link");

        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().join("base")));
        match role {
            "adapter" => {
                spec.adapters
                    .push(AdapterSpec::new(selected.clone(), 1.0, AdapterKind::Lora));
            }
            "component" => {
                spec.components
                    .insert("vae".into(), WeightsSource::File(selected.clone()));
            }
            "control" => spec.control = Some(WeightsSource::File(selected.clone())),
            _ => unreachable!(),
        }
        spec.prepare_file_sources()
            .expect("prepare composite load identity");

        let start_mutation = Arc::new(Barrier::new(2));
        let mutation_done = Arc::new(Barrier::new(2));
        let writer_start = Arc::clone(&start_mutation);
        let writer_done = Arc::clone(&mutation_done);
        let writer_selected = selected.clone();
        let writer = std::thread::spawn(move || {
            writer_start.wait();
            let rebound = std::fs::rename(staged_b, &writer_selected)
                .and_then(|()| std::fs::rename(staged_a, &writer_selected));
            // Release the main thread whatever the rebinding did; see the sibling test above.
            // Returning ahead of this barrier strands the main thread on it and hangs the test
            // binary. The outcome is asserted on `join` instead.
            writer_done.wait();
            rebound
        });

        start_mutation.wait();
        mutation_done.wait();
        writer
            .join()
            .expect("writer thread")
            .expect("rebind the selected path through B back to a recreated A");
        assert_eq!(
            std::fs::read(&selected).expect("read final selected target"),
            b"same-size-a"
        );
        let error = spec
            .file_pin_for(&selected)
            .expect_err("provider must consume the stale prepared token");
        assert_artifact_seal_mismatch(error);
    }
}

fn frame(current: u32, total: u32) -> PreviewFrame {
    PreviewFrame {
        current,
        total,
        image: crate::media::Image {
            width: 2,
            height: 1,
            pixels: vec![0, 0, 0, 255, 255, 255],
        },
    }
}

/// A caller-owned token IS the flag the engine polls: cancelling through the consumer's own
/// `Arc` is observed by the engine's handle, with no bridge and no progress-callback hop.
#[test]
fn from_arc_shares_the_callers_token() {
    let token = Arc::new(AtomicBool::new(false));
    let flag = CancelFlag::from_arc(Arc::clone(&token));

    assert!(!flag.is_cancelled());
    token.store(true, Ordering::Relaxed);
    assert!(
        flag.is_cancelled(),
        "the engine handle must see the caller's cancel"
    );
}

/// And the reverse direction: `cancel()` on the engine handle is visible on the caller's token,
/// so a consumer can share one token across several in-flight requests.
#[test]
fn cancel_is_visible_on_the_shared_token() {
    let token = Arc::new(AtomicBool::new(false));
    let flag = CancelFlag::from_arc(Arc::clone(&token));

    flag.cancel();
    assert!(token.load(Ordering::Relaxed));
    assert!(
        Arc::ptr_eq(flag.as_arc(), &token),
        "as_arc must expose the same allocation"
    );
}

/// The inert default is the zero-cost path: engines gate their projection on `is_active`, and
/// `emit` on an inert sink must not panic (a provider may emit unconditionally).
#[test]
fn default_preview_sink_is_inert() {
    let sink = PreviewSink::default();
    assert!(!sink.is_active());
    sink.emit(frame(1, 8)); // must be a no-op, not a panic
}

/// An active sink receives every emitted frame, in order, through a clone of the handle — the
/// `CancelFlag` pattern: cloning the request must not detach the sink.
#[test]
fn active_preview_sink_receives_frames_through_a_clone() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&seen);
    let sink = PreviewSink::new(move |f: PreviewFrame| {
        recorder
            .lock()
            .unwrap()
            .push((f.current, f.total, f.image.pixels.len()));
    });
    assert!(sink.is_active());

    let cloned = sink.clone();
    sink.emit(frame(1, 8));
    cloned.emit(frame(2, 8));

    assert_eq!(*seen.lock().unwrap(), vec![(1, 8, 6), (2, 8, 6)]);
}

/// `Debug` reports liveness, never the closure — matching `CancelFlag`'s shape so a request
/// stays printable.
#[test]
fn preview_sink_debug_reports_liveness() {
    assert_eq!(
        format!("{:?}", PreviewSink::default()),
        "PreviewSink(false)"
    );
    assert_eq!(
        format!("{:?}", PreviewSink::new(|_| {})),
        "PreviewSink(true)"
    );
}

#[test]
fn prompt_enhancement_sink_reports_typed_outcomes_through_a_clone() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&seen);
    let sink = PromptEnhancementSink::new(move |report| {
        recorder.lock().unwrap().push(report);
    });
    sink.clone()
        .emit(PromptEnhancementReport::enhanced("a cat", "a detailed cat"));
    sink.emit(PromptEnhancementReport::fallback("a dog", "empty_output"));
    sink.emit(PromptEnhancementReport::absent("unchanged bytes\n"));

    assert_eq!(
        *seen.lock().unwrap(),
        vec![
            PromptEnhancementReport::enhanced("a cat", "a detailed cat"),
            PromptEnhancementReport::fallback("a dog", "empty_output"),
            PromptEnhancementReport::absent("unchanged bytes\n"),
        ]
    );
    assert_eq!(
        format!("{:?}", PromptEnhancementSink::default()),
        "PromptEnhancementSink(false)"
    );
}
