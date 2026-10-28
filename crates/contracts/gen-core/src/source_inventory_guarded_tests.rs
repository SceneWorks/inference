//! Actual Darwin walker/seal tests, including requested-layout allocator policing.
use super::*;
use crate::WeightsSource;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::os::unix::fs::symlink;

struct CheckedAllocator;
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static CREDIT: Cell<usize> = const { Cell::new(0) };
    static UNBILLED: Cell<usize> = const { Cell::new(0) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
fn allocation(bytes: usize) {
    if ACTIVE.try_with(Cell::get).unwrap_or(false) {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        CREDIT.with(|credit| {
            if bytes > credit.get() {
                UNBILLED.with(|n| n.set(n.get() + 1));
            }
            credit.set(credit.get().saturating_sub(bytes));
        });
    }
}
unsafe impl GlobalAlloc for CheckedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocation(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        allocation(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CheckedAllocator = CheckedAllocator;

/// No test assertion, logging, fixture IO or callback Vec runs inside tracking.
/// A callback resets credit: a previous overbill cannot conceal a later late bill.
pub(crate) fn checked<T>(
    refuse: Option<usize>,
    run: impl FnOnce(&mut dyn FnMut(u64) -> Result<()>) -> Result<T>,
) -> (Result<T>, usize, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ACTIVE.with(|v| v.set(false));
        }
    }
    let mut bills = 0;
    let mut check = |bytes: u64| {
        CREDIT.with(|v| v.set(0));
        if bytes > 0 {
            let index = bills;
            bills += 1;
            if refuse == Some(index) {
                return Err(Error::Canceled);
            }
        }
        CREDIT.with(|v| v.set(bytes as usize));
        Ok(())
    };
    UNBILLED.with(|v| v.set(0));
    ALLOCATIONS.with(|v| v.set(0));
    CREDIT.with(|v| v.set(0));
    ACTIVE.with(|v| v.set(true));
    let reset = Reset;
    let result = run(&mut check);
    drop(reset);
    let unbilled = UNBILLED.with(Cell::get);
    let allocations = ALLOCATIONS.with(Cell::get);
    assert_eq!(
        unbilled, 0,
        "every requested allocation needs a preceding sufficient bill"
    );
    (result, bills, allocations)
}
fn capture(spec: &LoadSpec) -> SourceInventory {
    let (result, _, allocations) = checked(None, |next| {
        SourceInventory::capture_with_capacity(spec, next)
    });
    assert!(allocations > 0, "the actual walker/sealer must execute");
    result.unwrap()
}

#[test]
fn guarded_fresh_mixed_recursive_closure_matches_native_and_never_rehashes_verification() {
    let root = tempfile::tempdir().unwrap();
    let tree = root.path().join("tree");
    std::fs::create_dir_all(tree.join("nested")).unwrap();
    std::fs::write(tree.join("nested/a"), b"alpha").unwrap();
    std::fs::write(tree.join("b"), b"beta").unwrap();
    std::fs::create_dir(tree.join(".hidden-dir")).unwrap();
    std::fs::write(tree.join(".hidden-dir/not-source"), b"hidden").unwrap();
    std::fs::write(tree.join("._sidecar"), b"hidden").unwrap();
    let aux = root.path().join("aux");
    std::fs::write(&aux, b"aux").unwrap();
    symlink("nested/a", tree.join("alias.safetensors")).unwrap();
    let mut spec = LoadSpec::new(WeightsSource::Dir(tree.clone()));
    spec.components
        .insert("aux".into(), WeightsSource::File(aux));
    spec.components
        .insert("duplicate".into(), WeightsSource::Dir(tree.clone()));
    // Overlapping roots are legal, as are configured directory symlinks.
    symlink(&tree, root.path().join("root-link")).unwrap();
    spec.control = Some(WeightsSource::Dir(root.path().join("root-link")));
    let native = SourceInventory::capture(&spec).unwrap();
    let guarded = capture(&spec);
    assert_eq!(guarded.roots, native.roots);
    assert_eq!(guarded.entries, native.entries);
    assert_eq!(guarded.pins, native.pins);
    let hashes = crate::runtime::test_full_hash_work_count();
    for _ in 0..3 {
        checked(None, |next| guarded.verify_for_with_capacity(&spec, next))
            .0
            .unwrap();
    }
    assert_eq!(crate::runtime::test_full_hash_work_count(), hashes);
    assert!(guarded
        .entries
        .iter()
        .all(|path| !path.ends_with("._sidecar")));
}

#[test]
fn guarded_fresh_file_directory_and_prelude_refuse_before_every_requested_allocation() {
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("file");
    std::fs::write(&file, b"small fixture payload").unwrap();
    for source in [
        WeightsSource::File(file),
        WeightsSource::Dir(root.path().to_owned()),
    ] {
        let spec = LoadSpec::new(source);
        let (_, bills, _) = checked(None, |next| {
            SourceInventory::capture_with_capacity(&spec, next)
        });
        for refusal in 0..bills {
            let (result, _, _) = checked(Some(refusal), |next| {
                SourceInventory::capture_with_capacity(&spec, next)
            });
            assert!(
                matches!(result, Err(Error::Canceled)),
                "allocation boundary {refusal}"
            );
        }
    }
    let (_, bills, _) = checked(None, |next| {
        SourceInventory::preflight_directory_with_capacity(root.path(), next)
    });
    for refusal in 0..bills {
        assert!(matches!(
            checked(Some(refusal), |next| {
                SourceInventory::preflight_directory_with_capacity(root.path(), next)
            })
            .0,
            Err(Error::Canceled)
        ));
    }
}

#[test]
fn guarded_membership_detects_many_long_extras_hidden_names_and_all_mutations() {
    for mutation in [
        "extra",
        "missing",
        "renamed",
        "replaced",
        "in-place",
        "type",
        "directory-link",
    ] {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("sealed");
        std::fs::write(&file, b"original").unwrap();
        std::fs::create_dir(root.path().join("empty")).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        let inventory = capture(&spec);
        std::fs::write(root.path().join(".ignored"), b"not a member").unwrap();
        checked(None, |next| inventory.verify_unchanged_with_capacity(next))
            .0
            .unwrap();
        match mutation {
            "extra" => {
                for index in 0..257 {
                    std::fs::write(
                        root.path().join(format!("{index:03}-{}", "x".repeat(240))),
                        b"extra",
                    )
                    .unwrap();
                }
            }
            "missing" => std::fs::remove_file(&file).unwrap(),
            "renamed" => std::fs::rename(&file, root.path().join("renamed")).unwrap(),
            "replaced" => {
                let replacement = root.path().join("replacement");
                std::fs::write(&replacement, b"original").unwrap();
                std::fs::rename(replacement, &file).unwrap();
            }
            "in-place" => std::fs::write(&file, b"mutated!").unwrap(),
            "type" => {
                std::fs::remove_file(&file).unwrap();
                std::fs::create_dir(&file).unwrap();
            }
            "directory-link" => {
                std::fs::remove_dir(root.path().join("empty")).unwrap();
                symlink(root.path(), root.path().join("empty")).unwrap();
            }
            _ => unreachable!(),
        }
        assert!(
            checked(None, |next| inventory.verify_unchanged_with_capacity(next))
                .0
                .is_err(),
            "{mutation}"
        );
    }
}

#[test]
fn guarded_pin_preserves_parent_entry_resolution_target_and_symlink_policy() {
    let root = tempfile::tempdir().unwrap();
    let a = root.path().join("a");
    let b = root.path().join("b");
    std::fs::create_dir(&a).unwrap();
    std::fs::create_dir(&b).unwrap();
    std::fs::write(a.join("weights"), b"same").unwrap();
    std::fs::write(b.join("weights"), b"same").unwrap();
    let link = root.path().join("parent");
    symlink(&a, &link).unwrap();
    let spec = LoadSpec::new(WeightsSource::File(link.join("weights")));
    let inventory = capture(&spec);
    std::fs::remove_file(&link).unwrap();
    symlink(&b, &link).unwrap();
    assert!(
        checked(None, |next| inventory.verify_unchanged_with_capacity(next))
            .0
            .is_err()
    );
    // A symlinked *subdirectory* is forbidden, even if its current target is valid.
    let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
    assert!(
        matches!(checked(None, |next| SourceInventory::capture_with_capacity(&spec, next)).0,
        Err(Error::Unsupported(reason)) if reason.contains("symlinked subdirectory"))
    );
}

#[test]
fn guarded_failed_read_still_checks_after_and_prepared_tokens_are_reused() {
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("sealed");
    std::fs::write(&file, b"original").unwrap();
    let mut spec = LoadSpec::new(WeightsSource::File(file.clone()));
    spec.prepare_file_sources().unwrap();
    let hashes = crate::runtime::test_full_hash_work_count();
    let inventory = capture(&spec);
    assert_eq!(crate::runtime::test_full_hash_work_count(), hashes);
    let original = spec.prepared_file_pins().get(&file).unwrap();
    let retained = inventory.member_pin(&file).unwrap().unwrap();
    assert_eq!(original, retained);
    let mut called = false;
    // Fixture IO intentionally outside allocator policing; the guard's read closure
    // is caller-owned work, not the metadata allocation law under test.
    let result: Result<()> = inventory.read_unchanged_with_capacity(&mut |_| Ok(()), |_| {
        called = true;
        std::fs::write(&file, b"mutated!")?;
        Err(Error::Canceled)
    });
    assert!(called);
    assert!(
        matches!(result, Err(Error::Unsupported(_))),
        "post-read identity wins over read error"
    );
    let mut called = false;
    let result: Result<()> =
        inventory.read_unchanged_with_capacity(&mut |_| Err(Error::Canceled), |_| {
            called = true;
            Ok(())
        });
    assert!(matches!(result, Err(Error::Canceled)));
    assert!(!called);
}

#[test]
fn guarded_relative_and_non_normalized_literal_binding_matches_native() {
    let root = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
    std::fs::create_dir(root.path().join("dir")).unwrap();
    std::fs::write(root.path().join("sealed"), b"source").unwrap();
    let cwd = std::env::current_dir().unwrap();
    let relative = root
        .path()
        .strip_prefix(cwd)
        .unwrap()
        .join("dir/.././sealed");
    let spec = LoadSpec::new(WeightsSource::File(relative.clone()));
    let native = SourceInventory::capture(&spec).unwrap();
    let inventory = capture(&spec);
    assert_eq!(inventory.roots, native.roots);
    assert_eq!(inventory.pins, native.pins);
    checked(None, |next| inventory.verify_for_with_capacity(&spec, next))
        .0
        .unwrap();
    let retained = checked(None, |next| {
        inventory.member_pin_with_capacity(&relative, next)
    })
    .0
    .unwrap();
    assert!(retained.is_some());
    let changed = LoadSpec::new(WeightsSource::File(root.path().join("sealed")));
    assert!(
        checked(None, |next| inventory
            .verify_for_with_capacity(&changed, next))
        .0
        .is_err(),
        "canonical alias must not silently replace the configured lexical binding"
    );
}

#[test]
fn guarded_full_spec_clone_accounts_empty_split_deep_maps_and_nested_quality_owners() {
    use crate::memory_strategy::*;
    for count in [0, 1, 11, 12, 64, 1024] {
        let mut spec = LoadSpec::new(WeightsSource::File("/clone-only/not-opened".into()));
        spec.resolved_route = Some("resolved route".repeat(211));
        spec.control = Some(WeightsSource::Dir("control".repeat(211).into()));
        spec.extra_controls = vec![WeightsSource::File("extra".repeat(311).into()); 3];
        spec.ip_adapter = Some(WeightsSource::Dir("ip".repeat(211).into()));
        spec.text_encoder = Some(WeightsSource::File("encoder".repeat(211).into()));
        spec.pid = Some(crate::PidWeights {
            checkpoint: WeightsSource::File("checkpoint".repeat(211).into()),
            gemma: WeightsSource::Dir("gemma".repeat(211).into()),
        });
        spec.identity = Some(crate::IdentityWeights {
            encoder: Some(WeightsSource::File("identity".repeat(211).into())),
            eva: Some(WeightsSource::Dir("eva".repeat(211).into())),
            face_dir: Some(WeightsSource::Dir("face".repeat(211).into())),
        });
        spec.adapters = vec![crate::AdapterSpec::new(
            "adapter".repeat(211).into(),
            0.75,
            crate::AdapterKind::Lora,
        )
        .with_pass_scales(vec![0.1; 17])
        .with_moe_expert(crate::MoeExpert::High)];
        let artifact = MemoryDecodeArtifactIdentity {
            repository: "repo".repeat(211),
            revision: "revision".repeat(211),
            variant: "variant".repeat(211),
            fingerprint: "fingerprint".repeat(211),
        };
        spec.decode_quality_runtime_identity = Some(MemoryDecodeQualityRuntimeIdentity {
            artifact: artifact.clone(),
        });
        spec.decode_geometry_policies = vec![
            MemoryDecodeGeometryPolicy {
                quality_abi: 1,
                family: "family".repeat(211),
                resolved_route: "route".repeat(211),
                backend: MemoryBackend::Mlx,
                tier: MemoryNumericTier {
                    precision: crate::Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[]
                },
                load_shape: crate::LoadShape::EagerMaterialization,
                artifact,
                implementation_fingerprint: "implementation".repeat(211),
                mode: MemoryMode::Other("custom-mode".repeat(211)),
                overlay: Some("overlay".repeat(211)),
                geometry: MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    reference_count: 0
                },
                use_pid: false,
                tile_edge: 256,
                overlap: 32,
                metric: "metric".repeat(211),
                maximum_error: 1,
                fixtures: vec![
                    MemoryDecodeQualityFixture {
                        seed: 4,
                        production_latent_provenance: "provenance".repeat(211),
                        production_latent_sha256: "latent".repeat(211),
                        dense_output_sha256: "dense".repeat(211),
                        tiled_output_sha256: "tiled".repeat(211),
                        observed_error: 0,
                    };
                    3
                ],
                production_evidence_sha256: "evidence".repeat(211),
                disposition: MemoryDecodeQualityDisposition::Refused {
                    reason: "reason".repeat(211)
                },
            };
            2
        ];
        spec.decode_geometry_policies.reserve(9);
        for index in 0..count {
            spec.components.insert(
                format!("{index:05}-{}", "key".repeat(211)),
                WeightsSource::Dir(format!("/{}", "long-component-path".repeat(211)).into()),
            );
        }
        let (copy, bills, allocations) = checked(None, |next| spec.clone_with_capacity(next));
        let copy = copy.unwrap();
        assert_eq!(bills, 1, "one structural aggregate, before Clone");
        assert!(allocations > 0);
        assert_eq!(copy.resolved_route, spec.resolved_route);
        assert_eq!(copy.components, spec.components);
        assert_eq!(copy.decode_geometry_policies, spec.decode_geometry_policies);
        assert_eq!(
            copy.decode_quality_runtime_identity,
            spec.decode_quality_runtime_identity
        );
        assert_eq!(copy.adapters[0].pass_scales, spec.adapters[0].pass_scales);
        let (refused, _, allocations) = checked(Some(0), |next| spec.clone_with_capacity(next));
        assert!(matches!(refused, Err(Error::Canceled)));
        assert_eq!(
            allocations, 0,
            "refusal must precede even the first clone owner"
        );
    }
}

#[test]
fn guarded_inventory_clone_binding_and_shared_spec_clone_are_all_prechecked() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("source");
    std::fs::write(&source, b"fixture").unwrap();
    let mut spec = LoadSpec::new(WeightsSource::File(source.clone()));
    let inventory = capture(&spec);
    checked(None, |next| {
        inventory.preflight_admission_binding_with_capacity(&spec, next)
    })
    .0
    .unwrap();
    let (copy, _, _) = checked(None, |next| inventory.clone_with_capacity(next));
    assert_eq!(copy.unwrap().pins, inventory.pins);
    let (result, bills, _) = checked(None, |next| {
        spec.bind_source_inventory_with_capacity(inventory.clone_with_capacity(next)?, next)
    });
    result.unwrap();
    assert!(spec.prepared_file_pins().is_finalized());
    assert_eq!(
        spec.prepared_file_pins().keys().collect::<Vec<_>>(),
        vec![&source]
    );
    assert_eq!(
        spec.prepared_file_pins().get(&source),
        inventory.member_pin(&source).unwrap()
    );
    let copy = checked(None, |next| spec.clone_with_capacity(next))
        .0
        .unwrap();
    assert!(
        std::ptr::eq(
            copy.source_inventory().unwrap(),
            spec.source_inventory().unwrap()
        ),
        "cloning a spec preserves the shared retained inventory"
    );
    assert!(bills > 0);
    for refusal in 0..bills {
        let mut unbound = LoadSpec::new(WeightsSource::File(source.clone()));
        let (result, _, _) = checked(Some(refusal), |next| {
            unbound.bind_source_inventory_with_capacity(inventory.clone_with_capacity(next)?, next)
        });
        assert!(matches!(result, Err(Error::Canceled)));
        assert!(unbound.source_inventory().is_none());
        assert!(
            !unbound.prepared_file_pins().is_prepared(),
            "failed bind must be atomic"
        );
    }
}

#[test]
fn guarded_fresh_many_long_entries_crosses_syscall_batches_and_keeps_sorted_membership() {
    let root = tempfile::tempdir().unwrap();
    for index in 0..257 {
        std::fs::write(
            root.path().join(format!("{index:03}-{}", "x".repeat(240))),
            b"small",
        )
        .unwrap();
    }
    let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
    let native = SourceInventory::capture(&spec).unwrap();
    let inventory = capture(&spec);
    assert_eq!(inventory.entries, native.entries);
    assert_eq!(inventory.pins, native.pins);
    checked(None, |next| inventory.verify_for_with_capacity(&spec, next))
        .0
        .unwrap();
}

#[test]
fn guarded_prepared_path_set_clone_covers_empty_split_and_deep_tree_nodes() {
    use super::bounded_clone::CloneLaw;
    for count in [0, 1, 11, 12, 64, 1024] {
        let paths: std::collections::BTreeSet<PathBuf> = (0..count)
            .map(|index| format!("{index:05}-{}", "receipt-path".repeat(211)).into())
            .collect();
        let copy = checked(None, |next| {
            next(paths.clone_bytes()?)?;
            Ok(paths.clone())
        })
        .0
        .unwrap();
        assert_eq!(copy, paths);
    }
}

#[test]
fn guarded_actual_walker_rejects_a_subdirectory_replaced_by_symlink_between_stats() {
    use std::os::unix::ffi::OsStrExt;
    let root = tempfile::tempdir().unwrap();
    let directory = root
        .path()
        .join("a-uniquely-long-subdirectory-name-for-the-race");
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(directory.join("weights"), b"fixture").unwrap();
    let moved = root.path().join(".moved-by-test");
    let original_c = std::ffi::CString::new(directory.as_os_str().as_bytes()).unwrap();
    let moved_c = std::ffi::CString::new(moved.as_os_str().as_bytes()).unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
    let mut matches = 0;
    let mut raced = false;
    let result = checked(None, |next| {
        let mut racing_check = |bytes| {
            next(bytes)?;
            // The two actual stat input buffers for this exact discovered path:
            // first following the target, then the no-follow symlink check.
            if bytes == original_c.as_bytes_with_nul().len() as u64 {
                matches += 1;
                if matches == 2 {
                    let renamed = unsafe { libc::rename(original_c.as_ptr(), moved_c.as_ptr()) };
                    if renamed != 0 {
                        return Err(std::io::Error::last_os_error().into());
                    }
                    let linked = unsafe { libc::symlink(moved_c.as_ptr(), original_c.as_ptr()) };
                    if linked != 0 {
                        return Err(std::io::Error::last_os_error().into());
                    }
                    raced = true;
                }
            }
            Ok(())
        };
        SourceInventory::capture_with_capacity(&spec, &mut racing_check)
    })
    .0;
    assert!(
        raced,
        "the test must mutate the actual walker between its two stats"
    );
    assert!(
        matches!(result, Err(Error::Unsupported(reason)) if reason.contains("symlinked subdirectory"))
    );
}

#[test]
fn guarded_capture_covers_every_configured_file_and_directory_source_slot() {
    let root = tempfile::tempdir().unwrap();
    let file = |name: &str| {
        let path = root.path().join(name);
        std::fs::write(&path, name.as_bytes()).unwrap();
        path
    };
    let directory = |name: &str| {
        let path = root.path().join(name);
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("member"), name.as_bytes()).unwrap();
        WeightsSource::Dir(path)
    };
    let mut spec = LoadSpec::new(WeightsSource::File(file("weights")));
    spec.control = Some(directory("control"));
    spec.extra_controls = vec![
        WeightsSource::File(file("extra-control")),
        directory("extra-control-dir"),
    ];
    spec.ip_adapter = Some(directory("ip-adapter"));
    spec.adapters = vec![crate::AdapterSpec::new(
        file("adapter"),
        1.0,
        crate::AdapterKind::Lora,
    )];
    spec.pid = Some(crate::PidWeights {
        checkpoint: WeightsSource::File(file("pid")),
        gemma: directory("gemma"),
    });
    spec.identity = Some(crate::IdentityWeights {
        encoder: Some(WeightsSource::File(file("identity"))),
        eva: Some(directory("eva")),
        face_dir: Some(directory("faces")),
    });
    spec.text_encoder = Some(directory("text-encoder"));
    spec.components
        .insert("named-file".into(), WeightsSource::File(file("named-file")));
    spec.components
        .insert("named-dir".into(), directory("named-directory"));
    let native = SourceInventory::capture(&spec).unwrap();
    let inventory = capture(&spec);
    assert_eq!(inventory.roots, native.roots);
    assert_eq!(inventory.entries, native.entries);
    assert_eq!(inventory.pins, native.pins);
    let mut configured = Vec::new();
    spec.visit_source_paths(|path, _| configured.push(std::path::absolute(path).unwrap()));
    configured.sort_unstable();
    configured.dedup();
    assert_eq!(
        inventory.roots, configured,
        "all declared source slots, with sorted duplicate-free lexical binding"
    );
}

#[test]
fn guarded_os_layout_and_callback_error_paths_have_no_unchecked_owned_diagnostic() {
    let root = tempfile::tempdir().unwrap();
    for source in [
        WeightsSource::File(root.path().join("missing-file")),
        WeightsSource::Dir(root.path().join("missing-directory")),
    ] {
        let spec = LoadSpec::new(source);
        assert!(matches!(
            checked(None, |next| SourceInventory::capture_with_capacity(
                &spec, next
            ))
            .0,
            Err(Error::Io(_))
        ));
    }
    let result = checked(None, |next| bounded_fs::vec::<u64>(usize::MAX, next));
    assert!(
        matches!(result.0, Err(Error::Io(ref error)) if error.raw_os_error() == Some(libc::EOVERFLOW))
    );
    assert_eq!(
        result.1, 0,
        "a layout error requires neither a callback nor an allocation"
    );
    assert_eq!(result.2, 0);
    let file = root.path().join("exists");
    std::fs::write(&file, b"exists").unwrap();
    let result = checked(None, |next| {
        let mut refuse = |bytes| {
            next(bytes)?;
            Err(Error::Io(std::io::Error::from_raw_os_error(libc::ENOMEM)))
        };
        bounded_fs::is_file_at(&file, &mut refuse)
    });
    assert!(
        matches!(result.0, Err(Error::Io(ref error)) if error.raw_os_error() == Some(libc::ENOMEM)),
        "Path::is_file-style OS error handling must NEVER swallow a callback refusal"
    );
    assert_eq!(result.2, 0);
}
