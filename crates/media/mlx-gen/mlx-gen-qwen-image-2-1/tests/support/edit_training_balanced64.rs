//! Fixed training-only balanced64 recipe; no evaluation image or seed input.
use sha2::{Digest, Sha256};
pub const PLAN_SHA256: &str = "6903bcda286ac03e4524b736d579a2efcce299b1484d742460899b53e3e6675e";
pub const VERSION: &str = "balanced64-v1";
pub const EDGE: u32 = 512;
pub const ITEMS: u32 = 6;
const MULTIPLIERS: [u32; 6] = [13, 17, 29, 37, 45, 53];
const OFFSETS: [u32; 6] = [7, 19, 31, 43, 55, 3];
const SOURCE_HASHES: [&str; 6] = [
    "7bb13657f750512f953eb17a0509e0869422ba51c2a61e610f1e25a291138565",
    "4dbc037a60b1e29191e9c60efadb59b6a07cd185efe3784ab986c4d9d379fadb",
    "5e1303a277233d35878f86b5d7ac0a88a26e18c3bbfdfb787199256ed411a50b",
    "be036fbe4c9c90e72ce2fe1b67ed89ed5a739a90f45954344fbd701e3ba07604",
    "462d5793375c2ca45618fa344c62cd677b99254e872c3a6d7bd8dbc5bf374dad",
    "948f08687280073439e8d16eaea85a30807a05824c754effbce993ba71c17a43",
];
const TARGET_HASHES: [&str; 6] = [
    "8e20702fad90785b277408ebf04c0b7c25bf72ee2aab60202f26437e0dab14c8",
    "69ef8c6378d394ec24aa48f39ab0279c8137a168d8a3ece91f757ec4b63580b1",
    "0b2976bb621804d5ba80124e2018ebe6f167752ba16925d12225bd5529c2382c",
    "55f593905c8ac186f552accf6d470311fe19c2fd2abad0b294e8acb3ad051f7d",
    "f336b0e45190e538d759a8615b18efbbd2020b5feafc7190667c5b686591969d",
    "795496800b8a240bb37697aeb2eafdb7cc7425634e3315767b1ab66a693a107f",
];
fn coordinates(item: u32, x: u32, y: u32) -> (u32, u32) {
    assert!(item < ITEMS && x < EDGE && y < EDGE);
    let tri = |z: u32| 64 - ((z % 256) as i32 - 128).abs();
    match item {
        0 => (x, y),
        1 => (y, 511 - x),
        2 => ((x + y) % 512, y),
        3 => (x, (x + y) % 512),
        4 => ((x as i32 + tri(y)).rem_euclid(512) as u32, y),
        5 => (x, (y as i32 + tri(x)).rem_euclid(512) as u32),
        _ => unreachable!(),
    }
}
fn pixel_parts(item: u32, x: u32, y: u32) -> ([u8; 3], [u32; 3]) {
    let (u, v) = coordinates(item, x, y);
    let tile = (v / 64) * 8 + u / 64;
    let k = (MULTIPLIERS[item as usize] * tile + OFFSETS[item as usize]) % 64;
    let q = [k / 16, (k / 4) % 4, k % 4];
    let (lu, lv) = (u % 64, v % 64);
    let offsets = [
        (lu + 3 * lv + 5 * item + tile) % 64,
        (5 * lu + lv + 11 * item + 3 * tile) % 64,
        (lu + 7 * lv + 17 * item + 5 * tile) % 64,
    ];
    let source = std::array::from_fn(|c| (64 * (3 - q[c]) + offsets[c]) as u8);
    (source, q)
}
pub fn training_pixel(item: u32, x: u32, y: u32, edge: u32) -> [u8; 3] {
    assert_eq!(
        edge, EDGE,
        "heldout/native768 dimensions must never enter training generator"
    );
    pixel_parts(item, x, y).0
}
#[derive(Debug)]
pub struct Audit {
    pub source_sha256: String,
    pub target_sha256: String,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Recipe {
    pub items: usize,
    pub rank: u32,
    pub alpha: f32,
    pub learning_rate: f32,
    pub seed: u64,
    pub checkpointing: bool,
    pub edge: u32,
    pub references: usize,
    pub steps: u32,
    pub lokr: bool,
    pub adamw: bool,
}
pub fn recipe_is_fixed(mut actual: Recipe, bounded_probe: bool) -> bool {
    if bounded_probe {
        if !(1..=3).contains(&actual.steps) {
            return false;
        }
        actual.steps = 120; // Preserve the existing explicit safety-probe selector.
    }
    actual
        == Recipe {
            items: 6,
            rank: 16,
            alpha: 16.0,
            learning_rate: 1e-4,
            seed: 42,
            checkpointing: true,
            edge: 512,
            references: 1,
            steps: 120,
            lokr: true,
            adamw: true,
        }
}
/// Checks the actual generated RGB bytes BEFORE writing a training request.
/// Targets remain authored by the existing edit_transform in the caller.
pub fn audit(item: u32, source: &[u8], target: &[u8]) -> Audit {
    assert!(item < ITEMS);
    assert_eq!(source.len(), (EDGE * EDGE * 3) as usize);
    assert_eq!(target.len(), source.len());
    let mut colours = [0usize; 64];
    let mut input_counts = [[0usize; 256]; 3];
    for y in 0..EDGE {
        for x in 0..EDGE {
            let offset = ((y * EDGE + x) * 3) as usize;
            let (expected, q) = pixel_parts(item, x, y);
            assert_eq!(&source[offset..offset + 3], expected);
            let expected_target = q.map(|c| (c * 85) as u8);
            assert_eq!(
                &target[offset..offset + 3],
                expected_target,
                "actual canonical transform must match frozen target authority"
            );
            let index = (q[0] * 16 + q[1] * 4 + q[2]) as usize;
            colours[index] += 1;
            for c in 0..3 {
                input_counts[c][source[offset + c] as usize] += 1;
            }
        }
    }
    assert!(colours.iter().all(|&count| count == 4096));
    assert!(input_counts.iter().flatten().all(|&count| count == 1024));
    let source_sha256 = format!("{:x}", Sha256::digest(source));
    let target_sha256 = format!("{:x}", Sha256::digest(target));
    assert_eq!(source_sha256, SOURCE_HASHES[item as usize]);
    assert_eq!(target_sha256, TARGET_HASHES[item as usize]);
    Audit {
        source_sha256,
        target_sha256,
    }
}
#[cfg(test)]
use super::edit_protocol as canonical;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exhaustive_real_rgb_counts_canonical_transform_and_frozen_raw_hashes() {
        assert_eq!(VERSION, "balanced64-v1");
        assert_eq!(
            PLAN_SHA256,
            "6903bcda286ac03e4524b736d579a2efcce299b1484d742460899b53e3e6675e"
        );
        let mut hashes = std::collections::BTreeSet::new();
        for item in 0..ITEMS {
            let source: Vec<u8> = (0..EDGE)
                .flat_map(|y| (0..EDGE).flat_map(move |x| training_pixel(item, x, y, EDGE)))
                .collect();
            let target: Vec<u8> = source
                .iter()
                .map(|&v| canonical::transformed_channel(v))
                .collect();
            let actual = audit(item, &source, &target);
            assert!(hashes.insert(actual.source_sha256));
            assert!(hashes.insert(actual.target_sha256));
            // Training layouts are never the canonical evaluation palette layout.
            assert!((0..EDGE).any(|y| (0..EDGE)
                .any(|x| pixel_parts(item, x, y).1.map(|q| (85 * q) as u8)
                    != canonical::palette_pixel(x, y, EDGE))));
        }
        assert_eq!(hashes.len(), 12);
    }
    #[test]
    fn every_spatial_layout_is_bijective_and_heldout_inputs_are_refused() {
        for item in 0..ITEMS {
            let mut seen = vec![false; (EDGE * EDGE) as usize];
            for y in 0..EDGE {
                for x in 0..EDGE {
                    let (u, v) = coordinates(item, x, y);
                    let index = (v * EDGE + u) as usize;
                    assert!(!seen[index]);
                    seen[index] = true;
                }
            }
            assert!(seen.iter().all(|&v| v));
        }
        for item in [6, 99, 1000] {
            assert!(std::panic::catch_unwind(|| training_pixel(item, 0, 0, EDGE)).is_err());
        }
        assert!(std::panic::catch_unwind(|| training_pixel(0, 0, 0, 768)).is_err());
    }
    #[test]
    fn fixed_recipe_rejects_every_training_parameter_drift() {
        let recipe = Recipe {
            items: 6,
            rank: 16,
            alpha: 16.0,
            learning_rate: 1e-4,
            seed: 42,
            checkpointing: true,
            edge: 512,
            references: 1,
            steps: 120,
            lokr: true,
            adamw: true,
        };
        assert!(recipe_is_fixed(recipe, false));
        let mutations = [
            Recipe { items: 5, ..recipe },
            Recipe { rank: 4, ..recipe },
            Recipe {
                alpha: 4.0,
                ..recipe
            },
            Recipe {
                learning_rate: 1e-3,
                ..recipe
            },
            Recipe { seed: 99, ..recipe },
            Recipe {
                checkpointing: false,
                ..recipe
            },
            Recipe {
                edge: 768,
                ..recipe
            },
            Recipe {
                references: 2,
                ..recipe
            },
            Recipe {
                steps: 200,
                ..recipe
            },
            Recipe {
                lokr: false,
                ..recipe
            },
            Recipe {
                adamw: false,
                ..recipe
            },
        ];
        for altered in mutations {
            assert!(!recipe_is_fixed(altered, false));
        }
        assert!(recipe_is_fixed(Recipe { steps: 2, ..recipe }, true));
        assert!(!recipe_is_fixed(Recipe { steps: 2, ..recipe }, false));
        assert!(!recipe_is_fixed(Recipe { steps: 4, ..recipe }, true));
    }
}
