//! Fixed training-only balanced64 recipe; no evaluation image or seed input.
use sha2::{Digest, Sha256};
pub const PLAN_SHA256: &str = "d10ba00492eea5cf38af59964f75c671490aebfd21f8a7b0314b9ac7ab9bb078";
pub const VERSION: &str = "balanced64-native768-v2";
pub const EDGE: u32 = 768;
const LOGICAL_EDGE: u32 = 512;
pub const ITEMS: u32 = 6;
/// Native Rust PNG encoder and the existing ordered dataset receipt schema.
pub const DATASET_PNG_SHA256: &str =
    "62d762a977a23d6df0a8435e73ee2abcc026909e3558c6a1f20bfb17e091ef9c";
const MULTIPLIERS: [u32; 6] = [13, 17, 29, 37, 45, 53];
const OFFSETS: [u32; 6] = [7, 19, 31, 43, 55, 3];
const SOURCE_HASHES: [&str; 6] = [
    "4d8ef7e419ad067c19a753e8921b2738c736e11e8343f2120206b9f1582bc1ec",
    "4e1574282a97569857742a1548ed9e7d02278334e137aafe3ccd36059f18db16",
    "aefe45e5caa08e94dda6e26b175e89ae59ac052049b8fb2edb6e58d23e32b033",
    "7bb37fb6dd545f16050bec670d84b4849c462beb8962c7c2ba439ac4f025bd49",
    "3c312732ff08d0e5c79c9011a4a7326ea3c4420733e144ea14d4a5c1b18bf87d",
    "d92441ffc91f0eda81e744d78b305725bac99564471e8b8851a4a702ec52eb39",
];
const TARGET_HASHES: [&str; 6] = [
    "f7677e63d03369b231eaab8651e3d1dbca2d93a6947557869e48caf13f63c5b2",
    "2ee36ede5bfaeed0e0e3f9da23f6dd578393fd032fd7a70606d8e34bca5cb8ae",
    "917a8e1f97fccefe018efb65cfbd99289278ff816e830f14cbe5e901638ed784",
    "f1b5bafe24331c56cf8e8bddf6ebaab6a310098caf48d91331ef2e424b09b620",
    "0d44cde7e5c1ee0b2ae27485462515581fe177c71f68da2db009d21796db6782",
    "b58c5ced92159f7b361c83367478781066f01f13df7a02935648a7643518f239",
];
fn coordinates(item: u32, x: u32, y: u32) -> (u32, u32) {
    assert!(item < ITEMS && x < LOGICAL_EDGE && y < LOGICAL_EDGE);
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
/// Native 768 coordinates repeat the original 512 layout by floor(2*x/3).
/// This is deliberately not a bijection: each logical pixel occurs 1, 2 or 4 times.
fn native_pixel_parts(item: u32, x: u32, y: u32) -> ([u8; 3], [u32; 3]) {
    assert!(x < EDGE && y < EDGE);
    pixel_parts(item, x * LOGICAL_EDGE / EDGE, y * LOGICAL_EDGE / EDGE)
}
pub fn training_pixel(item: u32, x: u32, y: u32, edge: u32) -> [u8; 3] {
    assert_eq!(
        edge, EDGE,
        "only the frozen native768 training dimensions are supported"
    );
    native_pixel_parts(item, x, y).0
}
#[derive(Debug)]
pub struct Audit {
    pub source_sha256: String,
    pub target_sha256: String,
    pub source_channel_counts: Vec<Vec<usize>>,
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
            edge: EDGE,
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
            let (expected, q) = native_pixel_parts(item, x, y);
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
    assert!(colours.iter().all(|&count| count == 9216));
    assert!(input_counts[..2]
        .iter()
        .flatten()
        .all(|&count| count == 2304));
    let blue_range = if item < 2 { (2048, 2560) } else { (1536, 3072) };
    assert_eq!(*input_counts[2].iter().min().unwrap(), blue_range.0);
    assert_eq!(*input_counts[2].iter().max().unwrap(), blue_range.1);
    assert!(input_counts.iter().flatten().all(|&count| count > 0));
    let source_sha256 = format!("{:x}", Sha256::digest(source));
    let target_sha256 = format!("{:x}", Sha256::digest(target));
    assert_eq!(source_sha256, SOURCE_HASHES[item as usize]);
    assert_eq!(target_sha256, TARGET_HASHES[item as usize]);
    Audit {
        source_sha256,
        target_sha256,
        source_channel_counts: input_counts.iter().map(|c| c.to_vec()).collect(),
    }
}
#[cfg(test)]
use super::edit_protocol as canonical;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exhaustive_real_rgb_counts_canonical_transform_and_frozen_raw_hashes() {
        assert_eq!(VERSION, "balanced64-native768-v2");
        assert_eq!(
            PLAN_SHA256,
            "d10ba00492eea5cf38af59964f75c671490aebfd21f8a7b0314b9ac7ab9bb078"
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
                .any(|x| native_pixel_parts(item, x, y).1.map(|q| (85 * q) as u8)
                    != canonical::palette_pixel(x, y, EDGE))));
        }
        assert_eq!(hashes.len(), 12);
    }
    #[test]
    fn logical512_layouts_are_bijective_and_heldout_inputs_are_refused() {
        for item in 0..ITEMS {
            let mut seen = vec![false; (LOGICAL_EDGE * LOGICAL_EDGE) as usize];
            for y in 0..LOGICAL_EDGE {
                for x in 0..LOGICAL_EDGE {
                    let (u, v) = coordinates(item, x, y);
                    let index = (v * LOGICAL_EDGE + u) as usize;
                    assert!(!seen[index]);
                    seen[index] = true;
                }
            }
            assert!(seen.iter().all(|&v| v));
        }
        for item in [6, 99, 1000] {
            assert!(std::panic::catch_unwind(|| training_pixel(item, 0, 0, EDGE)).is_err());
        }
        assert!(std::panic::catch_unwind(|| training_pixel(0, 0, 0, 512)).is_err());
    }
    #[test]
    fn native768_mapping_has_exact_one_two_four_multiplicities() {
        let mut counts = vec![0u32; (LOGICAL_EDGE * LOGICAL_EDGE) as usize];
        for y in 0..EDGE {
            for x in 0..EDGE {
                let (u, v) = (x * LOGICAL_EDGE / EDGE, y * LOGICAL_EDGE / EDGE);
                counts[(v * LOGICAL_EDGE + u) as usize] += 1;
            }
        }
        for v in 0..LOGICAL_EDGE {
            for u in 0..LOGICAL_EDGE {
                let expected = (if u % 2 == 0 { 2 } else { 1 }) * (if v % 2 == 0 { 2 } else { 1 });
                assert_eq!(counts[(v * LOGICAL_EDGE + u) as usize], expected);
            }
        }
        assert_eq!(counts.iter().filter(|&&n| n == 1).count(), 65536);
        assert_eq!(counts.iter().filter(|&&n| n == 2).count(), 131072);
        assert_eq!(counts.iter().filter(|&&n| n == 4).count(), 65536);
        assert_eq!(counts.iter().sum::<u32>(), EDGE * EDGE);
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
            edge: EDGE,
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
                edge: 512,
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
