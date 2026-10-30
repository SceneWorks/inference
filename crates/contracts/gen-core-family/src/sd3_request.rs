//! Pure host-side SD3 request work scheduling.

/// Prepare an optional seed-independent reference exactly once, then run the per-output body with
/// the standard distinct `base_seed + output_index` identity.
///
/// Keeping preparation outside the loop is load-bearing for reference VAE encoding: an img2img
/// request with `count > 1` reuses one clean latent while noise and denoising remain per-seed.
pub fn map_sd3_seeded_outputs<R, P, O, E>(
    reference: Option<&R>,
    base_seed: u64,
    count: u32,
    prepare: impl FnOnce(&R) -> Result<P, E>,
    mut output: impl FnMut(u64, Option<&P>) -> Result<O, E>,
) -> Result<Vec<O>, E> {
    let prepared = reference.map(prepare).transpose()?;
    let mut outputs = Vec::with_capacity(count as usize);
    for index in 0..count {
        outputs.push(output(
            base_seed.wrapping_add(u64::from(index)),
            prepared.as_ref(),
        )?);
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn reference_is_prepared_once_while_each_output_gets_a_distinct_seed() {
        let prepare_count = Cell::new(0);
        let source = 21_u32;
        let mut prepared_addresses = Vec::new();
        let outputs = map_sd3_seeded_outputs(
            Some(&source),
            100,
            3,
            |value| {
                prepare_count.set(prepare_count.get() + 1);
                Ok::<_, ()>(value * 2)
            },
            |seed, prepared| {
                let prepared = prepared.expect("reference route carries prepared state");
                prepared_addresses.push(std::ptr::from_ref(prepared));
                Ok::<_, ()>((seed, *prepared))
            },
        )
        .unwrap();

        assert_eq!(prepare_count.get(), 1);
        assert_eq!(outputs, [(100, 42), (101, 42), (102, 42)]);
        assert!(prepared_addresses.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn no_reference_skips_preparation_and_preserves_wrapping_seed_identity() {
        let prepare_count = Cell::new(0);
        let outputs = map_sd3_seeded_outputs(
            None::<&u8>,
            u64::MAX,
            2,
            |_| {
                prepare_count.set(prepare_count.get() + 1);
                Ok::<_, ()>(())
            },
            |seed, prepared| {
                assert!(prepared.is_none());
                Ok::<_, ()>(seed)
            },
        )
        .unwrap();

        assert_eq!(prepare_count.get(), 0);
        assert_eq!(outputs, [u64::MAX, 0]);
    }
}
