//! Pure (start, length) plan for bounded, one-pass logical tensor readback.
// Each F32 readback is at most 16 MiB; SHA serialization uses its separate 64 KiB blocks.
const CHUNK_FLOATS: usize = 4_194_304;

pub(super) fn audit_chunks(elements: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..elements)
        .step_by(CHUNK_FLOATS)
        .map(move |start| (start, CHUNK_FLOATS.min(elements - start)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_lengths_cover_each_element_once_with_bounded_uneven_tail() {
        for elements in [
            0,
            1,
            CHUNK_FLOATS - 1,
            CHUNK_FLOATS,
            CHUNK_FLOATS + 1,
            3 * CHUNK_FLOATS + 7,
        ] {
            let mut covered = 0;
            for (start, len) in audit_chunks(elements) {
                assert_eq!(start, covered, "no gaps or overlaps");
                assert!(len > 0 && len <= CHUNK_FLOATS);
                covered += len;
                assert!(covered <= elements, "slice end stays inside the tensor");
            }
            assert_eq!(covered, elements);
        }
        assert_eq!(audit_chunks(0).count(), 0);
        assert_eq!(
            audit_chunks(3 * CHUNK_FLOATS + 7).collect::<Vec<_>>(),
            [
                (0, CHUNK_FLOATS),
                (CHUNK_FLOATS, CHUNK_FLOATS),
                (2 * CHUNK_FLOATS, CHUNK_FLOATS),
                (3 * CHUNK_FLOATS, 7)
            ]
        );
    }
}
