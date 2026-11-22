// Large outputs use a bounded grid and pass the full element count to the kernel.
pub(super) fn plan(elements: usize) -> (u32, &'static str) {
    match u32::try_from(elements) {
        Ok(elements) => (elements, "im2col"),
        Err(_) => (1 << 22, "im2col_large"),
    }
}

#[cfg(test)]
mod tests {
    use super::plan;

    #[test]
    fn qwen_vae_small_grids_keep_the_original_launch() {
        for side in [2048, 2224] {
            let elements = side * side * 96 * 9;
            assert_eq!(plan(elements), (u32::try_from(elements).unwrap(), "im2col"));
        }
        assert_eq!(plan(u32::MAX as usize), (u32::MAX, "im2col"));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn qwen_vae_large_grids_use_the_wide_kernel() {
        let (launch_elements, kernel) = plan(u32::MAX as usize + 1);
        assert_eq!(kernel, "im2col_large");

        // LaunchConfig uses 1024 threads per block. A thread at tail % stride
        // must still address the final output after crossing the u32 boundary.
        let stride = launch_elements.div_ceil(1024) as usize * 1024;
        assert_eq!(stride, 1 << 22);
        for side in [2240, 4096] {
            let elements = side * side * 96 * 9;
            assert_eq!(plan(elements), (launch_elements, kernel));
            let tail = elements - 1;
            let first = tail % stride;
            let iteration = tail / stride;
            assert!(first < stride);
            assert!(first + (iteration - 1) * stride < tail);
            assert_eq!(first + iteration * stride, tail);
            assert!(first + (iteration + 1) * stride >= elements);
            assert!(tail > u32::MAX as usize);
        }
    }
}
