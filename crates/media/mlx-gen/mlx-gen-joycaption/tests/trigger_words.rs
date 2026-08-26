//! Backend-neutral trigger-word conformance for the MLX JoyCaption route.

#[test]
fn trigger_words_follow_gen_core_conformance_matrix() {
    for case in mlx_gen::caption::CAPTION_TRIGGER_WORD_CONFORMANCE {
        let triggers = case
            .trigger_words
            .iter()
            .map(|word| (*word).to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            mlx_gen::caption::apply_caption_trigger_words(case.caption, &triggers),
            case.expected
        );
    }
}
