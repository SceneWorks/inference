//! Backend-neutral trigger-word conformance for the Candle JoyCaption route.

#[test]
fn trigger_words_follow_gen_core_conformance_matrix() {
    for case in candle_gen::gen_core::caption::CAPTION_TRIGGER_WORD_CONFORMANCE {
        let triggers = case
            .trigger_words
            .iter()
            .map(|word| (*word).to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            candle_gen::gen_core::apply_caption_trigger_words(case.caption, &triggers),
            case.expected
        );
    }
}
