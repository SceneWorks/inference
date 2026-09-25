# YuE mm tokenizer (test fixture, sc-19376)

`tokenizer.json` is the mm tokenizer that ships in every YuE stage-1 tier dir and in
`SceneWorks/xcodec-mini-infer` (`mm_tokenizer_v0.2_hf/tokenizer.json`). It is a byte-fallback BPE
derived by `scripts/audio/prepare_yue_assets.py` from the upstream SentencePiece
`mm_tokenizer_v0.2_hf/tokenizer.model` (`m-a-p/xcodec_mini_infer` @
`fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5`), with the mm special tokens added at their ids.

| File | SHA-256 |
|---|---|
| `tokenizer.json` | `9df6fac7ae0b63fddf01491d30403013a0846f3b1c6a3fd28941b21877758582` |

It is committed so the id-parity tests in `src/tokenizer.rs`
(`real_tokenizer_matches_upstream_sentencepiece`, `real_prompt_builder_matches_the_reference`)
run in ordinary CI against the Python goldens in `../yue_prompt_reference.json`. Set
`YUE_S1_SNAPSHOT` to a staged stage-1 tier dir to run them against that snapshot's copy instead.

## License

Apache-2.0, © 2025 Ruibin Yuan and core contributors from M-A-P and HKUST
(https://github.com/multimodal-art-projection/YuE). `LICENSE` and `NOTICE` are the upstream files,
retained per Section 4(d) of the Apache License.
