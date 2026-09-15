//! LTX prompt tokenization — the candle twin of `mlx-gen-ltx/src/tokenizer.rs` (sc-18762).
//!
//! LTX-2.3 provisioned Gemma-3 as a separate ~26.4 GB `google/gemma-3-12b-it` snapshot purely to
//! reach `tokenizer.json`. The **2.5** text encoder is self-contained: `tokenizer.json` and its
//! `tokenizer_config.json` sidecar are packed as tensors inside the single `.safetensors`, so
//! [`Ltx25Tokenizer::from_packed_te_file`] reads them with a header parse plus two seeks and needs
//! no other file.
//!
//! Both generations run the same `gen_core::tokenizer::ensure_single_leading_bos` policy, because
//! they break in opposite directions: the Gemma 4 `tokenizer.json` post-processor emits **no**
//! `<bos>` (measured on the shipped 2.5 tokenizer — a `TemplateProcessing` whose `single` is a bare
//! `$A`), while Gemma 3's emits one and would be double-BOSed by an unconditional prepend.
//!
//! Reference: `ltx_core.text_encoders.gemma.tokenizer.LTXGemmaTokenizer` @ `d1511477`.

use std::path::Path;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::gemma_assets::LtxGemmaTokenizer;
use candle_gen::gen_core::tokenizer::ensure_single_leading_bos;
use candle_gen::{CandleError, Result as CResult};

/// The backend-neutral packed-asset layer, re-exported so the 2.5 **weight** loader reaches the
/// same unpacked config/sidecars and the same key canonicalization this tokenizer used, rather than
/// re-deriving them. [`GemmaTeKeyMap`] accepts the ComfyUI-flattened and the legacy HF tower
/// spellings and turns every missed lookup into an error — the strict counterpart to the non-strict
/// load that once left 11 tower tensors at random init.
pub use candle_gen::gen_core::gemma_assets::{
    flatten_gemma4_unified_key, is_gemma_asset_key, GemmaAssets, GemmaTeKeyMap,
};

/// The literal `<bos>` piece both Gemma generations use. The id is resolved from the vocabulary,
/// never hard-coded.
pub(crate) const GEMMA_BOS_TOKEN: &str = "<bos>";

/// Resolve the `<bos>` id out of a raw HF tokenizer, failing loudly when the vocabulary has none —
/// a tokenizer without a BOS cannot satisfy the LTX encode contract.
pub(crate) fn gemma_bos_id(tokenizer: &tokenizers::Tokenizer) -> CResult<u32> {
    tokenizer.token_to_id(GEMMA_BOS_TOKEN).ok_or_else(|| {
        CandleError::Msg(format!(
            "ltx: gemma tokenizer has no {GEMMA_BOS_TOKEN} token — the encode path requires a \
             leading BOS"
        ))
    })
}

/// Apply the exactly-one-BOS policy to an already-truncated `u32` id sequence.
pub(crate) fn ensure_single_leading_bos_u32(ids: &mut Vec<u32>, bos_id: u32, max_length: usize) {
    let mut signed: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    ensure_single_leading_bos(&mut signed, bos_id as i32, max_length);
    *ids = signed.into_iter().map(|id| id as u32).collect();
}

/// The LTX-2.5 prompt tokenizer (Gemma 4), built entirely from the single-file text encoder.
pub struct Ltx25Tokenizer {
    inner: LtxGemmaTokenizer,
}

impl Ltx25Tokenizer {
    /// Unpack the tokenizer out of the single-file 2.5 text encoder (or a legacy Gemma directory
    /// root — [`GemmaAssets::load`] dispatches). Only the packed assets' byte ranges are read.
    pub fn from_packed_te_file(te_path: &Path) -> CResult<Self> {
        let assets = GemmaAssets::load(te_path).map_err(|e| CandleError::Msg(e.to_string()))?;
        Self::from_assets(&assets)
    }

    /// Build from already-loaded [`GemmaAssets`], so a caller that unpacks the file once can share
    /// the config and sidecars with the weight loader.
    pub fn from_assets(assets: &GemmaAssets) -> CResult<Self> {
        Ok(Self {
            inner: LtxGemmaTokenizer::from_assets(assets)
                .map_err(|e| CandleError::Msg(e.to_string()))?,
        })
    }

    /// The resolved `<bos>` id (2 on the shipped 2.5 tokenizer).
    pub fn bos_id(&self) -> i32 {
        self.inner.bos_id()
    }

    /// The resolved `<pad>` id (0 on the shipped 2.5 tokenizer).
    pub fn pad_id(&self) -> i32 {
        self.inner.pad_id()
    }

    /// Encode a raw prompt → `([1, max_length] u32 input_ids, [max_length] 0/1 mask)`, left-padded
    /// with exactly one leading `<bos>`. Left-padding places the real tokens at the high RoPE
    /// positions, which is what the Gemma forward was validated against. An empty prompt is legal
    /// and encodes to a lone `<bos>`, matching the reference.
    pub fn encode(
        &self,
        prompt: &str,
        max_length: usize,
        device: &Device,
    ) -> CResult<(Tensor, Vec<u32>)> {
        let out = self
            .inner
            .encode(prompt, max_length)
            .map_err(|e| CandleError::Msg(e.to_string()))?;
        let ids: Vec<u32> = out.ids.iter().map(|&id| id as u32).collect();
        let mask: Vec<u32> = out.mask.iter().map(|&m| m as u32).collect();
        let input_ids = Tensor::from_vec(ids, (1, max_length), device)?;
        Ok((input_ids, mask))
    }

    /// Detokenize ids → text, dropping special tokens.
    pub fn decode(&self, ids: &[u32]) -> CResult<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| CandleError::Msg(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny real HF WordLevel tokenizer with **no** BOS post-processor — the Gemma 4 shape.
    /// Byte-identical to `gen-core/tests/fixtures/tiny_gemma4_tokenizer.json`; inlined rather than
    /// `include_str!`d so this crate does not reach across a crate boundary for a test fixture.
    /// Regenerate both with `gen-core/tests/fixtures/gen_tiny_gemma_tokenizers.py`.
    const TINY_GEMMA4: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"<pad>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":1,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":2,"content":"<bos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":3,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3,"a":4,"red":5,"fox":6,"in":7,"the":8,"snow":9,"café":10,"日本語":11},"unk_token":"<unk>"}}"#;
    /// The same vocabulary **with** a `<bos>`-prepending post-processor — the Gemma 3 shape.
    const TINY_GEMMA3: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"<pad>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":1,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":2,"content":"<bos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":3,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"<bos>":{"id":"<bos>","ids":[2],"tokens":["<bos>"]}}},"decoder":null,"model":{"type":"WordLevel","vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3,"a":4,"red":5,"fox":6,"in":7,"the":8,"snow":9,"café":10,"日本語":11},"unk_token":"<unk>"}}"#;

    fn tokenizer(json: &str) -> tokenizers::Tokenizer {
        tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("tiny tokenizer")
    }

    #[test]
    fn bos_id_resolves_from_the_vocabulary() {
        assert_eq!(gemma_bos_id(&tokenizer(TINY_GEMMA4)).expect("bos"), 2);
    }

    #[test]
    fn policy_adds_a_missing_bos_and_never_duplicates_one() {
        // Gemma 4 shape: the raw encode has no BOS at all.
        let g4 = tokenizer(TINY_GEMMA4);
        let mut ids = g4
            .encode("a red fox in the snow", true)
            .expect("encode")
            .get_ids()
            .to_vec();
        assert_ne!(ids.first(), Some(&2), "gemma-4 raw encode must have no BOS");
        ensure_single_leading_bos_u32(&mut ids, 2, 256);
        assert_eq!(ids, vec![2, 4, 5, 6, 7, 8, 9]);

        // Gemma 3 shape: the post-processor already supplied it — the policy is a no-op.
        let g3 = tokenizer(TINY_GEMMA3);
        let mut ids = g3
            .encode("a red fox in the snow", true)
            .expect("encode")
            .get_ids()
            .to_vec();
        assert_eq!(ids.first(), Some(&2));
        ensure_single_leading_bos_u32(&mut ids, 2, 256);
        assert_eq!(ids, vec![2, 4, 5, 6, 7, 8, 9]);
        assert_eq!(ids.iter().filter(|id| **id == 2).count(), 1);

        // Prepending onto a full sequence re-truncates.
        let mut ids = vec![4u32, 5, 6, 7];
        ensure_single_leading_bos_u32(&mut ids, 2, 4);
        assert_eq!(ids, vec![2, 4, 5, 6]);
    }

    #[test]
    fn packed_tokenizer_encodes_left_padded_with_one_bos() {
        let tokenizer_config =
            r#"{"bos_token":"<bos>","eos_token":"<eos>","pad_token":"<pad>"}"#.to_string();
        let inner =
            LtxGemmaTokenizer::from_parts(TINY_GEMMA4.as_bytes(), &tokenizer_config, "tiny_gemma4")
                .expect("build");
        let tokenizer = Ltx25Tokenizer { inner };
        assert_eq!(tokenizer.bos_id(), 2);
        assert_eq!(tokenizer.pad_id(), 0);

        let device = Device::Cpu;
        let (ids, mask) = tokenizer.encode("a red fox", 6, &device).expect("encode");
        assert_eq!(ids.dims(), &[1, 6]);
        assert_eq!(
            ids.flatten_all()
                .expect("flat")
                .to_vec1::<u32>()
                .expect("vec"),
            vec![0, 0, 2, 4, 5, 6]
        );
        assert_eq!(mask, vec![0, 0, 1, 1, 1, 1]);

        // An empty prompt is legal on the 2.5 path: a lone BOS, left-padded.
        let (_, mask) = tokenizer.encode("", 4, &device).expect("encode empty");
        assert_eq!(mask, vec![0, 0, 0, 1]);
    }
}
