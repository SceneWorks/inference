//! Provider-private PiD replay work basis.
//!
//! Carries a source input dimension, NEVER prompt contents or a tensor/scalar payload. Existing
//! flow1 recipe serialization and public entrypoint signatures remain unchanged. Only PiD-active
//! Z work tokens use the new opaque namespace; old successful non-PiD tokens pass through as-is.
use mlx_gen::{Error, Result};
use mlx_gen::sampler::flow_sampling::FlowSamplingRecipe;

const PREFIX: &str = "zpidwork1-";
// Pinned flow_sampling::decode_recipe_token accepts at most 16384 inner hex bytes.
// A canonical inner token is "flow1-" + that hex (no JSON escapes). The private
// tuple [1,"<flow token>",<u64>] adds seven punctuation/version bytes and at most
// twenty decimal digits. This bounds decoding allocation, NOT prompt length.
const MAX_INNER_HEX_BYTES: usize = 16384;
const MAX_INNER_TOKEN_BYTES: usize = "flow1-".len() + MAX_INNER_HEX_BYTES;
const MAX_WIRE_BYTES: usize = MAX_INNER_TOKEN_BYTES + 7 + 20;
const MAX_OUTER_HEX_BYTES: usize = MAX_WIRE_BYTES * 2;
const MAX_TOKEN_BYTES: usize = PREFIX.len() + MAX_OUTER_HEX_BYTES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PidWorkBasis {
    /// Exact ORIGINAL user caption UTF-8 length, BEFORE CHI concatenation/tokenization/truncation.
    pub(super) prompt_utf8_bytes: u64,
}
impl PidWorkBasis {
    pub(super) fn actual(req:&mlx_gen::GenerationRequest)->Self {
        Self {prompt_utf8_bytes:req.prompt.len() as u64}
    }
    pub(super) fn require(value:Option<Self>)->Result<Self> {
        value.ok_or_else(||Error::Msg("PiD token replay requires actual pre-truncation prompt byte work basis (zpidwork1 token); missing is not empty/300/OS capacity".into()))
    }
}

fn flow_uses_pid(recipe:&FlowSamplingRecipe)->Result<bool> {
    // Projection of the EXISTING validated serialized controls, without a synthetic request or
    // new shared accessor/DTO. No payload is read and no native policy/default is changed.
    let value=serde_json::to_value(recipe).map_err(|error|Error::Msg(error.to_string()))?;
    value.get("use_pid").and_then(serde_json::Value::as_bool)
        .ok_or_else(||Error::Msg("validated flow recipe missing use_pid".into()))
}
fn hex_encode(bytes:&[u8])->String {
    const HEX:&[u8;16]=b"0123456789abcdef";
    let mut output=String::with_capacity(bytes.len()*2);
    for &byte in bytes {output.push(HEX[(byte>>4) as usize] as char);output.push(HEX[(byte&15) as usize] as char);}
    output
}
fn hex_decode(text:&str)->Result<Vec<u8>> {
    // Reject length and noncanonical bytes BEFORE collecting a Vec or entering serde.
    if text.len()>MAX_OUTER_HEX_BYTES {return Err(Error::Msg("PiD work token exceeds bounded wrapper encoding".into()));}
    if text.len()%2!=0 {return Err(Error::Msg("PiD work token has odd hex length".into()));}
    if !text.bytes().all(|byte|byte.is_ascii_digit()||(b'a'..=b'f').contains(&byte)) {
        return Err(Error::Msg("PiD work token requires canonical lower hex".into()));
    }
    let nibble=|byte:u8|->u8 {if byte<=b'9' {byte-b'0'} else {byte-b'a'+10}};
    Ok(text.as_bytes().chunks_exact(2).map(|pair|(nibble(pair[0])<<4)|nibble(pair[1])).collect())
}

/// Private tuple wire encoding (no new public API): serialize
/// (1u8, EXISTING canonical flow1 token, actual request.prompt.len() as u64), lower hex, prefix.
/// Canonical A owns coherent outer-token preservation and inner solver replay; this codec
/// alone does not establish canonical wiring or derive actual bytes from a synthetic request.
pub(super) fn encode(flow:&FlowSamplingRecipe,basis:PidWorkBasis)->Result<String> {
    let flow_token=flow.canonical_token();
    if !flow_uses_pid(flow)? {return Ok(flow_token);}
    if flow_token.len()>MAX_INNER_TOKEN_BYTES {return Err(Error::Msg("PiD work token inner flow exceeds existing decoder bound".into()));}
    let wire=serde_json::to_vec(&(1u8,flow_token,basis.prompt_utf8_bytes)).map_err(|error|Error::Msg(error.to_string()))?;
    Ok(format!("{PREFIX}{}",hex_encode(&wire)))
}

/// Decode with no GenerationRequest construction. In actual-request mode an independently supplied
/// actual basis MUST match the token's claimed dimension, preventing capture/key laundering.
/// A legacy flow1 PiD token returns None, not a fabricated zero: the private PiD path requires its
/// basis only if it actually reaches an active base-PiD decoder. Control's native non-PiD path is
/// not widened by a serialized flag or a LoadSpec field.
pub(super) fn decode(token:&str,actual:Option<PidWorkBasis>)->Result<(FlowSamplingRecipe,Option<PidWorkBasis>)> {
    if let Some(encoded)=token.strip_prefix(PREFIX) {
        if token.len()>MAX_TOKEN_BYTES {return Err(Error::Msg("PiD work token exceeds bounded wrapper encoding".into()));}
        let bytes=hex_decode(encoded)?;
        let (version,flow_token,prompt_utf8_bytes):(u8,String,u64)=serde_json::from_slice(&bytes)
            .map_err(|error|Error::Msg(format!("invalid PiD work token: {error}")))?;
        if version!=1 {return Err(Error::Msg("unsupported provider-private PiD work token version".into()));}
        let flow=FlowSamplingRecipe::from_token(&flow_token)?;
        if !flow_uses_pid(&flow)? {return Err(Error::Msg("PiD work token wraps a non-PiD flow recipe".into()));}
        let basis=PidWorkBasis {prompt_utf8_bytes};
        if actual.is_some_and(|value|value!=basis) {return Err(Error::Msg("PiD work token prompt-byte basis differs from actual request".into()));}
        // Strict tuple length/version and canonical re-encoding; no hidden extra work facts.
        if encode(&flow,basis)?!=token {return Err(Error::Msg("noncanonical PiD work token".into()));}
        Ok((flow,Some(basis)))
    } else {
        Ok((FlowSamplingRecipe::from_token(token)?,actual))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn flow(use_pid:bool)->FlowSamplingRecipe {
        let req=mlx_gen::GenerationRequest {use_pid,..Default::default()};
        crate::sampling::resolve(&req).unwrap().recipe().clone()
    }
    #[test]
    fn actual_utf8_input_bytes_roundtrip_without_prompt_or_synthetic_replay_request() {
        let req=mlx_gen::GenerationRequest {use_pid:true,prompt:"é界".into(),..Default::default()};
        let basis=PidWorkBasis::actual(&req);assert_eq!(basis.prompt_utf8_bytes,5);
        let recipe=flow(true);let token=encode(&recipe,basis).unwrap();
        let (decoded,work)=decode(&token,None).unwrap();assert_eq!(decoded,recipe);assert_eq!(work,Some(basis));
        assert!(decode(&token,Some(PidWorkBasis {prompt_utf8_bytes:4})).is_err());
    }
    #[test]
    fn missing_is_not_empty_or_fixed300_and_old_successful_non_pid_token_is_identical() {
        let recipe=flow(false);let raw=recipe.canonical_token();
        assert_eq!(encode(&recipe,PidWorkBasis {prompt_utf8_bytes:9001}).unwrap(),raw);
        assert_eq!(decode(&raw,None).unwrap(),(recipe,None));
        let legacy_pid=flow(true).canonical_token();let (_,work)=decode(&legacy_pid,None).unwrap();
        assert_eq!(work,None);assert!(PidWorkBasis::require(work).is_err());
        let empty=PidWorkBasis {prompt_utf8_bytes:0};assert_eq!(PidWorkBasis::require(Some(empty)).unwrap(),empty);
        let (_,work)=decode(&encode(&flow(true),empty).unwrap(),None).unwrap();assert_eq!(work,Some(empty));
    }
    #[test]
    fn canonical_a_tuple_encoding_is_accepted_by_the_existing_provider_replay_codec() {
        // Independent owner-side wire recipe: no private helper called to build this expectation.
        let flow=flow(true);let bytes=serde_json::to_vec(&(1u8,flow.canonical_token(),513u64)).unwrap();
        let text=bytes.iter().map(|b|format!("{b:02x}")).collect::<String>();
        let token=format!("zpidwork1-{text}");
        assert_eq!(decode(&token,None).unwrap().1,Some(PidWorkBasis {prompt_utf8_bytes:513}));
        let bad=serde_json::to_vec(&(2u8,flow.canonical_token(),513u64)).unwrap();
        let text=bad.iter().map(|b|format!("{b:02x}")).collect::<String>();
        assert!(decode(&format!("zpidwork1-{text}"),None).is_err());
    }
    #[test]
    fn outer_decode_bound_is_inner_limit_plus_tuple_u64_not_a_prompt_cap() {
        let inner=format!("flow1-{}","0".repeat(MAX_INNER_HEX_BYTES));
        let wire=serde_json::to_vec(&(1u8,inner,u64::MAX)).unwrap();
        assert_eq!(wire.len(),MAX_WIRE_BYTES);
        let hex=hex_encode(&wire);
        assert_eq!(hex.len(),MAX_OUTER_HEX_BYTES);
        assert_eq!(hex_decode(&hex).unwrap(),wire);
        let exact=format!("{PREFIX}{hex}");
        assert_eq!(exact.len(),MAX_TOKEN_BYTES);
        // Exactly bounded encoding reaches inner validation (this synthetic inner is invalid).
        assert!(!decode(&exact,None).unwrap_err().to_string().contains("bounded wrapper"));
        let over=format!("{exact}00");
        assert!(decode(&over,None).unwrap_err().to_string().contains("bounded wrapper"));
        assert!(hex_decode(&format!("{hex}00")).unwrap_err().to_string().contains("bounded wrapper"));
        assert!(hex_decode(&"0".repeat(MAX_OUTER_HEX_BYTES+1)).unwrap_err().to_string().contains("bounded wrapper"));
        // Full u64 range remains a valid source dimension; there is no prompt-length policy cap.
        let basis=PidWorkBasis {prompt_utf8_bytes:u64::MAX};
        let token=encode(&flow(true),basis).unwrap();
        assert_eq!(decode(&token,None).unwrap().1,Some(basis));
    }
    #[test]
    fn malformed_outer_tokens_preserve_prefix_lowerhex_tuple_and_version_checks() {
        let recipe=flow(true);
        let token=encode(&recipe,PidWorkBasis {prompt_utf8_bytes:42}).unwrap();
        assert!(decode(&token.replacen(PREFIX,"zpidwork2-",1),None).is_err());
        assert!(decode("zpidwork1-0",None).unwrap_err().to_string().contains("odd hex"));
        assert!(decode("zpidwork1-AA",None).unwrap_err().to_string().contains("lower hex"));
        assert!(decode("zpidwork1-gg",None).unwrap_err().to_string().contains("lower hex"));
        assert!(decode("zpidwork1-é",None).unwrap_err().to_string().contains("lower hex"));
        for wire in [
            serde_json::to_vec(&(2u8,recipe.canonical_token(),42u64)).unwrap(),
            serde_json::to_vec(&(1u8,recipe.canonical_token(),42u64,0u8)).unwrap(),
            serde_json::to_vec(&(1u8,recipe.canonical_token())).unwrap(),
            serde_json::to_vec(&(1u8,flow(false).canonical_token(),42u64)).unwrap(),
            format!("[1,\"{}\",18446744073709551616]",recipe.canonical_token()).into_bytes(),
            format!("[1,\"{}\",42] ",recipe.canonical_token()).into_bytes(),
        ] {
            assert!(decode(&format!("{PREFIX}{}",hex_encode(&wire)),None).is_err());
        }
    }
}
