//! The shared check of the per-backend speculative default (epic sc-24432 E5, story sc-24446):
//! a request that leaves the speculative option unset runs the provider's default, and anything
//! the request sets — including an explicit off, in the new or the legacy spelling — wins.

use core_llm::{
    resolve_speculative, MtpMode, ProposerKind, Sampling, Speculative, TextLlm, TextLlmOutput,
    TextLlmRequest,
};

use crate::speculative::BenchPrompt;

/// Check a provider whose unset-request speculative default is `default`: an unset request runs
/// what `default` resolves to on this model; `speculative: Some(Off)` and the legacy
/// `mtp: Some(MtpMode::Off)` run plain; all three emit the same greedy text (E1). Run it once with
/// the provider as loaded (`default` = its defaults-table row) and once after setting a default
/// that resolves to a proposer, so the hook is observable whatever the table says.
pub fn check_speculative_default(
    provider: &dyn TextLlm,
    default: Speculative,
    prompt: &BenchPrompt,
    max_new_tokens: u32,
) {
    let expected = resolve_speculative(default, &provider.descriptor().capabilities)
        .plan
        .proposer();
    let run = |speculative: Option<Speculative>, mtp: Option<MtpMode>| -> TextLlmOutput {
        let request = TextLlmRequest {
            messages: prompt.messages.clone(),
            sampling: Sampling::greedy(),
            max_new_tokens,
            seed: Some(0),
            speculative,
            mtp,
            ..Default::default()
        };
        provider.generate(&request, &mut |_| {}).unwrap()
    };
    let proposer = |out: &TextLlmOutput| out.decode.as_ref().expect("a decode report").proposer;

    let unset = run(None, None);
    assert_eq!(
        proposer(&unset),
        expected,
        "an unset option runs the default"
    );
    let explicit_off = run(Some(Speculative::Off), None);
    assert_eq!(proposer(&explicit_off), ProposerKind::None, "explicit off");
    let legacy_off = run(None, Some(MtpMode::Off));
    assert_eq!(
        proposer(&legacy_off),
        ProposerKind::None,
        "an explicit legacy `mtp: Off` is an explicit off"
    );
    assert_eq!(unset.text, explicit_off.text, "greedy-identical (E1)");
    assert_eq!(legacy_off.text, explicit_off.text);
}
