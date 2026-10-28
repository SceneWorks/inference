//! One parser for boolean process-environment knobs (sc-27313).
//!
//! The measurement / experimental opt-ins (`SVD_FORCE_F16`, `SDXL_LORA_VENDORED`,
//! `WAN_ATTN_SOFTMAX_BF16`, `PID_PIXEL_POS_ABS`, …) used to each parse their own variable: some
//! treated *any* value — including `=0` — as on, others accepted only an exact `"1"`. [`env_flag`] is
//! the shared rule: a knob is on only for an explicit truthy spelling.

/// Whether `value` is an explicit truthy spelling: `1`, `true`, `yes` or `on` (case-insensitive,
/// surrounding whitespace ignored). Everything else — `0`, `false`, empty, garbage — is off.
pub fn is_truthy_flag(value: &str) -> bool {
    let value = value.trim();
    ["1", "true", "yes", "on"]
        .iter()
        .any(|truthy| value.eq_ignore_ascii_case(truthy))
}

/// Whether the boolean process-environment knob `name` is on: set to a value [`is_truthy_flag`]
/// accepts. Unset, non-UTF-8 or any other value is off.
pub fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| is_truthy_flag(&value))
}

#[cfg(test)]
mod tests {
    use super::is_truthy_flag;

    #[test]
    fn only_explicit_truthy_spellings_turn_a_knob_on() {
        for on in [
            "1", "true", "TRUE", "True", "yes", "YES", "on", " 1 ", "On\n",
        ] {
            assert!(is_truthy_flag(on), "{on:?} must be on");
        }
        for off in ["", "0", "false", "no", "off", "2", "enabled", "y", " "] {
            assert!(!is_truthy_flag(off), "{off:?} must be off");
        }
    }
}
