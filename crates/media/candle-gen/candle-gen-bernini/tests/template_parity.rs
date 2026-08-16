//! sc-10995: `BerniniTemplate::encode_messages` matches the reference, bit-exact, on the real
//! Qwen2.5-VL tokenizer — candle port of the mlx lane's `template_parity`.
//!
//! Golden (`tools/dump_bernini_template_golden.py`) runs the reference `encode_messages` (verbatim,
//! indexed-pad → plain-pad remap) on the snapshot tokenizer for four task mixes (t2i / i2i / r2v /
//! rv2v). This test reproduces the same conversations via `generate_unified_inputs`, encodes them with
//! the native [`BerniniTemplate`] (plain-pad-during-assembly), and asserts `input_ids`, `token_type`,
//! `token_segment_ids`, `flex_token_types`, and the vit/vae/target-mask lists are **bit-exact**.
//!
//! Requires the converted snapshot's `mllm/tokenizer.json` (~11 MB, not committed); `#[ignore]`
//! otherwise. Point `BERNINI_MLLM_TOKENIZER` at the `tokenizer.json` and run:
//!   `cargo test -p candle-gen-bernini --test template_parity -- --ignored --nocapture`

mod common;

use common::{
    Golden, SHARED_FIXTURE_TEMPLATE_IMAGE_TOKEN_NUMS, SHARED_FIXTURE_TEMPLATE_INPUT_IMAGE_HW,
    SHARED_FIXTURE_TEMPLATE_INPUT_VIDEO_COUNT, SHARED_FIXTURE_TEMPLATE_OUTPUT_H,
    SHARED_FIXTURE_TEMPLATE_OUTPUT_T, SHARED_FIXTURE_TEMPLATE_OUTPUT_W,
    SHARED_FIXTURE_TEMPLATE_PROMPTS, SHARED_FIXTURE_TEMPLATE_TASKS,
    SHARED_FIXTURE_TEMPLATE_VIDEO_TOKEN_NUMS,
};

use candle_gen_bernini::process::generate_unified_inputs;
use candle_gen_bernini::template::{BerniniTemplate, TemplateOutput};

/// (task, conversation, image_token_nums, video_token_nums) — grids match the process golden:
/// token_num = t·(h/2)·(w/2).
#[allow(clippy::type_complexity)]
fn cases() -> Vec<(&'static str, Vec<serde_json::Value>, Vec<i64>, Vec<i64>)> {
    (0..SHARED_FIXTURE_TEMPLATE_TASKS.len())
        .map(|i| {
            (
                SHARED_FIXTURE_TEMPLATE_TASKS[i],
                generate_unified_inputs(
                    SHARED_FIXTURE_TEMPLATE_PROMPTS[i],
                    SHARED_FIXTURE_TEMPLATE_INPUT_IMAGE_HW[i],
                    SHARED_FIXTURE_TEMPLATE_INPUT_VIDEO_COUNT[i],
                    SHARED_FIXTURE_TEMPLATE_OUTPUT_T[i],
                    SHARED_FIXTURE_TEMPLATE_OUTPUT_H,
                    SHARED_FIXTURE_TEMPLATE_OUTPUT_W,
                ),
                SHARED_FIXTURE_TEMPLATE_IMAGE_TOKEN_NUMS[i].to_vec(),
                SHARED_FIXTURE_TEMPLATE_VIDEO_TOKEN_NUMS[i].to_vec(),
            )
        })
        .collect()
}

#[test]
#[ignore = "needs the converted snapshot's mllm/tokenizer.json (~11 MB, not committed); set BERNINI_MLLM_TOKENIZER"]
fn template_matches_reference() {
    let g = Golden::load("template_golden");
    let tok_path = std::env::var("BERNINI_MLLM_TOKENIZER")
        .expect("set BERNINI_MLLM_TOKENIZER to the snapshot mllm/tokenizer.json");
    let tmpl = BerniniTemplate::from_tokenizer_file(&tok_path).expect("tokenizer");

    for (task, conv, img_nums, vid_nums) in cases() {
        let o: TemplateOutput = tmpl
            .encode_messages(&conv, &img_nums, &vid_nums, task)
            .expect("encode_messages");

        let ids32: Vec<i32> = o.input_ids.iter().map(|&x| x as i32).collect();
        let checks: [(&str, &Vec<i32>); 9] = [
            ("input_ids", &ids32),
            ("token_type", &o.token_type),
            ("token_segment_ids", &o.token_segment_ids),
            ("flex_token_types", &o.flex_token_types),
            ("vit_type_list", &o.vit_type_list),
            ("vit_img_and_vid_id_list", &o.vit_img_and_vid_id_list),
            ("image_target_mask", &o.image_target_mask),
            ("video_target_mask", &o.video_target_mask),
            ("vae_type_list", &o.vae_type_list),
        ];
        for (field, got) in checks {
            let key = format!("{task}.{field}");
            // Empty lists are dumped as a zero-length I32 tensor.
            let want = if g.has(&key) && g.shape(&key).iter().product::<usize>() > 0 {
                g.i32(&key)
            } else {
                Vec::new()
            };
            assert_eq!(got, &want, "{task}.{field}");
        }
        println!("{task}: L={} all 9 fields bit-exact", o.input_ids.len());
    }
}
