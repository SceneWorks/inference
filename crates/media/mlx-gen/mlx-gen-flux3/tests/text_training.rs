//! Supported-host frozen teacher tests. Windows edit/fmt only; not locally executed.
use crate::common::{cpu_process, fixture, host_f32, on_cpu, tiny_config, tiny_package};
use gen_core_flux3::{assemble_window, TextWindow};
use mlx_gen_flux3::Flux3TextEncoder;
use mlx_rs::{Array, Dtype};
fn prompt(name: &str) -> &'static str {
    match name {"short"=>"a red fox in the snow","exact"=>"a red fox in the snow , a cat on a roof","long"=>"a red fox in the snow , a cat on a roof with a hat , golden hour , blue sky and sun of the","empty"=>"",_=>panic!("fixture prompt")}
}
fn close(label: &str, actual: &Array, reference: &Array, dtype: Dtype) {
    assert_eq!(actual.shape(), reference.shape());
    let (actual, reference) = (host_f32(actual), host_f32(reference));
    let peak = reference
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    let errors = actual
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .collect::<Vec<_>>();
    let maximum = errors.iter().copied().fold(0.0f32, f32::max);
    let mean = errors.iter().sum::<f32>() / errors.len() as f32;
    assert!(actual.iter().all(|x| x.is_finite()));
    // BF16 uses the existing teacher's absolute budget; F16 scales that budget
    // by its eightfold finer significand. F32 has a separate tight numeric gate.
    let (max_bound, mean_bound) = match dtype {
        Dtype::Bfloat16 => (0.265, 0.05),
        Dtype::Float16 => (0.034, 0.00625),
        _ => (2e-5 + 1e-4 * peak, 2e-6 + 1e-5 * peak),
    };
    assert!(
        maximum <= max_bound && mean <= mean_bound,
        "{label} {dtype:?}: raw max {maximum}, mean {mean}, peak {peak}"
    );
}
#[test]
fn training_teacher_real_precision_frequency_profile_and_window_match_hf_reference() {
    cpu_process("text_training::training_teacher_real_precision_frequency_profile_and_window_match_hf_reference",||on_cpu(|| {
        let cfg=tiny_config();let root=tiny_package().join("text_encoder");let golden=fixture("flux3_text_training_golden.safetensors");
        for (label,dtype) in [("f32",Dtype::Float32),("bf16",Dtype::Bfloat16),("f16",Dtype::Float16)] {
            let encoder=Flux3TextEncoder::load_for_training(&root,&cfg.text_encoder,cfg.dit.text_in_channels,dtype,24).unwrap();
            assert_eq!(encoder.compute_dtype(),dtype);assert_eq!(encoder.output_dtype(),dtype);
            for name in ["short","exact","long","empty"] {let output=encoder.encode(prompt(name)).unwrap();assert_eq!(output.tokens.dtype(),dtype);close(name,&output.tokens,golden.require(&format!("{label}/{name}/tokens")).unwrap(),dtype);}
            let short=encoder.window(prompt("short")).unwrap();let empty=encoder.window("").unwrap();
            let empty=assemble_window(&empty.input_ids[..empty.real],encoder.pad_id(),short.length(),24).unwrap();
            close("matched-window empty",&encoder.encode_window(empty).unwrap().tokens,golden.require(&format!("{label}/empty_at_short_window/tokens")).unwrap(),dtype);
            let cap16=Flux3TextEncoder::load_for_training(&root,&cfg.text_encoder,cfg.dit.text_in_channels,dtype,16).unwrap();
            let output=cap16.encode(prompt("long")).unwrap();assert_eq!(output.window.length(),16);assert!(output.window.truncated_tokens>0);
            close("cap16",&output.tokens,golden.require(&format!("{label}/long_cap16/tokens")).unwrap(),dtype);assert_ne!(encoder.precision_profile_key(),cap16.precision_profile_key());
            let ints=|field:&str| golden.require(&format!("{label}/rotary_probe/{field}")).unwrap().as_slice::<i32>().to_vec();
            let output=encoder.encode_window(TextWindow {input_ids:ints("input_ids"),mask:ints("mask"),positions:ints("positions"),real:4,truncated_tokens:0}).unwrap();
            close("positions0/127/255/511",&output.tokens,golden.require(&format!("{label}/rotary_probe/tokens")).unwrap(),dtype);
        }
    }));
}
