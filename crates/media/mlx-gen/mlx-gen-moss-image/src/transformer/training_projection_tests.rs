use super::*;
use mlx_gen::adapters::{AdaptableLinear, Adapter};
#[test]
#[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
fn custom_stack_is_fp32_scaled_and_keeps_dmd_separate() {
    let _device = crate::cpu_test_device();
    let weight = Array::from_slice(&[1f32, 0., 0., 1.], &[2, 2]);
    let mut p = Projection {
        lokr: Vec::new(),
        weight: weight.clone(),
        bias: None,
        adapter: Some((
            Array::from_slice(&[1f32, 2.], &[1, 2]),
            Array::from_slice(&[3f32, 4.], &[2, 1]),
            0.5,
        )),
        custom: AdaptableLinear::dense(weight, None),
        edit_adapter: false,
        edit_fused_bias: false,
    };
    let x = Array::from_slice(&[2f32, 1.], &[1, 2]);
    let bare_dmd = p.forward(&x).unwrap();
    p.custom.set_training_adapters(vec![Adapter::Lora {
        a: Array::from_slice(&[2f32, -1.], &[2, 1]),
        b: Array::from_slice(&[0.25f32, -0.5], &[1, 2]),
        scale: 2.,
    }]);
    let both = p.forward(&x).unwrap();
    assert_eq!(
        crate::pipeline::logical_f32_readback(&bare_dmd).unwrap(),
        vec![8., 9.]
    );
    assert_eq!(
        crate::pipeline::logical_f32_readback(&both).unwrap(),
        vec![9.5, 6.]
    );
    p.custom.set_adapters(Vec::new());
    assert_eq!(
        crate::pipeline::logical_f32_readback(&p.forward(&x).unwrap()).unwrap(),
        vec![8., 9.]
    );
    assert_eq!(
        crate::pipeline::logical_f32_readback(&p.weight).unwrap(),
        vec![1., 0., 0., 1.]
    );
    p.adapter = None;
    p.weight = p.weight.as_dtype(Dtype::Bfloat16).unwrap();
    p.custom.set_adapters(vec![Adapter::Lora {
        a: Array::from_slice(&[2f32, -1.], &[2, 1]),
        b: Array::from_slice(&[1f32, 1.], &[1, 2]),
        scale: 0.,
    }]);
    assert_eq!(
        p.forward(&x).unwrap().dtype(),
        Dtype::Bfloat16,
        "disabled adapter widened a BF16 backbone"
    );
}
