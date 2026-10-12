//! e5rt execution — real compute on the ANE, no CoreML.

use std::path::PathBuf;

const ADD_MIL: &str = r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 4]> x, tensor<fp16, [1, 4]> y) {
        tensor<fp16, [1, 4]> z = add(x = x, y = y)[name = string("z")];
    } -> (z);
}
"#;

fn fixture() -> PathBuf {
    let dir = std::env::temp_dir().join("e5rt_test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("add.mil");
    std::fs::write(&p, ADD_MIL).unwrap();
    p
}

fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

#[test]
fn ane_executes_add() {
    let mil = fixture();
    let prog = match mil_infer::e5rt::Program::compile(
        &mil,
        mil_infer::e5rt::DeviceMask::Ane,
        &[("x", 4), ("y", 4)],
        &[("z", 4)],
        &[],
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("e5rt unavailable ({e}) — skipping");
            return;
        }
    };
    eprintln!("functions: {:?}", prog.functions);
    prog.set_input_f16("x", &[f16(1.0), f16(2.0), f16(3.0), f16(4.0)])
        .unwrap();
    prog.set_input_f16("y", &[f16(10.0), f16(20.0), f16(30.0), f16(40.0)])
        .unwrap();
    prog.execute().unwrap();
    let out = prog.output_f16("z").unwrap();
    let got: Vec<f32> = out
        .iter()
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();
    assert_eq!(got, vec![11.0, 22.0, 33.0, 44.0], "ANE computed {got:?}");
}
