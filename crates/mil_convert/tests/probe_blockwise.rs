//! Scratch probe: does `constexpr_blockwise_shift_scale` accept a scale
//! blob with a smaller trailing-block dim (blockwise broadcast) and does
//! the runtime compute bit-exact `scale * (data - offset)` in fp16?
//!
//! Run: cargo test -p mil_convert --test probe_blockwise -- --ignored --nocapture

#![cfg(target_os = "macos")]

use mil_spec::{bind, Block, DType, Feature, ModelMeta, TensorType, ValueType, WeightBin, NVT};

fn f16(shape: &[i64]) -> ValueType {
    ValueType::Tensor(TensorType {
        dtype: DType::Fp16,
        shape: shape.to_vec(),
    })
}

fn compile_and_run(b: &Block, wbytes: &[u8], x: &[f32], xshape: &[i64], tag: &str) -> Vec<f32> {
    let dir = std::env::temp_dir().join(format!("bwprobe_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let inputs = [Feature {
        name: "x".into(),
        shape: xshape.to_vec(),
        dtype: DType::Fp16,
        is_state: false,
    }];
    let outputs = [Feature {
        name: "y".into(),
        shape: b
            .ops
            .last()
            .and_then(|o| {
                o.outputs.first().and_then(|o| {
                    if let ValueType::Tensor(t) = &o.ty {
                        Some(t.shape.clone())
                    } else {
                        None
                    }
                })
            })
            .unwrap(),
        dtype: DType::Fp16,
        is_state: false,
    }];
    let fin = [NVT {
        name: "x".into(),
        ty: ValueType::Tensor(TensorType {
            dtype: DType::Fp16,
            shape: xshape.to_vec(),
        }),
    }];
    let spec = mil_spec::encode_model(
        &inputs,
        &outputs,
        &[],
        b,
        &fin,
        &ModelMeta::new(10, "CoreML9"),
    );
    let pkg = dir.join("m.mlpackage");
    mil_spec::write_mlpackage(&pkg, &spec, Some(wbytes)).unwrap();
    let comp = match mil_compile::compile(&pkg, &dir.join("c")) {
        Ok(c) => c,
        Err(e) => panic!("compile failed for {tag}: {e}"),
    };
    let data: Vec<u8> = x
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let m = mil_infer::Model::load(&comp.path, mil_infer::ComputeUnits::All).unwrap();
    let pr = m
        .predict(&[mil_infer::Input {
            name: "x",
            shape: xshape,
            data: &data,
            dtype: DType::Fp16,
        }])
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    pr.outputs[0].values()
}

/// Q8_0-style: data int8 (out,in,1,1), scale fp16 (out,in/32,1,1).
/// Emits constexpr_blockwise_shift_scale + conv1x1 and compares against
/// f64 conv over exactly-dequantized weights.
#[test]
#[ignore = "manual probe"]
fn q8_0_blockwise_probe() {
    let out: i64 = 4;
    let ins: i64 = 64; // 2 blocks of 32 per row
                       // deterministic pseudo-random codes in [-127, 127]
    let codes: Vec<i8> = (0..out * ins)
        .map(|i| (((i * 37 + 11) % 253) as i32 - 126) as i8)
        .collect();
    let code_bytes: Vec<u8> = codes.iter().map(|c| *c as u8).collect();
    // scales: 2 blocks/row, distinct non-trivial fp16 values
    let scales: Vec<f32> = (0..out * ins / 32)
        .map(|i| 0.001 * ((i * 13 % 50) as f32 + 1.0))
        .collect();
    let scale_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let mut wb = WeightBin::new();
    let doff = wb.put("w", DType::Int8, &[out, ins, 1, 1], &code_bytes);
    let soff = wb.put("s", DType::Fp16, &[out, ins / 32, 1, 1], &scale_bytes);
    let wbytes = wb.finish();

    let mut b = Block::new();
    let file = "@model_path/weights/weight.bin";
    let q = b.konst_blob("w_q8", file, doff, DType::Int8, &[out, ins, 1, 1]);
    let s = b.konst_blob("w_sc", file, soff, DType::Fp16, &[out, ins / 32, 1, 1]);
    let w = b.o1(
        "constexpr_blockwise_shift_scale",
        vec![("data".into(), bind(&q).1), ("scale".into(), bind(&s).1)],
        "w",
        f16(&[out, ins, 1, 1]),
    );
    let y = b.conv1x1_s("x", &w, None, out, 1, "y");
    b.outputs = vec![y];

    // reference: y[o] = sum_i x[i] * code[o,i]*scale[o,i/32] in f64
    let x: Vec<f32> = (0..ins).map(|i| (i as f32 * 0.031) - 1.0).collect();
    let want: Vec<f64> = (0..out as usize)
        .map(|o| {
            (0..ins as usize)
                .map(|i| {
                    let deq = half::f16::from_f32(
                        codes[o * ins as usize + i] as f32
                            * scales[o * (ins as usize / 32) + i / 32],
                    )
                    .to_f32() as f64;
                    (x[i] as f64) * deq
                })
                .sum()
        })
        .collect();
    let got = compile_and_run(&b, &wbytes, &x, &[1, ins, 1, 1], "q8");
    eprintln!("want {want:?}");
    eprintln!("got  {got:?}");
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert!(
            ((*g as f64) - w).abs() < 0.01 + 0.02 * w.abs(),
            "{g} vs {w}"
        );
    }
}

/// Q4_0-style with offset: data uint8 nibbles (0..15) (out,in,1,1),
/// offset uint8 =8 same shape as scale (out,in/32,1,1), scale fp16.
/// output = scale * (data - offset).
#[test]
#[ignore = "manual probe"]
fn q4_0_blockwise_probe() {
    let out: i64 = 4;
    let ins: i64 = 64;
    let nibbles: Vec<u8> = (0..out * ins).map(|i| ((i * 29 + 5) % 16) as u8).collect();
    let scales: Vec<f32> = (0..out * ins / 32)
        .map(|i| 0.002 * ((i * 7 % 40) as f32 + 1.0))
        .collect();
    let scale_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let off_bytes: Vec<u8> = vec![8u8; (out * ins / 32) as usize];
    let mut wb = WeightBin::new();
    let doff = wb.put("w", DType::Int8, &[out, ins, 1, 1], &nibbles);
    let soff = wb.put("s", DType::Fp16, &[out, ins / 32, 1, 1], &scale_bytes);
    let ooff = wb.put("o", DType::Int8, &[out, ins / 32, 1, 1], &off_bytes);
    let wbytes = wb.finish();

    let mut b = Block::new();
    let file = "@model_path/weights/weight.bin";
    let q = b.konst_blob("w_q4", file, doff, DType::Int8, &[out, ins, 1, 1]);
    let s = b.konst_blob("w_sc", file, soff, DType::Fp16, &[out, ins / 32, 1, 1]);
    let o = b.konst_blob("w_of", file, ooff, DType::Int8, &[out, ins / 32, 1, 1]);
    let w = b.o1(
        "constexpr_blockwise_shift_scale",
        vec![
            ("data".into(), bind(&q).1),
            ("scale".into(), bind(&s).1),
            ("offset".into(), bind(&o).1),
        ],
        "w",
        f16(&[out, ins, 1, 1]),
    );
    let y = b.conv1x1_s("x", &w, None, out, 1, "y");
    b.outputs = vec![y];

    let x: Vec<f32> = (0..ins).map(|i| (i as f32 * 0.031) - 1.0).collect();
    let want: Vec<f64> = (0..out as usize)
        .map(|o| {
            (0..ins as usize)
                .map(|i| {
                    let deq = half::f16::from_f32(
                        (nibbles[o * ins as usize + i] as i32 - 8) as f32
                            * scales[o * (ins as usize / 32) + i / 32],
                    )
                    .to_f32() as f64;
                    (x[i] as f64) * deq
                })
                .sum()
        })
        .collect();
    let got = compile_and_run(&b, &wbytes, &x, &[1, ins, 1, 1], "q4u8");
    eprintln!("want {want:?}");
    eprintln!("got  {got:?}");
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert!(
            ((*g as f64) - w).abs() < 0.01 + 0.02 * w.abs(),
            "{g} vs {w}"
        );
    }
}

/// Same as q4_0_blockwise_probe but with PACKED uint4 data + uint4 offset.
/// ggml packs (elem j → low nibble, elem j+16 → high); MIL nibble order
/// probed as (2j → low, 2j+1 → high). If the runtime rejects uint4 data
/// this test fails at compile — the int8 path is the fallback.
#[test]
#[ignore = "manual probe"]
fn q4_0_uint4_probe() {
    let out: i64 = 4;
    let ins: i64 = 64;
    let nibbles: Vec<u8> = (0..out * ins).map(|i| ((i * 29 + 5) % 16) as u8).collect();
    // MIL packing guess: byte j = lo(elem 2j) | hi(elem 2j+1)
    let packed: Vec<u8> = (0..out * ins / 2)
        .map(|j| nibbles[2 * j as usize] | (nibbles[2 * j as usize + 1] << 4))
        .collect();
    let scales: Vec<f32> = (0..out * ins / 32)
        .map(|i| 0.002 * ((i * 7 % 40) as f32 + 1.0))
        .collect();
    let scale_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    // uint4 offset = 8 per element → packed 0x88
    let off_bytes: Vec<u8> = vec![0x88u8; (out * ins / 64) as usize];
    let mut wb = WeightBin::new();
    let doff = wb.put("w", DType::Uint4, &[out, ins, 1, 1], &packed);
    let soff = wb.put("s", DType::Fp16, &[out, ins / 32, 1, 1], &scale_bytes);
    let ooff = wb.put("o", DType::Uint4, &[out, ins / 32, 1, 1], &off_bytes);
    let wbytes = wb.finish();

    let mut b = Block::new();
    let file = "@model_path/weights/weight.bin";
    let q = b.konst_blob("w_q4", file, doff, DType::Uint4, &[out, ins, 1, 1]);
    let s = b.konst_blob("w_sc", file, soff, DType::Fp16, &[out, ins / 32, 1, 1]);
    let o = b.konst_blob("w_of", file, ooff, DType::Uint4, &[out, ins / 32, 1, 1]);
    let w = b.o1(
        "constexpr_blockwise_shift_scale",
        vec![
            ("data".into(), bind(&q).1),
            ("scale".into(), bind(&s).1),
            ("offset".into(), bind(&o).1),
        ],
        "w",
        f16(&[out, ins, 1, 1]),
    );
    let y = b.conv1x1_s("x", &w, None, out, 1, "y");
    b.outputs = vec![y];

    let x: Vec<f32> = (0..ins).map(|i| (i as f32 * 0.031) - 1.0).collect();
    let want: Vec<f64> = (0..out as usize)
        .map(|o| {
            (0..ins as usize)
                .map(|i| {
                    let deq = half::f16::from_f32(
                        (nibbles[o * ins as usize + i] as i32 - 8) as f32
                            * scales[o * (ins as usize / 32) + i / 32],
                    )
                    .to_f32() as f64;
                    (x[i] as f64) * deq
                })
                .sum()
        })
        .collect();
    let got = compile_and_run(&b, &wbytes, &x, &[1, ins, 1, 1], "q4u4");
    eprintln!("want {want:?}");
    eprintln!("got  {got:?}");
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert!(
            ((*g as f64) - w).abs() < 0.01 + 0.02 * w.abs(),
            "{g} vs {w}"
        );
    }
}
