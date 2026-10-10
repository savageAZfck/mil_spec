//! Scratch probe: EnumeratedShapes — does coremlc accept field-21
//! enumeratedShapes on multiArrayType, and does the program compute
//! correctly at non-default shapes when shape values are baked into
//! consts vs computed at runtime?

#![cfg(target_os = "macos")]

use mil_spec::{
    bind, bind_many, Block, DType, EnumeratedShapes, Feature, ModelMeta, TensorType, ValueType, NVT,
};
use std::collections::BTreeMap;
use std::path::Path;

fn tt(dt: DType, shape: &[i64]) -> ValueType {
    ValueType::Tensor(TensorType {
        dtype: dt,
        shape: shape.to_vec(),
    })
}
fn f16(shape: &[i64]) -> ValueType {
    tt(DType::Fp16, shape)
}

fn pkg_for(
    tag: &str,
    b: &Block,
    inputs: &[Feature],
    outputs: &[Feature],
    fin: &[NVT],
    flex: &BTreeMap<String, EnumeratedShapes>,
    syms: &BTreeMap<String, Vec<Option<String>>>,
) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("esprobe_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let spec = mil_spec::encode_model_flex(
        inputs,
        outputs,
        &[],
        b,
        fin,
        &ModelMeta::new(10, "CoreML9"),
        flex,
        syms,
    );
    let pkg = dir.join("m.mlpackage");
    mil_spec::write_mlpackage(&pkg, &spec, None).unwrap();
    match mil_compile::compile(&pkg, &dir.join("c")) {
        Ok(c) => {
            eprintln!("[{tag}] compiled via {}", c.backend);
            c.path
        }
        Err(e) => panic!("[{tag}] compile failed: {e}"),
    }
}

fn predict_at(compiled: &Path, inputs: &[mil_infer::Input]) -> Result<Vec<f32>, String> {
    let m = mil_infer::Model::load(compiled, mil_infer::ComputeUnits::All)
        .map_err(|e| e.to_string())?;
    match m.predict(inputs) {
        Ok(p) => Ok(p.outputs[0].values()),
        Err(e) => Err(e.to_string()),
    }
}

fn fx(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect()
}

/// A. x + x with declared output (1,4,2,1); inputs enumerated {s=2, s=4}.
/// Question: does predict at s=4 work when no const bakes s?
#[test]
#[ignore = "manual probe"]
fn enum_elementwise() {
    let mut b = Block::new();
    let y = b.add("x", "x", &[1, 4, 2, 1], "y");
    b.outputs = vec![y];
    let inputs = [Feature {
        name: "x".into(),
        shape: vec![1, 4, 2, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let outputs = [Feature {
        name: "y".into(),
        shape: vec![1, 4, 2, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let fin = [NVT {
        name: "x".into(),
        ty: f16(&[1, 4, 2, 1]),
    }];
    let mut flex = BTreeMap::new();
    let es = EnumeratedShapes {
        shapes: vec![vec![1, 4, 2, 1], vec![1, 4, 4, 1]],
    };
    flex.insert(
        "x".to_string(),
        EnumeratedShapes {
            shapes: es.shapes.clone(),
        },
    );
    flex.insert("y".to_string(), es);
    // Every seq-varying op output must declare the dim symbolic — with
    // a constant declared dim the runtime allocates the buffer at the
    // DEFAULT shape and silently truncates (verified: removing "y"'s
    // sym gives an 8-element output at s=4).
    let mut syms = BTreeMap::new();
    for n in ["x", "y"] {
        syms.insert(n.to_string(), vec![None, None, Some("s".into()), None]);
    }
    let compiled = pkg_for("ew", &b, &inputs, &outputs, &fin, &flex, &syms);

    let x2: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let xb = fx(&x2);
    let r = predict_at(
        &compiled,
        &[mil_infer::Input {
            name: "x",
            shape: &[1, 4, 2, 1],
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    eprintln!("s=2: {r:?}");

    let x4: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let xb = fx(&x4);
    let r = predict_at(
        &compiled,
        &[mil_infer::Input {
            name: "x",
            shape: &[1, 4, 4, 1],
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    eprintln!("s=4: {r:?}");
    assert!(r.is_ok(), "predict at non-default seq failed: {r:?}");
    let got = r.unwrap();
    assert_eq!(got.len(), 16);
    for (g, w) in got.iter().zip(x4.iter()) {
        assert_eq!(*g, w * 2.0, "x+x must double every element");
    }
}

/// B. reshape with the seq baked into a const: works at s=2, expected
/// failure at s=4 unless CoreML re-parameterizes consts per shape.
#[test]
#[ignore = "manual probe"]
fn enum_baked_reshape() {
    let mut b = Block::new();
    // transpose is shape-free: (1,4,s,1) -> (1,s,4,1)
    let t = b.transpose("x", &[0, 2, 1, 3], &[1, 2, 4, 1], "t");
    // reshape bakes s=2 into the shape const value
    let r = b.reshape(&t, &[1, 2, 4, 1], "y");
    b.outputs = vec![r];
    let inputs = [Feature {
        name: "x".into(),
        shape: vec![1, 4, 2, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let outputs = [Feature {
        name: "y".into(),
        shape: vec![1, 2, 4, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let fin = [NVT {
        name: "x".into(),
        ty: f16(&[1, 4, 2, 1]),
    }];
    let mut flex = BTreeMap::new();
    flex.insert(
        "x".to_string(),
        EnumeratedShapes {
            shapes: vec![vec![1, 4, 2, 1], vec![1, 4, 4, 1]],
        },
    );
    flex.insert(
        "y".to_string(),
        EnumeratedShapes {
            shapes: vec![vec![1, 2, 4, 1], vec![1, 4, 4, 1]],
        },
    );
    let mut syms = BTreeMap::new();
    syms.insert("x".to_string(), vec![None, None, Some("s".into()), None]);
    for n in ["t", "y"] {
        syms.insert(n.to_string(), vec![None, Some("s".into()), None, None]);
    }
    let compiled = pkg_for("baked", &b, &inputs, &outputs, &fin, &flex, &syms);

    let x4: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let xb = fx(&x4);
    let r = predict_at(
        &compiled,
        &[mil_infer::Input {
            name: "x",
            shape: &[1, 4, 4, 1],
            data: &xb,
            dtype: DType::Fp16,
        }],
    );
    eprintln!("s=4 with baked reshape const: {r:?}");
}

/// C. Shape-generic reshape: shape arg computed at runtime as
/// concat([1], seq, [4, 1]) where `seq` is an int32[1] graph input.
/// Output declared at default (1,2,4,1).
#[test]
#[ignore = "manual probe"]
fn enum_runtime_reshape() {
    let mut b = Block::new();
    let t = b.transpose("x", &[0, 2, 1, 3], &[1, 2, 4, 1], "t");
    // shape = concat([1], seq, [4,1]) → int32[4]
    let one = b.konst_i32("one", &[1]);
    let tail = b.konst_i32("tail", &[4, 1]);
    let ax = b.konst_scalar_i32("ax", 0);
    let il = b.konst_bool("il", false);
    let shp = b.o1(
        "concat",
        vec![
            ("values".into(), bind_many(&[&one, "seq", &tail])),
            ("axis".into(), bind(&ax).1),
            ("interleave".into(), bind(&il).1),
        ],
        "shp",
        tt(DType::Int32, &[4]),
    );
    let r = b.o1(
        "reshape",
        vec![("x".into(), bind(&t).1), ("shape".into(), bind(&shp).1)],
        "y",
        f16(&[1, 2, 4, 1]),
    );
    b.outputs = vec![r];
    let inputs = [
        Feature {
            name: "x".into(),
            shape: vec![1, 4, 2, 1],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "seq".into(),
            shape: vec![1],
            dtype: DType::Int32,
            is_state: false,
        },
    ];
    let outputs = [Feature {
        name: "y".into(),
        shape: vec![1, 2, 4, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let fin = [
        NVT {
            name: "x".into(),
            ty: f16(&[1, 4, 2, 1]),
        },
        NVT {
            name: "seq".into(),
            ty: tt(DType::Int32, &[1]),
        },
    ];
    let mut flex = BTreeMap::new();
    flex.insert(
        "x".to_string(),
        EnumeratedShapes {
            shapes: vec![vec![1, 4, 2, 1], vec![1, 4, 4, 1]],
        },
    );
    flex.insert(
        "y".to_string(),
        EnumeratedShapes {
            shapes: vec![vec![1, 2, 4, 1], vec![1, 4, 4, 1]],
        },
    );
    let mut syms = BTreeMap::new();
    syms.insert("x".to_string(), vec![None, None, Some("s".into()), None]);
    for n in ["t", "y"] {
        syms.insert(n.to_string(), vec![None, Some("s".into()), None, None]);
    }
    let compiled = pkg_for("rt", &b, &inputs, &outputs, &fin, &flex, &syms);

    for s in [2i64, 4] {
        let xv: Vec<f32> = (0..4 * s).map(|i| i as f32).collect();
        let xb = fx(&xv);
        let sq: Vec<u8> = (s as i32).to_le_bytes().to_vec();
        let r = predict_at(
            &compiled,
            &[
                mil_infer::Input {
                    name: "x",
                    shape: &[1, 4, s, 1],
                    data: &xb,
                    dtype: DType::Fp16,
                },
                mil_infer::Input {
                    name: "seq",
                    shape: &[1],
                    data: &sq,
                    dtype: DType::Int32,
                },
            ],
        );
        eprintln!("s={s}: {r:?}");
        if let Ok(got) = &r {
            // transpose(1,4,s,1)->(1,s,4,1): y[s_i, c] = x[c, s_i]
            assert_eq!(got.len(), (4 * s) as usize);
            for si in 0..s as usize {
                for c in 0..4usize {
                    let want = xv[c * s as usize + si];
                    let g = got[si * 4 + c];
                    assert_eq!(g, want, "s={s} si={si} c={c}");
                }
            }
        }
        assert!(r.is_ok(), "predict at s={s} failed: {r:?}");
    }
}

/// D. The builder's flex pieces in isolation: int32 `mul` (g*seq),
/// open-end slice (end_mask + i32::MAX), tile, and a reshape whose
/// target is concat([1], gseq, [4, 1]). If the real flex model fails
/// even at its default seq, one of these ops is the culprit.
#[test]
#[ignore = "manual probe"]
fn enum_flex_pieces() {
    let mut b = Block::new();
    // gseq = seq * 2 (int32 mul)
    let gk = b.konst_i32("gk", &[2]);
    let gseq = b.o1(
        "mul",
        vec![("x".into(), bind("seq").1), ("y".into(), bind(&gk).1)],
        "gseq",
        tt(DType::Int32, &[1]),
    );
    // open-end slice on dim 2: y = x[:, :, 0:MAX, :]
    let b0 = b.konst_i32("beg", &[0, 0, 0, 0]);
    let e0 = b.konst_i32("end", &[1, 4, i32::MAX, 1]);
    let st = b.konst_i32("stride", &[1, 1, 1, 1]);
    let bm = b.op(
        "const",
        vec![],
        vec![("bmk", tt(DType::Bool, &[4]))],
        vec![(
            "val".into(),
            mil_spec::Value::bools(&[false, false, false, false]),
        )],
    )[0]
    .clone();
    let em = b.op(
        "const",
        vec![],
        vec![("emk", tt(DType::Bool, &[4]))],
        vec![(
            "val".into(),
            mil_spec::Value::bools(&[false, false, true, false]),
        )],
    )[0]
    .clone();
    let sm = b.op(
        "const",
        vec![],
        vec![("smk", tt(DType::Bool, &[4]))],
        vec![(
            "val".into(),
            mil_spec::Value::bools(&[false, false, false, false]),
        )],
    )[0]
    .clone();
    let sl = b.o1(
        "slice_by_index",
        vec![
            ("x".into(), bind("x").1),
            ("begin".into(), bind(&b0).1),
            ("end".into(), bind(&e0).1),
            ("stride".into(), bind(&st).1),
            ("begin_mask".into(), bind(&bm).1),
            ("end_mask".into(), bind(&em).1),
            ("squeeze_mask".into(), bind(&sm).1),
        ],
        "sl",
        f16(&[1, 4, 2, 1]),
    );
    // tile the slice (1,4,s,1) → (1,8,s,1)? — reps are fixed, no seq.
    let reps = b.konst_i32("reps", &[1, 2, 1, 1]);
    let tl = b.o1(
        "tile",
        vec![("x".into(), bind(&sl).1), ("reps".into(), bind(&reps).1)],
        "tl",
        f16(&[1, 8, 2, 1]),
    );
    // shape = concat([1], gseq, [4, 1]) → reshape (1,8,s,1)→(1,2s,4,1)
    let one = b.konst_i32("one", &[1]);
    let tail = b.konst_i32("tail", &[4, 1]);
    let ax = b.konst_scalar_i32("ax", 0);
    let il = b.konst_bool("il", false);
    let shp = b.o1(
        "concat",
        vec![
            ("values".into(), bind_many(&[&one, &gseq, &tail])),
            ("axis".into(), bind(&ax).1),
            ("interleave".into(), bind(&il).1),
        ],
        "shp",
        tt(DType::Int32, &[4]),
    );
    let r = b.o1(
        "reshape",
        vec![("x".into(), bind(&tl).1), ("shape".into(), bind(&shp).1)],
        "y",
        f16(&[1, 4, 4, 1]),
    );
    b.outputs = vec![r];

    let inputs = [
        Feature {
            name: "x".into(),
            shape: vec![1, 4, 2, 1],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "seq".into(),
            shape: vec![1],
            dtype: DType::Int32,
            is_state: false,
        },
    ];
    let outputs = [Feature {
        name: "y".into(),
        shape: vec![1, 4, 4, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let fin = [
        NVT {
            name: "x".into(),
            ty: f16(&[1, 4, 2, 1]),
        },
        NVT {
            name: "seq".into(),
            ty: tt(DType::Int32, &[1]),
        },
    ];
    let mut flex = BTreeMap::new();
    flex.insert(
        "x".to_string(),
        EnumeratedShapes {
            shapes: vec![vec![1, 4, 2, 1], vec![1, 4, 4, 1]],
        },
    );
    flex.insert(
        "y".to_string(),
        EnumeratedShapes {
            shapes: vec![vec![1, 4, 4, 1], vec![1, 8, 4, 1]],
        },
    );
    let mut syms = BTreeMap::new();
    syms.insert("x".to_string(), vec![None, None, Some("s".into()), None]);
    for (n, sym) in [
        ("sl", vec![None, None, Some("s".into()), None]),
        ("tl", vec![None, None, Some("s".into()), None]),
        ("y", vec![None, Some("s".into()), None, None]),
    ] {
        syms.insert(n.to_string(), sym);
    }
    let compiled = pkg_for("fp", &b, &inputs, &outputs, &fin, &flex, &syms);

    for s in [2i64, 4] {
        let xv: Vec<f32> = (0..4 * s).map(|i| i as f32).collect();
        let xb = fx(&xv);
        let sq: Vec<u8> = (s as i32).to_le_bytes().to_vec();
        let r = predict_at(
            &compiled,
            &[
                mil_infer::Input {
                    name: "x",
                    shape: &[1, 4, s, 1],
                    data: &xb,
                    dtype: DType::Fp16,
                },
                mil_infer::Input {
                    name: "seq",
                    shape: &[1],
                    data: &sq,
                    dtype: DType::Int32,
                },
            ],
        );
        eprintln!("flex-pieces s={s}: {r:?}");
        assert!(r.is_ok(), "predict at s={s} failed: {r:?}");
        let got = r.unwrap();
        // y = reshape(tile(x,[1,2,1,1])) → (1,2s,4,1): flat index i
        // reads tl[i/s, i%s] = x[(i/s)%4, i%s].
        assert_eq!(got.len(), (8 * s) as usize);
        for (i, g) in got.iter().enumerate() {
            let want = xv[(i / s as usize % 4) * s as usize + i % s as usize];
            assert_eq!(*g, want, "s={s} i={i}");
        }
    }
}

/// E. Stateful slice_update in an enumerated model — the piece the
/// elementwise/reshape probes don't cover. `end[2] = pos + seq` with
/// `seq` a runtime input; read the row back and add it to the output.
#[test]
#[ignore = "manual probe"]
fn enum_stateful_slice_update() {
    let mut b = Block::new();
    // state (1,1,8,1); write x (1,1,s,1) at cols [pos, pos+seq)
    let kv = b.read_state("kv", &[1, 1, 8, 1], "kv0");
    // beg = concat([0],[0],[pos],[0]); end = concat([1],[1],[pos+seq],[1])
    let z = b.konst_i32("z", &[0]);
    let one = b.konst_i32("one", &[1]);
    let p2 = b.o1(
        "add",
        vec![("x".into(), bind("pos").1), ("y".into(), bind("seq").1)],
        "p2",
        tt(DType::Int32, &[1]),
    );
    let ax = b.konst_scalar_i32("ax", 0);
    let il = b.konst_bool("il", false);
    let beg = b.o1(
        "concat",
        vec![
            ("values".into(), bind_many(&[&z, &z, "pos", &z])),
            ("axis".into(), bind(&ax).1),
            ("interleave".into(), bind(&il).1),
        ],
        "beg",
        tt(DType::Int32, &[4]),
    );
    let ax2 = b.konst_scalar_i32("ax2", 0);
    let il2 = b.konst_bool("il2", false);
    let end = b.o1(
        "concat",
        vec![
            ("values".into(), bind_many(&[&one, &one, &p2, &one])),
            ("axis".into(), bind(&ax2).1),
            ("interleave".into(), bind(&il2).1),
        ],
        "end",
        tt(DType::Int32, &[4]),
    );
    let st = b.konst_i32("st", &[1, 1, 1, 1]);
    let mkc = |b: &mut Block, n: &str| {
        b.op(
            "const",
            vec![],
            vec![(n, tt(DType::Bool, &[4]))],
            vec![(
                "val".into(),
                mil_spec::Value::bools(&[false, false, false, false]),
            )],
        )[0]
        .clone()
    };
    let bm = mkc(&mut b, "bm");
    let em = mkc(&mut b, "em");
    let sm = mkc(&mut b, "sm");
    let su = b.o1(
        "slice_update",
        vec![
            ("x".into(), bind(&kv).1),
            ("update".into(), bind("x").1),
            ("begin".into(), bind(&beg).1),
            ("end".into(), bind(&end).1),
            ("stride".into(), bind(&st).1),
            ("begin_mask".into(), bind(&bm).1),
            ("end_mask".into(), bind(&em).1),
            ("squeeze_mask".into(), bind(&sm).1),
        ],
        "kv1",
        f16(&[1, 1, 8, 1]),
    );
    b.write_state("kv", &su);
    // output = sum over the state cols actually written
    let axes = b.konst_i32("axes", &[2]);
    let kd = b.konst_bool("kd", false);
    let y = b.o1(
        "reduce_sum",
        vec![
            ("x".into(), bind(&su).1),
            ("axes".into(), bind(&axes).1),
            ("keep_dims".into(), bind(&kd).1),
        ],
        "y",
        f16(&[1, 1, 1]),
    );
    b.outputs = vec![y];

    let inputs = [
        Feature {
            name: "x".into(),
            shape: vec![1, 1, 2, 1],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "pos".into(),
            shape: vec![1],
            dtype: DType::Int32,
            is_state: false,
        },
        Feature {
            name: "seq".into(),
            shape: vec![1],
            dtype: DType::Int32,
            is_state: false,
        },
    ];
    let outputs = [Feature {
        name: "y".into(),
        shape: vec![1, 1, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let states = [Feature {
        name: "kv".into(),
        shape: vec![1, 1, 8, 1],
        dtype: DType::Fp16,
        is_state: true,
    }];
    let fin = [
        NVT {
            name: "x".into(),
            ty: f16(&[1, 1, 2, 1]),
        },
        NVT {
            name: "pos".into(),
            ty: tt(DType::Int32, &[1]),
        },
        NVT {
            name: "seq".into(),
            ty: tt(DType::Int32, &[1]),
        },
        NVT {
            name: "kv".into(),
            ty: ValueType::State(TensorType::f16(&[1, 1, 8, 1])),
        },
    ];
    let mut flex = BTreeMap::new();
    flex.insert(
        "x".to_string(),
        EnumeratedShapes {
            shapes: vec![vec![1, 1, 2, 1], vec![1, 1, 4, 1]],
        },
    );
    let mut syms = BTreeMap::new();
    syms.insert("x".to_string(), vec![None, None, Some("s".into()), None]);
    let compiled = {
        let dir = std::env::temp_dir().join(format!("esprobe_su_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let spec = mil_spec::encode_model_flex(
            &inputs,
            &outputs,
            &states,
            &b,
            &fin,
            &ModelMeta::new(10, "CoreML9"),
            &flex,
            &syms,
        );
        let pkg = dir.join("m.mlpackage");
        mil_spec::write_mlpackage(&pkg, &spec, None).unwrap();
        match mil_compile::compile(&pkg, &dir.join("c")) {
            Ok(c) => c.path,
            Err(e) => panic!("[su] compile failed: {e}"),
        }
    };

    let m = mil_infer::Model::load(&compiled, mil_infer::ComputeUnits::All).unwrap();
    for s in [2i64, 4] {
        let xv: Vec<f32> = (0..s).map(|i| i as f32 + 1.0).collect();
        let xb = fx(&xv);
        let z4: Vec<u8> = 0i32.to_le_bytes().to_vec();
        let sq: Vec<u8> = (s as i32).to_le_bytes().to_vec();
        let st = m.new_state().unwrap();
        let r = m
            .predict_with_state(
                Some(&st),
                &[
                    mil_infer::Input {
                        name: "x",
                        shape: &[1, 1, s, 1],
                        data: &xb,
                        dtype: DType::Fp16,
                    },
                    mil_infer::Input {
                        name: "pos",
                        shape: &[1],
                        data: &z4,
                        dtype: DType::Int32,
                    },
                    mil_infer::Input {
                        name: "seq",
                        shape: &[1],
                        data: &sq,
                        dtype: DType::Int32,
                    },
                ],
            )
            .map(|p| p.outputs[0].values())
            .map_err(|e| e.to_string());
        eprintln!("stateful s={s}: {r:?}");
        if let Ok(g) = &r {
            let want: f32 = xv.iter().sum();
            assert_eq!(g.len(), 1);
            assert_eq!(g[0], want, "s={s}");
        }
        assert!(r.is_ok(), "predict at s={s} failed: {r:?}");
    }
}

/// F. conv + matmul with a symbolic seq dim — the ops the decoder
/// actually runs. If these can't take unknown dims, EnumeratedShapes
/// can't help this architecture at all.
#[test]
#[ignore = "manual probe"]
fn enum_conv_matmul() {
    let mut b = Block::new();
    // conv1x1 weight (4,4,1,1) identity-ish
    let wv: Vec<f32> = (0..16)
        .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
        .collect();
    let w = b.op(
        "const",
        vec![],
        vec![("w", f16(&[4, 4, 1, 1]))],
        vec![("val".into(), mil_spec::Value::f16s(&[4, 4, 1, 1], &wv))],
    )[0]
    .clone();
    let c = b.conv1x1_s("x", &w, None, 4, 2, "cv");
    // matmul over the seq dim, no baked const: transpose x to
    // (1,4,1,s) then (1,4,1,s) @ (1,4,s,1) → (1,4,1,1) — x·x per channel.
    let t = b.transpose(&c, &[0, 1, 3, 2], &[1, 4, 1, 2], "t");
    let sc = b.matmul(&t, &c, false, &[1, 4, 1, 1], "sc");
    let sm = b.softmax(&sc, 1, &[1, 4, 1, 1], "y", false);
    b.outputs = vec![sm];

    let inputs = [Feature {
        name: "x".into(),
        shape: vec![1, 4, 2, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let outputs = [Feature {
        name: "y".into(),
        shape: vec![1, 4, 1, 1],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let fin = [NVT {
        name: "x".into(),
        ty: f16(&[1, 4, 2, 1]),
    }];
    let mut flex = BTreeMap::new();
    flex.insert(
        "x".to_string(),
        EnumeratedShapes {
            shapes: vec![vec![1, 4, 2, 1], vec![1, 4, 4, 1]],
        },
    );
    let mut syms = BTreeMap::new();
    for n in ["x", "cv", "t"] {
        syms.insert(n.to_string(), vec![None, None, Some("s".into()), None]);
    }
    syms.insert("t".to_string(), vec![None, None, None, Some("s".into())]);
    let compiled = pkg_for("cm", &b, &inputs, &outputs, &fin, &flex, &syms);

    for s in [2i64, 4] {
        let xv: Vec<f32> = (0..4 * s).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let xb = fx(&xv);
        let r = predict_at(
            &compiled,
            &[mil_infer::Input {
                name: "x",
                shape: &[1, 4, s, 1],
                data: &xb,
                dtype: DType::Fp16,
            }],
        );
        eprintln!("conv/mm s={s}: {r:?}");
        assert!(r.is_ok(), "predict at s={s} failed: {r:?}");
    }
}

/// G. Replicate `l0_sc`: matmul(qg, k_full^T) where qg has a symbolic
/// seq dim and k_full is a slice_by_index of the kv state with
/// `transpose_y`. `mark_y` adds the builder's (possibly over-eager)
/// symbolic dim on k_full's dim1; `ty` toggles transpose_y.
fn sc_probe(tag: &str, mark_y: bool, ty: bool) -> Result<Vec<f32>, String> {
    let mut b = Block::new();
    let kv = b.read_state("kv", &[4, 1, 16, 8], "kv_0");
    let kf = b.slice(&kv, &[0, 0, 0, 0], &[1, 1, 16, 8], &[1, 1, 16, 8], "kfull");
    let one = b.konst_f16("one", 1.0);
    let qg = b.mul("x", &one, &[1, 1, 2, 8], "qg");
    let sc = b.matmul(&qg, &kf, ty, &[1, 1, 2, 16], "sc");
    b.write_state("kv", &kv);
    b.outputs = vec![sc];

    let inputs = [Feature {
        name: "x".into(),
        shape: vec![1, 1, 2, 8],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let outputs = [Feature {
        name: "sc".into(),
        shape: vec![1, 1, 2, 16],
        dtype: DType::Fp16,
        is_state: false,
    }];
    let states = [Feature {
        name: "kv".into(),
        shape: vec![4, 1, 16, 8],
        dtype: DType::Fp16,
        is_state: true,
    }];
    let fin = [
        NVT {
            name: "x".into(),
            ty: f16(&[1, 1, 2, 8]),
        },
        NVT {
            name: "kv".into(),
            ty: mil_spec::ValueType::State(mil_spec::TensorType::f16(&[4, 1, 16, 8])),
        },
    ];
    let mut flex = BTreeMap::new();
    flex.insert(
        "x".to_string(),
        EnumeratedShapes {
            shapes: vec![vec![1, 1, 2, 8], vec![1, 1, 8, 8]],
        },
    );
    flex.insert(
        "sc".to_string(),
        EnumeratedShapes {
            shapes: vec![vec![1, 1, 2, 16], vec![1, 1, 8, 16]],
        },
    );
    let mut syms = BTreeMap::new();
    syms.insert("x".to_string(), vec![None, None, Some("s".into()), None]);
    syms.insert("qg".to_string(), vec![None, None, Some("s".into()), None]);
    syms.insert("sc".to_string(), vec![None, None, Some("s".into()), None]);
    if mark_y {
        syms.insert(
            "kfull".to_string(),
            vec![None, Some("s".into()), None, None],
        );
    }

    let dir = std::env::temp_dir().join(format!("esprobe_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let spec = mil_spec::encode_model_flex(
        &inputs,
        &outputs,
        &states,
        &b,
        &fin,
        &ModelMeta::new(10, "CoreML9"),
        &flex,
        &syms,
    );
    let pkg = dir.join("m.mlpackage");
    mil_spec::write_mlpackage(&pkg, &spec, None).unwrap();
    let c = mil_compile::compile(&pkg, &dir.join("c")).map_err(|e| format!("cc: {e}"))?;
    let m = mil_infer::Model::load(&c.path, mil_infer::ComputeUnits::All)
        .map_err(|e| format!("load: {e}"))?;
    for s in [2i64, 8] {
        let xv: Vec<f32> = (0..8 * s).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let xb = fx(&xv);
        let st = m.new_state().map_err(|e| format!("state: {e}"))?;
        let r = m
            .predict_with_state(
                Some(&st),
                &[mil_infer::Input {
                    name: "x",
                    shape: &[1, 1, s, 8],
                    data: &xb,
                    dtype: DType::Fp16,
                }],
            )
            .map_err(|e| format!("s={s} predict: {e}"))?;
        let got = r.outputs[0].values();
        // the kv state is zero-initialized so all scores are 0 — this
        // probe only exercises shape handling, not values.
        eprintln!("{tag} s={s}: {} vals", got.len());
        if got.len() != (s * 16) as usize {
            return Err(format!("s={s}: got {} vals want {}", got.len(), s * 16));
        }
    }
    Ok(vec![])
}

#[test]
#[ignore = "manual probe"]
fn enum_sc_matmul() {
    for (mark_y, ty) in [(false, false), (false, true), (true, true)] {
        let tag = format!("sc_{}_{}", mark_y as u8, ty as u8);
        match sc_probe(&tag, mark_y, ty) {
            Ok(_) => eprintln!("{tag}: OK"),
            Err(e) => eprintln!("{tag}: FAIL {e}"),
        }
    }
}
